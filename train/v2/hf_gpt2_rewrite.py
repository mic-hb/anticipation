import json
import re
from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from anticipation.v2.nanochat.flash_attention import flash_attn


@dataclass
class CausalLMOutputLite:
    loss: torch.Tensor | None
    logits: torch.Tensor
    # Optionally: past_key_values / hidden_states / attentions


@dataclass
class GPT2ConfigLite:
    vocab_size: int = 50257
    n_positions: int = 1024
    n_embd: int = 768
    n_layer: int = 12
    n_head: int = 12
    n_inner: int | None = None
    activation_function: str = "gelu_new"
    resid_pdrop: float = 0.1
    embd_pdrop: float = 0.1
    attn_pdrop: float = 0.1
    layer_norm_epsilon: float = 1e-5
    initializer_range: float = 0.02
    use_cache: bool = True
    scale_attn_weights: bool = True
    scale_attn_by_inverse_layer_idx: bool = False
    reorder_and_upcast_attn: bool = False

    # Positional embedding choice (non-HF extension)
    pos_emb: str = "absolute"  # "absolute" or "rope"
    rope_theta: float = 10000.0
    rope_pct: float = 1.0  # fraction of head_dim to rotate
    window_pattern: str = "L"

    @classmethod
    def from_json(cls, path: str):
        d = json.loads(open(path, "r", encoding="utf-8").read())
        # Normalize common aliases across versions
        d.setdefault("n_positions", d.get("max_position_embeddings", 1024))
        d.setdefault("n_embd", d.get("hidden_size", d.get("n_embd", 768)))
        d.setdefault("n_layer", d.get("num_hidden_layers", d.get("n_layer", 12)))
        d.setdefault("n_head", d.get("num_attention_heads", d.get("n_head", 12)))
        return cls(**{k: v for k, v in d.items() if hasattr(cls, k)})


CONV1D_SUFFIXES = (
    ".attn.c_attn.weight",
    ".attn.c_proj.weight",
    ".mlp.c_fc.weight",
    ".mlp.c_proj.weight",
)


def remap_state_dict(
    sd, *, rename_rules=None, drop_prefixes=(), drop_exact=(), conv1d_to_linear=False
):
    """
    sd: dict[str, Tensor]
    rename_rules: list[tuple[pattern, repl]] where pattern is regex
    """
    rename_rules = rename_rules or []
    out = {}

    for k, v in sd.items():
        if k in drop_exact:
            continue
        if any(k.startswith(p) for p in drop_prefixes):
            continue

        new_k = k
        for pat, repl in rename_rules:
            new_k = re.sub(pat, repl, new_k)

        # Optional: convert HF Conv1D weights to nn.Linear convention.
        if conv1d_to_linear and any(new_k.endswith(suf) for suf in CONV1D_SUFFIXES):
            # Conv1D stores [in, out]; nn.Linear wants [out, in]
            v = v.t().contiguous()

        out[new_k] = v

    return out


def load_hf_state_dict_into_model(model, hf_sd, *, strict=False, **remap_kwargs):
    sd = remap_state_dict(hf_sd, **remap_kwargs)
    missing, unexpected = model.load_state_dict(sd, strict=strict)
    return missing, unexpected


def save_hf_style_checkpoint(save_dir, model, config_dict):
    import json
    from pathlib import Path

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    (save_dir / "config.json").write_text(
        json.dumps(config_dict, indent=2, sort_keys=True)
    )
    torch.save(model.state_dict(), save_dir / "pytorch_model.bin")


def rotary_cos_sin(seq_len: int, dim: int, theta: float, device, dtype):
    # dim must be even
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, dim, 2, device=device, dtype=dtype) / dim)
    )
    t = torch.arange(seq_len, device=device, dtype=dtype)
    freqs = torch.einsum("i,j->ij", t, inv_freq)  # [T, dim/2]
    cos = freqs.cos()[None, None, :, :]  # [1,1,T,dim/2]
    sin = freqs.sin()[None, None, :, :]  # [1,1,T,dim/2]
    return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: [B, H, T, D]; rotate first D_rope = 2 * cos.size(-1)
    d2 = cos.size(-1)
    x1 = x[..., : 2 * d2]
    x2 = x[..., 2 * d2 :]
    x1_even = x1[..., 0::2]
    x1_odd = x1[..., 1::2]
    rot_even = x1_even * cos - x1_odd * sin
    rot_odd = x1_even * sin + x1_odd * cos
    x1_rot = torch.stack([rot_even, rot_odd], dim=-1).flatten(-2)
    return torch.cat([x1_rot, x2], dim=-1)


class Conv1D(nn.Module):
    def __init__(self, nf: int, nx: int, *, init_std: float = 0.02):
        super().__init__()
        self.nf = nf
        self.weight = nn.Parameter(torch.empty(nx, nf))
        self.bias = nn.Parameter(torch.zeros(nf))
        nn.init.normal_(self.weight, std=init_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        return x.view(size_out)


class GPT2AttentionLite(nn.Module):
    def __init__(
        self, config: GPT2ConfigLite, layer_idx: int, ve_gate_channels: int = 32
    ):
        super().__init__()
        self.embed_dim = config.n_embd
        self.num_heads: int = config.n_head
        self.head_dim = self.embed_dim // self.num_heads
        assert self.head_dim * self.num_heads == self.embed_dim

        self.scale_attn_weights = config.scale_attn_weights
        self.scale_attn_by_inverse_layer_idx = config.scale_attn_by_inverse_layer_idx
        self.layer_idx = layer_idx

        self.c_attn = Conv1D(
            3 * self.embed_dim, self.embed_dim, init_std=config.initializer_range
        )
        self.c_proj = Conv1D(
            self.embed_dim, self.embed_dim, init_std=config.initializer_range
        )
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

        # Optional legacy buffers (keep for checkpoint compatibility)
        max_pos = config.n_positions
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(max_pos, max_pos, dtype=torch.uint8)).view(
                1, 1, max_pos, max_pos
            ),
        )
        self.register_buffer("masked_bias", torch.tensor(-1e4))

        self.use_rope = config.pos_emb == "rope"
        self.rope_theta = config.rope_theta
        self.rope_pct = config.rope_pct

        self.ve_gate_channels = ve_gate_channels
        self.ve_gate = (
            nn.Linear(self.ve_gate_channels, self.num_heads, bias=False)
            if has_ve(layer_idx, config.n_layer)
            else None
        )

    def _split_heads(self, x):
        b, t, c = x.shape
        x = x.view(b, t, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)  # [B,H,T,D]

    def _merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        b, t, h, d = x.shape
        return x.view(b, t, h * d)

    def forward(
        self,
        x,
        ve,
        window_size,
        attention_mask=None,
    ):
        # note: not using the attention mask because we always use causal attention
        # and that is controlled by a single variable in flash_attn
        b, t, _ = x.shape
        qkv = self.c_attn(x)  # [B,T,3*C]
        q, k, v = qkv.split(self.embed_dim, dim=2)
        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)  # (b, n_heads, t, d)

        if ve is not None:
            ve = ve.view(b, t, self.num_heads, self.head_dim)
            gate = 2 * torch.sigmoid(
                self.ve_gate(x[..., : self.ve_gate_channels])
            )  # (B, T, n_head), range (0, 2)
            _to_add = gate.unsqueeze(-1) * ve
            _to_add = _to_add.permute(0, 2, 1, 3).contiguous()
            v = v + _to_add

        if self.use_rope:
            rope_dim = int(self.rope_pct * self.head_dim)
            rope_dim = rope_dim - (rope_dim % 2)
            cos, sin = rotary_cos_sin(t, rope_dim, self.rope_theta, x.device, x.dtype)
            q = torch.cat(
                [apply_rope(q[..., :rope_dim], cos, sin), q[..., rope_dim:]], dim=-1
            )
            k = torch.cat(
                [apply_rope(k[..., :rope_dim], cos, sin), k[..., rope_dim:]], dim=-1
            )

        # B,H,T,D -> B,T,H,D
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        # B T H D -> B H T D
        y = y.permute(0, 2, 1, 3)

        y = self._merge_heads(y)  # [B,T,C]
        y = self.c_proj(y)
        y = self.resid_dropout(y)
        return y


class GPT2MLPLite(nn.Module):
    def __init__(self, config):
        super().__init__()
        inner = config.n_inner or (4 * config.n_embd)
        self.c_fc = Conv1D(inner, config.n_embd, init_std=config.initializer_range)
        self.c_proj = Conv1D(config.n_embd, inner, init_std=config.initializer_range)
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, x):
        x = self.c_fc(x)
        x = torch.nn.functional.gelu(
            x
        )  # keep minimal; HF supports multiple activations
        x = self.c_proj(x)
        return self.dropout(x)


class GPT2BlockLite(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = GPT2AttentionLite(config, layer_idx=layer_idx)
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.mlp = GPT2MLPLite(config)

    def forward(self, x, *, window_size, ve, attention_mask=None):
        x = x + self.attn(
            self.ln_1(x), ve=ve, window_size=window_size, attention_mask=attention_mask
        )
        x = x + self.mlp(self.ln_2(x))
        return x


def has_ve(layer_idx: int, n_layer: int) -> bool:
    """Returns True if GPT layer should have Value Embedding (alternating, last layer always included)."""
    return layer_idx % 2 == (n_layer - 1) % 2


class GPT2ModelLite(nn.Module):
    def __init__(self, config: GPT2ConfigLite):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList(
            [GPT2BlockLite(config, layer_idx=i) for i in range(config.n_layer)]
        )
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.gradient_checkpointing = False
        self.window_sizes = self._compute_window_sizes(config)

        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.full((config.n_layer,), 0.1))
        self.value_embeds = nn.ModuleDict(
            {
                str(i): nn.Embedding(config.vocab_size, config.n_embd)
                for i in range(config.n_layer)
                if has_ve(i, config.n_layer)
            }
        )
        self.reset_parameters()

    def _compute_window_sizes(self, config: GPT2ConfigLite):
        """
        Compute per-layer window sizes for sliding window attention.

        Returns list of (left, right) tuples for FA3's window_size parameter:
        - left: how many tokens before current position to attend to (-1 = unlimited)
        - right: how many tokens after current position to attend to (0 for causal)

        Pattern string is tiled across layers. Final layer always gets L (full context).
        Characters: L=long (full context), S=short (half context)
        """
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern), (
            f"Invalid window_pattern: {pattern}. Use only S and L."
        )
        # Map characters to window sizes
        long_window = config.n_positions
        short_window = long_window // 2
        char_to_window = {
            "L": (long_window, 0),
            "S": (short_window, 0),
        }
        # Tile pattern across layers
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        # Final layer always gets full context
        window_sizes[-1] = (long_window, 0)
        return window_sizes

    def reset_parameters(self):
        # Mirrors GPT2PreTrainedModel _init_weights: normal_(std=initializer_range) and LN=1/bias=0.
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding, Conv1D)):
                if hasattr(m, "weight") and m.weight is not None:
                    nn.init.normal_(m.weight, std=self.config.initializer_range)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, input_ids, attention_mask=None):
        b, t = input_ids.shape
        pos = torch.arange(0, t, device=input_ids.device).unsqueeze(0)
        x = self.wte(input_ids) + self.wpe(pos)
        x = self.drop(x)

        # Convert attention_mask (B,T) with 1=keep, 0=mask to additive mask (B,1,1,T) with 0 / -10000.
        attn_bias = None
        if attention_mask is not None:
            m = attention_mask[:, None, None, :].to(dtype=x.dtype)
            attn_bias = (1.0 - m) * -10000.0

        x0 = x
        block: GPT2BlockLite
        for i, block in enumerate(self.h):
            window_size = self.window_sizes[i]
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = (
                self.value_embeds[str(i)](input_ids)
                if str(i) in self.value_embeds
                else None
            )

            if self.gradient_checkpointing and self.training:
                # Explicitly set use_reentrant per PyTorch recommendation.
                x = checkpoint(
                    lambda _x: block(
                        _x, ve=ve, window_size=window_size, attention_mask=attn_bias
                    ),
                    x,
                    use_reentrant=False,
                )
            else:
                x = block(x, ve=ve, window_size=window_size, attention_mask=attn_bias)

        return self.ln_f(x)


class GPT2LMHeadModelLite(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = GPT2ModelLite(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # Tie weights like HF tie_weights: lm_head.weight points to wte.weight.
        self.lm_head.weight = self.transformer.wte.weight

    def gradient_checkpointing_enable(self):
        self.transformer.gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        self.transformer.gradient_checkpointing = False

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        return_dict=True,
        **kwargs,
    ):
        hidden = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.lm_head(hidden)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = nn.CrossEntropyLoss()(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )

        if return_dict:
            return CausalLMOutputLite(loss=loss, logits=logits)
        return (loss, logits) if loss is not None else (logits,)

    @property
    def device(self) -> torch.device:
        return self.wte.weight.device
