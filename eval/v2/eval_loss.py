import argparse
from typing import Any, Optional
from pathlib import Path
from pprint import pprint
from datetime import datetime, timezone
import json

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import plotly.graph_objects as go
from transformers import PretrainedConfig, GPT2LMHeadModel, GPT2Config

from anticipation.v2.config import AnticipationV2Settings, REPO_ROOT
from train.v2.dataset_utils import PreTokenizedDataset
from train.v2.hf_gpt2_rewrite import GPT2LMHeadModelLite


SAVE_RESULTS_TO = Path(__file__).parent / "results"
SAVE_RESULTS_TO.mkdir(exist_ok=True)


def get_time_as_string() -> str:
    dt = datetime.now(timezone.utc)
    # e.g. 2025_01_01_13_31_12
    return dt.strftime("%Y_%m_%d_%H_%M_%S")


def get_scaling_params_and_model_settings(gpt: GPT2LMHeadModel) -> dict[str, int]:
    # https://github.com/karpathy/nanochat/blob/c7ba25214276d165eeefca7cb2060587975db189/nanochat/gpt.py#L319
    # Count each group separately (mirrors the grouping in setup_optimizers)
    wte = sum(p.numel() for p in gpt.transformer.wte.parameters())
    lm_head = sum(p.numel() for p in gpt.lm_head.parameters())
    transformer_matrices = sum(p.numel() for p in gpt.transformer.h.parameters())

    # these two are unique to us, nanochat removes these - we will keep for now. Have
    # not yet ablated them to see if removing them improves for us
    wpe = sum(p.numel() for p in gpt.transformer.wpe.parameters())
    ln_f = sum(p.numel() for p in gpt.transformer.ln_f.parameters())

    total = wte + lm_head + transformer_matrices + ln_f + wpe

    # indeed using weight tie
    # if using weight tying the params are shared
    total -= wte

    actual = sum(p.numel() for p in gpt.parameters())
    assert total == actual, (
        f"Parameter count mismatch. Expected: {total:,}, Actual: {actual:,}."
    )

    model_config = gpt.config
    return {
        "wte": wte,
        "lm_head": lm_head,
        "transformer_matrices": transformer_matrices,
        "total": total,
        **model_config.to_dict(),
    }


def log_loss(
    model: GPT2LMHeadModel,
    dataset: PreTokenizedDataset,
    settings: AnticipationV2Settings,
    save_artfacts_to_enclosing_folder: Path,
    subsample_ratio: int = 10,
    context_limit: Optional[int] = 1024,
    cut_prefix: int = 0,
):
    total_params = sum(p.numel() for p in model.parameters())
    model_id = settings.md5_hash()
    # save_path = save_artfacts_to_enclosing_folder / f"ppl_plot.html"

    if context_limit is not None:
        assert settings.context_size % context_limit == 0

    total_samples = len(dataset)
    ce = torch.empty(0).cpu()
    ce_ticks = torch.empty(0).cpu()
    ce_events = torch.empty(0).cpu()
    total_ticks = 0

    ce_sum = None
    ce_count = None

    for sample_idx in tqdm(range(total_samples)):
        if sample_idx % subsample_ratio != 0:
            continue

        tokens = dataset[sample_idx]["input_ids"]
        tokens = tokens.unsqueeze(0).cuda()

        with torch.no_grad():
            # (s, vocab)
            logits = model(tokens).logits[0]

            # per token nll
            targets = tokens[0, 1:]
            curr_ce = F.cross_entropy(logits[:-1], targets, reduction="none")

            curr_ce = curr_ce[cut_prefix:context_limit]
            targets = targets[cut_prefix:context_limit]

            if ce_sum is None:
                ce_sum = torch.zeros_like(curr_ce, dtype=torch.float64, device="cpu")
                ce_count = torch.zeros_like(curr_ce, dtype=torch.long, device="cpu")

            n = min(len(curr_ce), len(ce_sum))
            ce_sum[:n] += curr_ce[:n].detach().cpu().to(torch.float64)
            ce_count[:n] += 1

            # do not measure for controls, those are given by us
            controls = targets >= settings.vocab.SPECIAL_OFFSET

            # isolate the tick cross entropy
            ticks = targets == settings.vocab.TICK
            tick_ce = curr_ce[ticks]
            total_ticks += tick_ce.numel()

            # anything that is not a tick and is not a control is a triple
            event_ce = curr_ce[(~ticks) & (~controls)]
            if event_ce.shape[0] % 3 != 0:
                # truncate incomplete remaining triple if needed
                event_ce = event_ce[: -1 * (event_ce.shape[0] % 3)]

            assert event_ce.shape[0] % 3 == 0

            ce = torch.cat([ce, curr_ce.cpu()])  # everything
            ce_events = torch.cat([ce_events, event_ce.cpu()])  # triples
            ce_ticks = torch.cat([ce_ticks, tick_ce.cpu()])  # ticks

    d_rounding = 4
    res = {}
    res["loss"] = np.round(ce.mean().item(), d_rounding)

    # PPL(e)
    res["event_ppl"] = np.round(np.exp(3 * ce_events.mean().item()), d_rounding)

    # consider the total log loss of onsets and ticks, average over number
    # of events
    num_events = ce_events.shape[0] // 3
    # PPL(t)
    res["onset_ppl"] = np.round(
        np.exp((ce_events[0::3].sum() + ce_ticks.sum()).item() / num_events), d_rounding
    )
    res["onset_ppl_no_ticks"] = np.round(
        np.exp((ce_events[0::3].sum()).item() / num_events), d_rounding
    )
    # PPL(d)
    res["dur_ppl"] = np.round(
        np.exp((ce_events[1::3].sum() + ce_ticks.sum()).item() / num_events), d_rounding
    )
    res["dur_ppl_no_ticks"] = np.round(
        np.exp((ce_events[1::3].sum()).item() / num_events), d_rounding
    )
    # PPL(n)
    res["note_ppl"] = np.round(
        np.exp((ce_events[2::3].sum() + ce_ticks.sum()).item() / num_events), d_rounding
    )
    res["note_ppl_no_ticks"] = np.round(
        np.exp((ce_events[2::3].sum()).item() / num_events), d_rounding
    )

    # ticks only
    res["tick_ppl"] = np.round(np.exp(ce_ticks.mean().item()), d_rounding)

    if (context_limit is None or context_limit == settings.context_size) and (
        cut_prefix == 0
    ):
        # WARNING: hard-coded lakh dataset test split
        # the test split is `f`, 560.98 hours total in there.
        num_seconds_in_test_split = 560.98 * 3600
    else:
        # approximate the number of seconds seen in the limited split
        num_seconds_in_test_split = (
            total_ticks * settings.tick_token_every_n_ticks
        ) / settings.time_resolution

        # remove the effect of subsampling because we've measured it directly for what
        # the model has seen
        num_seconds_in_test_split *= subsample_ratio

    res["bps"] = (
        subsample_ratio
        * ce_events.mean().item()
        * np.log2(np.e)
        * (len(ce) / (num_seconds_in_test_split))
    )

    res["total_params"] = total_params
    res["total_params_m"] = total_params / (1_000_000)
    res["subsample"] = subsample_ratio
    res["cut_prefix"] = cut_prefix
    res["model_id"] = model_id

    valid = ce_count > 0
    avg_ce = torch.zeros_like(ce_sum)
    avg_ce[valid] = ce_sum[valid] / ce_count[valid]
    avg_ppl = torch.exp(avg_ce[valid])
    positions = torch.arange(len(avg_ppl)).cpu().numpy()

    # plot it
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=positions,
            y=avg_ppl.cpu().numpy(),
            mode="lines",
            name="Average PPL",
        )
    )
    fig.update_layout(
        title=f"Average Perplexity vs Sequence Position (model ID={model_id})",
        xaxis_title="Sequence position",
        yaxis_title="Perplexity (log scale)",
        template="plotly_white",
    )
    fig.update_layout(
        yaxis_type="log",
    )
    stab_idx = find_stabilization_variance(avg_ppl.cpu().numpy())
    if stab_idx is not None:
        fig.add_vline(
            x=stab_idx,
            line_dash="dash",
            line_color="red",
            annotation_text=f"stabilization @ {stab_idx}",
        )
    fig.write_html(str(save_artfacts_to_enclosing_folder / "ppl_plot.html"))

    # save_path_tensor = SAVE_RESULTS_TO / f"{model_id}.pt"
    # torch.save(ce, save_path_tensor)

    # remove np. types
    res_casted = {}
    for k, v in res.items():
        if isinstance(v, np.float64):
            v = float(v)

        res_casted[k] = v

    res_casted["time"] = get_time_as_string()
    res_casted["checkpoint_folder"] = str(save_artfacts_to_enclosing_folder)
    res_casted["checkpoint_folder_last_part"] = str(
        save_artfacts_to_enclosing_folder.parts[-1]
    )

    p = str(save_artfacts_to_enclosing_folder.parts[-1])
    s_n, s_k = p.split("_")

    # midtraining parameters
    n = int(s_n)
    k = int(s_k)
    res_casted["n"] = n
    res_casted["k"] = k

    # get network parts and how many parameters they have
    scaling_params = get_scaling_params_and_model_settings(model)

    for k, v in scaling_params.items():
        res_casted["model_property_" + k] = v

    s = json.dumps(res_casted, sort_keys=False, indent=4)
    (save_artfacts_to_enclosing_folder / "results.json").write_text(s)

    return res_casted


def find_stabilization_variance(y, window=50, var_threshold=5e-3):
    # sliding window over the context to see when variance drops below
    # some value
    y = np.asarray(y)

    for i in range(len(y) - window):
        if np.var(y[i : i + window]) < var_threshold:
            return i

    return None


def format_result_row(res: dict[str, Any]) -> dict[str, str]:
    formatted = {}
    # name to use in spreadsheet
    keep_and_rename = {
        "total_params_m": "Params(m)",
        "bps": "BPS",
        "tick_ppl": "PPL(tick)",
        "loss": "Test Loss (Lakh)",
        "event_ppl": "PPL(e)",
        "onset_ppl": "PPL(t)",
        "dur_ppl": "PPL(d)",
        "note_ppl": "PPL(n)",
    }
    for k, v in keep_and_rename.items():
        result = res[k]
        if isinstance(result, np.float64):
            result = float(result)
        formatted[v] = result

    return formatted


def get_num_steps(pretrained_checkpoint_path: str, num_gpus: int = 1) -> int:
    # TODO: need to know number of epochs the model has seen
    # number of non-augmented epochs
    step = int(pretrained_checkpoint_path.split("step-")[-1])
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Eval Loss")
    # float
    parser.add_argument("--subsample", type=int, default=10, help="Subsample ratio")
    # path
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model's enclosing path",
    )
    args = parser.parse_args()
    return args


def main() -> None:
    data_dir_giga = "data/tokenized_datasets/giga_midi/6fb2094dfa7c0d16278dfaa4a401e3b8"
    data_dir_lmd = "data/tokenized_datasets/lmd_full/6fb2094dfa7c0d16278dfaa4a401e3b8"
    pretrained_checkpoint_path = "output/slurm_logs/2632/checkpoints/step-100000"

    checkpoint_dir = "/home/mf867/anticipation_isolated/anticipation/output/slurm_logs/232666/checkpoints/step-20000"
    checkpoint_dir = "/home/mf867/anticipation_isolated/anticipation/output/slurm_logs/263233/checkpoints/step-100000"

    checkpoint_dir = (
        "/home/mf867/anticipation/output/checkpoints/midtraining_testing/8203_6659"
    )
    data_dir = "data/tokenized_datasets/lakh_baseline/b0d0dbce322fc3318387b6cc12cf096a"

    args = parse_args()
    checkpoint_dir = args.checkpoint_dir
    subsample = args.subsample

    dataset = PreTokenizedDataset(Path(data_dir) / "test.npy")
    model = GPT2LMHeadModel.from_pretrained(
        checkpoint_dir,
    ).cuda()
    model = model.eval()

    settings_path_name = "settings_" + data_dir.split("/")[-1] + ".json"
    print(settings_path_name)
    settings = AnticipationV2Settings.load_from_disk(
        Path(data_dir) / settings_path_name
    )

    # same settings as in v1 baselines
    context_limit = 1024
    res = log_loss(
        model,
        dataset,
        settings,
        subsample_ratio=subsample,
        context_limit=context_limit,
        save_artfacts_to_enclosing_folder=Path(checkpoint_dir),
    )
    pprint(res)


if __name__ == "__main__":
    """

        PYTHONPATH=. python eval/v2/eval_loss.py

    """
    main()
