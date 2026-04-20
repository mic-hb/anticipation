#from regex import A
import json
import os
import time
import argparse
import logging
import math
import warnings
from pathlib import Path
from collections import deque
from typing import Any


import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, TQDMProgressBar, Callback
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities.rank_zero import rank_zero_info

from transformers import PretrainedConfig, GPT2LMHeadModel, GPT2Config

import wandb

from train.v2.dataset_utils import PreTokenizedDataset
from train.v2.hf_gpt2_rewrite import (
    GPT2ConfigLite,
    GPT2LMHeadModelLite,
    build_model_meta,
    estimate_flops,
    get_num_scaling_params,
    get_scaling_analysis_data,
    print0,
)

import warnings
warnings.filterwarnings(
    "ignore",
    message=".*does not have many workers.*",
    module="lightning.pytorch.trainer.connectors.data_connector",
)
# keep this!!!
torch.set_float32_matmul_precision("high")


class TipFilter(logging.Filter):
    """
    Credit to: https://github.com/Lightning-AI/pytorch-lightning/issues/21294#issuecomment-3410770397
    """

    def filter(self, record) -> bool:
        m = record.getMessage()
        return "💡 Tip:" not in m


logging.getLogger("lightning.pytorch.utilities.rank_zero").addFilter(TipFilter())

class MaxStepProgressBar(TQDMProgressBar):
    def __init__(self):
        super().__init__()
        self._persistent_bar = None

    def init_train_tqdm(self):
        if self._persistent_bar is None:
            bar = super().init_train_tqdm()
            bar.set_description("Training Progress")
            self._persistent_bar = bar

        return self._persistent_bar

    def on_train_epoch_start(self, trainer, pl_module):
        if self._persistent_bar is not None:
            self._persistent_bar.set_description("Training Progress")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        current = trainer.global_step
        total = trainer.max_steps
        self.train_progress_bar.n = current
        self.train_progress_bar.total = total
        self._persistent_bar.refresh()


class Phase1SequenceCheckpoint(L.Callback):
    def __init__(
        self,
        dirpath: str,
        seq_milestones: list[int],
        per_device_batch_size: int,
        grad_accum: int,
    ):
        self.dirpath = dirpath
        self.seq_milestones = sorted(seq_milestones)
        self.per_device_batch_size = per_device_batch_size
        self.grad_accum = grad_accum

        self._next_idx = 0

    def _sequences_seen(self, trainer):
        return (
            trainer.global_step
            * self.per_device_batch_size
            * trainer.world_size
            * self.grad_accum
        )

    def _get_wandb_run_id(self, trainer) -> str | None:
        logger = trainer.logger

        if logger is None:
            return None

        if hasattr(logger, "experiment"):
            exp = logger.experiment
            if exp is not None and hasattr(exp, "id"):
                return exp.id

        if hasattr(logger, "_logger_iterable"):
            for sublogger in logger._logger_iterable:
                if hasattr(sublogger, "experiment"):
                    exp = sublogger.experiment
                    if exp is not None and hasattr(exp, "id"):
                        return exp.id

        return None

    def _write_resume_metadata(
        self,
        trainer,
        pl_module,
        step_dir: Path,
        seq_target: int,
        ckpt_path: Path,
    ) -> None:
        payload = {
            "wandb_run_id": self._get_wandb_run_id(trainer),
            "resume_ckpt": str(ckpt_path),
            "global_step": trainer.global_step,
            "seq_milestone": seq_target,
            "sequences_seen_estimate": self._sequences_seen(trainer),
            "phase": 1,
            "total_optimizer_steps": getattr(pl_module, "total_optimizer_steps", None),
            "steps_ds1": getattr(pl_module, "steps_ds1", None),
        }
        if payload["total_optimizer_steps"] is None:
            raise ValueError("Must save the total optimizer steps. This field was null.")

        with open(step_dir / "resume_info.json", "w") as f:
            json.dump(payload, f, indent=4, sort_keys=True)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_rank != 0:
            return

        # only phase 1
        if getattr(pl_module, "phase_2", False):
            return

        if self._next_idx >= len(self.seq_milestones):
            return

        seen = self._sequences_seen(trainer)
        target = self.seq_milestones[self._next_idx]

        if seen < target:
            return

        # stop when a specific number of sequences is reached
        step_dir = Path(os.path.join(self.dirpath, f"phase1_seq-{target}"))

        ckpt_path = step_dir / "trainer.ckpt"
        trainer.save_checkpoint(ckpt_path, weights_only=False)
        raw_model = pl_module.model if hasattr(pl_module, "model") else pl_module
        raw_model.save_pretrained(
            step_dir, safe_serialization=True, max_shard_size="2GB"
        )
        self._write_resume_metadata(
            trainer=trainer,
            pl_module=pl_module,
            step_dir=Path(step_dir),
            seq_target=target,
            ckpt_path=ckpt_path,
        )
        self._next_idx += 1

class Phase2TopKCheckpoint(ModelCheckpoint):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def on_validation_end(self, trainer, pl_module):
        super().on_validation_end(trainer, pl_module)

    def _save_checkpoint(self, trainer, filepath):
        if trainer.global_rank != 0:
            return

        # override to use HF format instead of .ckpt
        step_dir = filepath.replace(".ckpt", "")
        raw_model = trainer.lightning_module.model if hasattr(trainer.lightning_module, "model") else trainer.lightning_module
        raw_model.save_pretrained(
            step_dir, safe_serialization=True, max_shard_size="2GB"
        )
        print(f"Saved Phase 2 checkpoint to: {step_dir}")


class OverfitStopper(Callback):
    """
    Stop training when validation loss has clearly risen above its best value,
    using smoothing and a consecutive-check requirement to avoid reacting to noise.
    """

    def __init__(
        self,
        monitor: str = "val/loss",
        warmup_checks: int = 5,
        window_size: int = 3,
        overfit_margin: float = 0.01,
        patience_checks: int = 3,
        verbose: bool = True,
        ignore_sanity_checks: bool = True,
    ) -> None:
        super().__init__()
        if warmup_checks < 0:
            raise ValueError("warmup_checks must be >= 0")
        if window_size <= 0:
            raise ValueError("window_size must be > 0")
        if overfit_margin < 0:
            raise ValueError("overfit_margin must be >= 0")
        if patience_checks <= 0:
            raise ValueError("patience_checks must be > 0")

        self.monitor = monitor
        self.warmup_checks = warmup_checks
        self.window_size = window_size
        self.overfit_margin = overfit_margin
        self.patience_checks = patience_checks
        self.verbose = verbose
        self.ignore_sanity_checks = ignore_sanity_checks

        self.best_val = float("inf")
        self.num_checks = 0
        self.bad_check_streak = 0
        self.recent_vals: deque[float] = deque(maxlen=window_size)

    def state_dict(self) -> dict[str, Any]:
        return {
            "best_val": self.best_val,
            "num_checks": self.num_checks,
            "bad_check_streak": self.bad_check_streak,
            "recent_vals": list(self.recent_vals),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.best_val = float(state_dict["best_val"])
        self.num_checks = int(state_dict["num_checks"])
        self.bad_check_streak = int(state_dict["bad_check_streak"])
        self.recent_vals = deque(state_dict["recent_vals"], maxlen=self.window_size)

    def on_validation_epoch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
    ) -> None:
        if self.ignore_sanity_checks and trainer.sanity_checking:
            return

        metric = trainer.callback_metrics.get(self.monitor)
        if metric is None:
            raise RuntimeError(
                f"Monitored metric {self.monitor!r} not found in trainer.callback_metrics. "
                "Make sure it is logged during validation."
            )

        val = float(metric.detach().cpu() if isinstance(metric, torch.Tensor) else metric)
        self.recent_vals.append(val)

        if val < self.best_val:
            self.best_val = val

        smoothed = sum(self.recent_vals) / len(self.recent_vals)

        pl_module.log("overfit/best_val", self.best_val, on_step=False, on_epoch=True, sync_dist=True)
        pl_module.log("overfit/smoothed_val", smoothed, on_step=False, on_epoch=True, sync_dist=True)
        pl_module.log(
            "overfit/excess_over_best",
            smoothed - self.best_val,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        pl_module.log(
            "overfit/num_checks",
            self.num_checks,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        self.num_checks += 1
        if (self.num_checks <= self.warmup_checks):
            pl_module.log(
                "overfit/bad_check_streak",
                float(self.bad_check_streak),
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
            return

        if len(self.recent_vals) < self.window_size:
            pl_module.log(
                "overfit/bad_check_streak",
                float(self.bad_check_streak),
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
            return

        if smoothed > self.best_val + self.overfit_margin:
            self.bad_check_streak += 1
        else:
            self.bad_check_streak = 0

        pl_module.log(
            "overfit/bad_check_streak",
            float(self.bad_check_streak),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        if self.bad_check_streak >= self.patience_checks:
            trainer.should_stop = True
            if self.verbose and trainer.is_global_zero:
                print(
                    f"Stopping due to overfitting: smoothed {self.monitor}={smoothed:.6f} "
                    f"> best {self.best_val:.6f} + margin {self.overfit_margin:.6f} "
                    f"for {self.bad_check_streak} consecutive validation checks."
                )


class FixedRandomSubsetDataset(Dataset):
    """
    Fixed random subset view over another dataset.

    The subset is sampled once at construction time using `seed`, and then
    remains unchanged for the lifetime of the object.
    """

    def __init__(self, dataset: Dataset, subset_size: int, seed: int) -> None:
        dataset_size = len(dataset)

        if subset_size < 0:
            raise ValueError(f"subset_size must be >= 0, got {subset_size}")
        if subset_size > dataset_size:
            raise ValueError(
                f"subset_size={subset_size} exceeds dataset size {dataset_size}"
            )

        self.dataset = dataset
        rng = np.random.default_rng(seed)
        self.indices = rng.choice(dataset_size, size=subset_size, replace=False)

    def __len__(self) -> int:
        return len(self.indices)

    @property
    def num_tokens(self) -> int:
        if hasattr(self.dataset, "data"):
            num_tokens_per_sequence = self.dataset.data.shape[1]
            return len(self) * num_tokens_per_sequence
        raise AttributeError("Underlying dataset does not expose .data")

    def __getitem__(self, idx: int):
        return self.dataset[int(self.indices[idx])]


def maybe_make_fixed_random_subset(
    dataset: Dataset,
    subset_size: int | None,
    seed: int | None,
) -> Dataset:
    if subset_size is None:
        return dataset
    if seed is None:
        raise ValueError("seed must be provided when subset_size is not None")
    return FixedRandomSubsetDataset(dataset, subset_size=subset_size, seed=seed)


def optimizer_steps_per_epoch(
    *,
    num_sequences: int,
    world_size: int,
    per_device_batch_size: int,
    grad_accum_steps: int,
    sampler_drop_last: bool = True,
    loader_drop_last: bool = True,
) -> int:
    """
    Compute optimizer steps per epoch for:
      - DistributedSampler(..., drop_last=sampler_drop_last)
      - DataLoader(..., drop_last=loader_drop_last)
      - Lightning automatic optimization with accumulate_grad_batches=grad_accum_steps
    """

    if world_size <= 0:
        raise ValueError(f"world_size must be positive, got {world_size}")
    if per_device_batch_size <= 0:
        raise ValueError(
            f"per_device_batch_size must be positive, got {per_device_batch_size}"
        )
    if grad_accum_steps <= 0:
        raise ValueError(f"grad_accum_steps must be positive, got {grad_accum_steps}")

    samples_per_rank = (
        num_sequences // world_size
        if sampler_drop_last
        else math.ceil(num_sequences / world_size)
    )

    num_batches_per_rank = (
        samples_per_rank // per_device_batch_size
        if loader_drop_last
        else math.ceil(samples_per_rank / per_device_batch_size)
    )

    if num_batches_per_rank == 0:
        return 0

    return math.ceil(num_batches_per_rank / grad_accum_steps)


class LMModule(L.LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        train_dataset_path: Path,
        val_dataset_path: Path,
        per_device_batch_size: int,
        val_batch_size: int,
        num_workers: int,
        learning_rate: float,
        weight_decay: float,
        total_optimizer_steps: int,
        pin_memory: bool = True,
        sampler_drop_last: bool = True,
        loader_drop_last: bool = True,
        subset_size_train: int | None = None,
        subset_size_val: int | None = None,
        subset_seed_train: int | None = None,
        subset_seed_val: int | None = None,
        warmup_steps: int = 0,
        do_gradient_checkpointing: bool = False,
        resume_training_state_path: str | None = None,
        inherited_global_step: int = 0,
    ) -> None:
        super().__init__()

        self.model = model
        self.model.config.bos_token_id = self.model.config.eos_token_id = 0

        if getattr(model.config, "do_torch_compile", False):
            self.model = torch.compile(self.model, dynamic=False, fullgraph=True)

        self.do_gradient_checkpointing = do_gradient_checkpointing
        if self.do_gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        self.train_dataset_path = Path(train_dataset_path)
        self.val_dataset_path = Path(val_dataset_path)

        self.per_device_batch_size = per_device_batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.total_optimizer_steps = total_optimizer_steps
        self.warmup_steps = warmup_steps
        self.pin_memory = pin_memory
        self.sampler_drop_last = sampler_drop_last
        self.loader_drop_last = loader_drop_last

        self.subset_size_train = subset_size_train
        self.subset_size_val = subset_size_val
        self.subset_seed_train = subset_seed_train
        self.subset_seed_val = subset_seed_val

        self.resume_training_state_path = resume_training_state_path
        self._loaded_training_state = False
        self.inherited_global_step = inherited_global_step
        self.train_dataset: Dataset | None = None

        self.save_hyperparameters(ignore=["model"])

    def setup(self, stage: str | None = None) -> None:
        if stage not in (None, "fit", "validate"):
            return

        if self.train_dataset is None:
            base = PreTokenizedDataset(self.train_dataset_path)
            self.train_dataset = maybe_make_fixed_random_subset(
                base,
                subset_size=self.subset_size_train,
                seed=self.subset_seed_train,
            )

    def train_dataloader(self) -> DataLoader:
        assert self.train_dataset is not None

        sampler = None
        shuffle = True

        if self.trainer is not None and self.trainer.world_size > 1:
            sampler = DistributedSampler(
                self.train_dataset,
                num_replicas=self.trainer.world_size,
                rank=self.global_rank,
                shuffle=True,
                drop_last=self.sampler_drop_last,
            )
            shuffle = False

        return DataLoader(
            self.train_dataset,
            batch_size=self.per_device_batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
            drop_last=self.loader_drop_last,
        )

    def val_dataloader(self) -> DataLoader:
        dataset = PreTokenizedDataset(self.val_dataset_path)
        subset = maybe_make_fixed_random_subset(
            dataset,
            subset_size=self.subset_size_val,
            seed=self.subset_seed_val,
        ) if self.subset_size_val is not None else dataset

        return DataLoader(
            subset,
            batch_size=self.val_batch_size,
            num_workers=0,
            shuffle=False,
            pin_memory=True,
        )

    def forward(self, **inputs):
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        labels = batch.pop("labels")
        outputs = self(**batch)

        logits = outputs.logits.float()
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))

        self.log("train_loss", loss, prog_bar=True, logger=True)
        self.log("train/phase", 1.0 if self.inherited_global_step == 0 else 2.0,
                 on_step=True,
                 on_epoch=False,
        )
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        accum = self.trainer.accumulate_grad_batches
        is_step_boundary = ((batch_idx + 1) % accum == 0)
        if is_step_boundary:
            self.log(
                "train/effective_global_step",
                float(self.inherited_global_step + self.global_step),
                on_step=True,
                on_epoch=False,
                logger=True,
                sync_dist=True,
            )

    def validation_step(self, batch, batch_idx):
        labels = batch.pop("labels")
        outputs = self(**batch)

        logits = outputs.logits.float()
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.hparams.learning_rate,
            betas=(0.9, 0.95),
        )

        pct_start = min(0.3, self.hparams.warmup_steps / self.total_optimizer_steps)
        scheduler = OneCycleLR(
            optimizer,
            max_lr=self.hparams.learning_rate,
            total_steps=self.total_optimizer_steps,
            pct_start=pct_start,
            div_factor=100.0,
            final_div_factor=0.1,
            anneal_strategy="cos",
            three_phase=False,
            cycle_momentum=False,
        )

        scheduler_config = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
        }

        return [optimizer], [scheduler_config]

    def on_train_start(self) -> None:
        # need to restore optimizer state and model
        if not self.resume_training_state_path or self._loaded_training_state:
            return

        ckpt_path = Path(self.resume_training_state_path) / "trainer.ckpt"
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        # 1) Restore model weights from the Lightning checkpoint payload.
        ckpt_state_dict = ckpt["state_dict"]
        missing, unexpected = self.load_state_dict(ckpt_state_dict, strict=False)

        if missing:
            raise RuntimeError(
                f"Missing keys when loading model state from {ckpt_path}: {missing}"
            )
        if unexpected:
            raise RuntimeError(
                f"Unexpected keys when loading model state from {ckpt_path}: {unexpected}"
            )

        # 2) Restore optimizer state very carefully.
        ckpt_optimizer_states = ckpt.get("optimizer_states", [])
        trainer_optimizers = self.trainer.optimizers

        if len(ckpt_optimizer_states) != len(trainer_optimizers):
            raise RuntimeError(
                "Optimizer count mismatch while restoring training state: "
                f"checkpoint has {len(ckpt_optimizer_states)}, "
                f"current trainer has {len(trainer_optimizers)}"
            )

        for idx, (optimizer, optimizer_state) in enumerate(
            zip(trainer_optimizers, ckpt_optimizer_states)
        ):
            try:
                optimizer.load_state_dict(optimizer_state)
            except Exception as e:
                raise RuntimeError(
                    f"Failed loading optimizer state for optimizer index {idx}"
                ) from e

        # 3) Restore scheduler state carefully.
        ckpt_scheduler_states = ckpt.get("lr_schedulers", [])
        trainer_scheduler_configs = self.trainer.lr_scheduler_configs

        if len(ckpt_scheduler_states) != len(trainer_scheduler_configs):
            raise RuntimeError(
                "Scheduler count mismatch while restoring training state: "
                f"checkpoint has {len(ckpt_scheduler_states)}, "
                f"current trainer has {len(trainer_scheduler_configs)}"
            )

        for idx, (scheduler_state, scheduler_config) in enumerate(
            zip(ckpt_scheduler_states, trainer_scheduler_configs)
        ):
            try:
                scheduler_config.scheduler.load_state_dict(scheduler_state)
            except Exception as e:
                raise RuntimeError(
                    f"Failed loading scheduler state for scheduler index {idx}"
                ) from e
        self._loaded_training_state = True
        rank_zero_info(f"Loaded branch training state from {ckpt_path}")

    def save_model_checkpoint(self, step_dir) -> None:
        assert isinstance(self.model, GPT2LMHeadModelLite)
        self.model.save_pretrained(
            step_dir,
            safe_serialization=True,
            max_shard_size="2GB",
        )

def build_lm_module(
    *,
    model: torch.nn.Module,
    train_dataset_path: Path,
    val_dataset_path: Path,
    per_device_batch_size: int,
    val_batch_size: int,
    num_workers: int = 0,
    learning_rate: float = 3e-4,
    weight_decay: float = 0.1,
    sampler_drop_last: bool = True,
    loader_drop_last: bool = True,
    subset_size_train: int | None = None,
    subset_size_val: int | None = None,
    subset_seed_train: int | None = None,
    subset_seed_val: int | None = None,
    warmup_steps: int = 0,
    total_optimizer_steps: int = 0,
    do_gradient_checkpointing: bool = False,
    resume_training_state_path: str | None = None,
    inherited_global_step: int = 0,
) -> LMModule:
    base = PreTokenizedDataset(train_dataset_path)
    print("Total seq in train dataset:", len(base))

    effective_num_sequences = (
        subset_size_train if subset_size_train is not None else len(base)
    )

    if effective_num_sequences > len(base):
        raise ValueError(
            f"subset_size_train={effective_num_sequences} exceeds dataset size {len(base)}"
        )

    if subset_size_train is not None and subset_seed_train is None:
        raise ValueError(
            "subset_seed_train must be provided when subset_size_train is not None"
        )

    if total_optimizer_steps <= 0:
        raise ValueError(
            f"total_optimizer_steps must be > 0, got {total_optimizer_steps}"
        )

    return LMModule(
        model=model,
        train_dataset_path=train_dataset_path,
        val_dataset_path=val_dataset_path,
        val_batch_size=val_batch_size,
        per_device_batch_size=per_device_batch_size,
        num_workers=num_workers,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        total_optimizer_steps=total_optimizer_steps,
        pin_memory=True,
        sampler_drop_last=sampler_drop_last,
        loader_drop_last=loader_drop_last,
        subset_size_train=subset_size_train,
        subset_size_val=subset_size_val,
        subset_seed_train=subset_seed_train,
        subset_seed_val=subset_seed_val,
        warmup_steps=warmup_steps,
        do_gradient_checkpointing=do_gradient_checkpointing,
        resume_training_state_path=resume_training_state_path,
        inherited_global_step=inherited_global_step,
    )

def get_scaling_dimensions_for_model(depth: int, aspect_ratio: int, head_dim: int):
    """
    From: https://github.com/karpathy/nanochat/blob/c7ba25214276d165eeefca7cb2060587975db189/scripts/base_train.py#L125
    """
    # Model dim is nudged up to nearest multiple of head_dim for clean division
    # (FA3 requires head_dim divisible by 8, and this guarantees head_dim == args.head_dim exactly)
    base_dim = depth * aspect_ratio
    model_dim = ((base_dim + head_dim - 1) // head_dim) * head_dim
    num_heads = model_dim // head_dim
    return {
        "num_heads": num_heads,
        "num_transformer_blocks": depth,
        "n_embed": model_dim,
    }

def do_training(args):
    num_devices = args.gpus_per_node
    batch_size = args.train_batch_size
    assert batch_size % num_devices == 0
    per_device_batch_size = batch_size // num_devices
    world_size = num_devices * args.num_nodes
    grad_accum_steps = args.gradient_accumulation_steps
    num_nodes = args.num_nodes

    L.seed_everything(args.seed)

    ds_1 = PreTokenizedDataset(Path(args.dataset1_path))
    ds_2 = PreTokenizedDataset(Path(args.dataset2_path))
    assert ds_1.seq_len == ds_2.seq_len
    seq_len = ds_1.seq_len
    del ds_1
    del ds_2
    vocab_size = 55028

    if args.start_phase_2_from is None:
        # PHASE 1
        model_dimensions = get_scaling_dimensions_for_model(depth=args.num_layers, aspect_ratio=args.aspect_ratio, head_dim=args.head_dim)
        model_config = GPT2ConfigLite(
            vocab_size=vocab_size,
            n_positions=seq_len,
            n_embd=model_dimensions["n_embed"],
            n_layer=model_dimensions["num_transformer_blocks"],
            n_head=model_dimensions["num_heads"],
            embd_pdrop=args.embed_pdrop,
            resid_pdrop=args.resid_pdrop,
            # as of right now doesn't do anything, we always use gelu
            activation_function="gelu_new",
            layer_norm_epsilon=1e-5,
            pos_emb=args.pos_emb,
            window_pattern="L",
            scale_attn_weights=True,
            scale_attn_by_inverse_layer_idx=True,
            use_cache=False,
            embedding_and_lm_head_weight_tying=True,
            use_value_embeds=False,
            do_torch_compile=args.do_torch_compile,
            mlp_style="GPT2",
        )
        model = GPT2LMHeadModelLite(model_config)
        scaling_params = get_num_scaling_params(model)
        resume_info = {}
        checkpointing_callbacks = [
            Phase1SequenceCheckpoint(
                dirpath=args.output_dir,
                seq_milestones=args.seq_milestones,
                per_device_batch_size=per_device_batch_size,
                grad_accum=args.gradient_accumulation_steps,
            ),
        ]
    else:
        # PHASE 2
        model_config = GPT2ConfigLite.from_json(
            str(Path(args.start_phase_2_from) / "config.json")
        )
        model = GPT2LMHeadModelLite.from_pretrained(
            args.start_phase_2_from, config=model_config
        )
        scaling_params = get_num_scaling_params(model)
        trainer_ckpt_path = Path(args.start_phase_2_from) / "trainer.ckpt"
        resume_info_path = Path(args.start_phase_2_from) / "resume_info.json"
        resume_info = json.loads(resume_info_path.read_text())
        print("loaded a phase 1 checkpoint to continue in phase 2.", trainer_ckpt_path)
        checkpointing_callbacks = [
            Phase2TopKCheckpoint(
                dirpath=args.output_dir,
                filename="phase2-{step}-{val_loss:.4f}",
                monitor="val_loss",
                mode="min",
                save_top_k=args.save_top_k_checkpoints_phase_2,
                save_last=False,
            ),
            OverfitStopper(
                monitor="val_loss",
                warmup_checks=args.overfit_warmup_checks,
                window_size=args.overfit_window_size,
                overfit_margin=args.overfit_margin,
                patience_checks=args.overfit_patience_checks,
            ),
        ]

    logger_dict = {}
    if args.use_wandb:
        num_layers = args.num_layers
        n = args.n_ds1
        k = args.k_ds2
        run_name = f"{num_layers}_ds1-{n}_ds2-{k}"
        if resume_info.get("wandb_run_id", None):
            run_name += " (phase 2)"

        wandb_logger = WandbLogger(
            project=args.wandb_project,
            name=run_name,
            save_dir=args.output_dir,
            config={
                **vars(args),
                **{f"scaling_param__{x}": y for x, y in scaling_params.items()},
            },
        )
        logger_dict["logger"] = wandb_logger

    log_every_n_steps = 10
    subset_size_2 = args.k_ds2
    subset_seed_2 = args.dataset2_subset_seed
    if args.start_phase_2_from is None:
        # --- PHASE 1 ---
        total_schedule_steps = optimizer_steps_per_epoch(
            num_sequences=args.n_ds1,
            world_size=world_size,
            per_device_batch_size=per_device_batch_size,
            grad_accum_steps=grad_accum_steps,
            sampler_drop_last=True,
            loader_drop_last=True,
        )
        lit_model = build_lm_module(
            model=model,
            train_dataset_path=Path(args.dataset1_path),
            val_dataset_path=Path(args.val_dataset_path),
            per_device_batch_size=per_device_batch_size,
            val_batch_size=args.val_batch_size,
            num_workers=0,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            sampler_drop_last=True,
            loader_drop_last=True,
            subset_size_train=args.n_ds1,
            subset_size_val=max(subset_size_2 // 10, 100) if subset_size_2 is not None else None,
            subset_seed_train=args.dataset1_subset_seed,
            subset_seed_val=subset_seed_2,
            warmup_steps=args.warmup_steps,
            total_optimizer_steps=total_schedule_steps,
            do_gradient_checkpointing=args.do_gradient_checkpointing,
            resume_training_state_path=None,
            inherited_global_step=0,
        )
        trainer = L.Trainer(
            accelerator="gpu",
            devices=num_devices,
            num_nodes=num_nodes,
            strategy=DDPStrategy(find_unused_parameters=False, static_graph=False),
            max_steps=total_schedule_steps,
            reload_dataloaders_every_n_epochs=1,
            callbacks=[
                *checkpointing_callbacks,
                LearningRateMonitor(logging_interval="step"),
                MaxStepProgressBar(),
            ],
            enable_progress_bar=True,
            precision="bf16-mixed" if args.bf16 else 32,
            gradient_clip_val=args.max_grad_norm,
            accumulate_grad_batches=args.gradient_accumulation_steps,
            log_every_n_steps=log_every_n_steps,
            val_check_interval=args.steps_per_eval,
            check_val_every_n_epoch=None,
            **logger_dict,
        )
        trainer.fit(lit_model)
        trainer.validate(lit_model)
    else:
        # --- PHASE 2 ---
        resume_training_state_path = args.start_phase_2_from
        inherited_global_step = resume_info["global_step"]
        total_steps_budget = resume_info["total_optimizer_steps"]
        steps_remaining = total_steps_budget - inherited_global_step
        lit_model = build_lm_module(
            model=model,
            train_dataset_path=Path(args.dataset2_path),
            val_dataset_path=Path(args.val_dataset_path),
            per_device_batch_size=per_device_batch_size,
            val_batch_size=args.val_batch_size,
            num_workers=0,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            sampler_drop_last=True,
            loader_drop_last=True,
            subset_size_train=args.k_ds2,
            subset_seed_train=args.dataset2_subset_seed,
            subset_size_val=max(subset_size_2 // 10, 100) if subset_size_2 is not None else None,
            subset_seed_val=subset_seed_2,
            warmup_steps=args.warmup_steps,
            total_optimizer_steps=total_steps_budget,
            do_gradient_checkpointing=args.do_gradient_checkpointing,
            resume_training_state_path=resume_training_state_path,
            inherited_global_step=inherited_global_step,
        )
        trainer = L.Trainer(
            accelerator="gpu",
            devices=num_devices,
            num_nodes=num_nodes,
            strategy=DDPStrategy(find_unused_parameters=False, static_graph=False),
            max_steps=steps_remaining,
            reload_dataloaders_every_n_epochs=1,
            callbacks=[
                *checkpointing_callbacks,
                LearningRateMonitor(logging_interval="step"),
                MaxStepProgressBar(),
            ],
            enable_progress_bar=True,
            precision="bf16-mixed" if args.bf16 else 32,
            gradient_clip_val=args.max_grad_norm,
            accumulate_grad_batches=args.gradient_accumulation_steps,
            log_every_n_steps=log_every_n_steps,
            val_check_interval=args.steps_per_eval,
            check_val_every_n_epoch=None,
            # skip it
            num_sanity_val_steps=0,
            **logger_dict,
        )
        trainer.fit(lit_model)
        trainer.validate(lit_model)

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPT-2 training script using PyTorch Lightning")

    # Midtraining and Dataset
    parser.add_argument("--dataset1_path", type=str, help="Dataset 1 (n, transcripts) numpy file")
    parser.add_argument("--dataset2_path", type=str, help="Dataset 1 (k, lakh) numpy file")
    parser.add_argument("--val_dataset_path", type=str, help="Lakh validation numpy file")
    parser.add_argument("--n_ds1", type=int, default=10, help="Number of sequences from transcripts")
    parser.add_argument("--k_ds2", type=int, default=10, help="Number of sequences from Lakh train")
    parser.add_argument("--dataset1_subset_seed", type=int, default=1234, help="Dataset 1 subset seed")
    parser.add_argument("--dataset2_subset_seed", type=int, default=5678, help="Dataset 1 subset seed")
    parser.add_argument(
        "--seq-milestones",
        type=int,
        nargs="+",
        default=[],
        help="Sequence-count milestones at which to save phase-1 branch checkpoints.",
    )

    # Model
    parser.add_argument("--head_dim", type=int, default=64, help="The number of dimensions per attention head")
    parser.add_argument("--num_layers", type=int, default=12, help="Model number of layers")
    parser.add_argument(
        "--aspect_ratio",
        type=int,
        default=64,
        help="model_dim = depth * aspect_ratio. Default is 64, same as nanochat",
    )
    parser.add_argument("--embed_pdrop", type=float, default=0.1, help="Apply embedding dropout")
    parser.add_argument("--resid_pdrop", type=float, default=0.1, help="Apply residual dropout")
    parser.add_argument(
        "--pos_emb",
        type=str,
        default="absolute",
        choices=["absolute", "rope"],
        help=(
            "The positional embedding choice: either `absolute` (vanilla) or RoPE. (rope)"
        ),
    )

    # Optimization
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--train_batch_size", type=int, default=512,
                        help="Batch size for training (total, not per device)")
    parser.add_argument("--val_batch_size", type=int, default=512,
                        help="Batch size for validation")
    parser.add_argument("--warmup_steps", type=int, default=200, help="Number of warmup steps")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")

    parser.add_argument("--overfit_window_size", type=int, default=1, help="Number of validation samples to use in smoothed best val loss.")
    parser.add_argument("--overfit_warmup_checks", type=int, default=10, help="Do not start counting bad val loss until this many checks after phase 2 starts")
    parser.add_argument("--overfit_patience_checks", type=int, default=5, help="Number of consecutive bad val loss samples we can withstand before early stopping.")
    parser.add_argument("--overfit_margin", type=float, default=0.05, help="Delta over best val after which considered a bad val loss.")
    parser.add_argument(
        "--do_gradient_checkpointing",
        action="store_true",
        help="Enable gradient checkpointing",
    )

    # System
    parser.add_argument("--num_nodes", type=int, default=1, help="Number of nodes")
    parser.add_argument("--gpus_per_node", type=int, default=1, help="Number of GPUs per node")
    parser.add_argument(
        "--do_torch_compile",
        action="store_true",
        help="Enable calling torch.compile on model instance",
    )
    parser.add_argument("--output_dir", type=str, help="Output directory for checkpoints")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                        help="Initialize model weights from this checkpoint (not supported)")
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16 mixed precision training")
    parser.add_argument("--steps_per_eval", type=int, default=1000, help="Number of steps between validation evals")
    parser.add_argument("--steps_per_checkpoint", type=int, default=1000,
                        help="Number of steps between checkpoints")
    parser.add_argument("--start_phase_2_from", type=str, default=None,
                        help="Start training on dataset 2 from this checkpoint dir")

    # Logging
    parser.add_argument(
        "--use_wandb", action="store_true", help="whether to use wandb logging"
    )
    parser.add_argument(
        "--wandb_tag", type=str, help="A tag to associate with this run"
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        help="The project to save this run ",
        default="gpt-anticipation-2.0",
    )
    parser.add_argument(
        "--wandb_resume_from_run_id",
        type=str,
        help="The run ID to resume. It will be in the URL of the wandb website.",
        default="",
    )
    parser.add_argument("--save_top_k_checkpoints_phase_2", type=int, default=3, help="Once phase 2 is reached preserve k checkpoints.")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    do_training(args)
