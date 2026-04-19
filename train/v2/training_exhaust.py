import os
import time
import argparse
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

from transformers import PretrainedConfig, GPT2LMHeadModel, GPT2Config

from train.v2.dataset_utils import PreTokenizedDataset




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

        self.num_checks += 1
        self.recent_vals.append(val)

        if val < self.best_val:
            self.best_val = val

        smoothed = sum(self.recent_vals) / len(self.recent_vals)

        # These are allowed here.
        pl_module.log("overfit/best_val", self.best_val, on_step=False, on_epoch=True, sync_dist=True)
        pl_module.log("overfit/smoothed_val", smoothed, on_step=False, on_epoch=True, sync_dist=True)
        pl_module.log(
            "overfit/excess_over_best",
            smoothed - self.best_val,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        if self.num_checks <= self.warmup_checks:
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


class HuggingFaceCheckpoint(ModelCheckpoint):
    def __init__(self, config, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.config = config

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)

        if (trainer.global_step % self._every_n_train_steps) != 0:
            return

        if trainer.global_step == 0:
            # don't save if we just started
            return

        step_dir = os.path.join(str(self.dirpath), f"step-{trainer.global_step}")
        raw_model = pl_module.model if hasattr(pl_module, "model") else pl_module
        raw_model.save_pretrained(
            step_dir, safe_serialization=True, max_shard_size="2GB"
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


class TwoPhaseLMModule(L.LightningModule):
    """
    Epoch 0:    train on dataset 1
    Epochs 1-4: train on dataset 2

    Optimizer and scheduler state remain continuous because everything happens
    inside one fit() call.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        dataset1_path: Path,
        dataset2_path: Path,
        val_dataset_path: Path,
        per_device_batch_size: int,
        val_batch_size: int,
        num_workers: int,
        learning_rate: float,
        weight_decay: float,
        total_optimizer_steps: int,
        steps_ds1: int,
        pin_memory: bool = True,
        sampler_drop_last: bool = True,
        loader_drop_last: bool = True,
        subset_size_1: int | None = None,
        subset_size_2: int | None = None,
        subset_seed_1: int | None = None,
        subset_seed_2: int | None = None,
        warmup_steps: int = 0,
    ) -> None:
        super().__init__()
        self.model = model
        self.model.config.bos_token_id = self.model.config.eos_token_id = 0

        # self.model = torch.compile(
        #     self.model, dynamic=False, fullgraph=True
        # )
        self.model.gradient_checkpointing_enable()

        self.dataset1_path = Path(dataset1_path)
        self.dataset2_path = Path(dataset2_path)
        self.val_dataset_path = Path(val_dataset_path)
        self.val_batch_size = val_batch_size

        self.per_device_batch_size = per_device_batch_size
        self.num_workers = num_workers
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay
        self.total_optimizer_steps = total_optimizer_steps
        self.pin_memory = pin_memory
        self.sampler_drop_last = sampler_drop_last
        self.loader_drop_last = loader_drop_last

        # use this to determined when the cut-off point is
        self.steps_ds1 = steps_ds1
        self._switch_logged = False

        self.subset_size_1 = subset_size_1
        self.subset_size_2 = subset_size_2
        self.subset_seed_1 = subset_seed_1
        self.subset_seed_2 = subset_seed_2

        self.dataset1: Dataset | None = None
        self.dataset2: Dataset | None = None

        self.save_hyperparameters(ignore=["model"])

    def setup(self, stage: str | None = None) -> None:
        if stage not in (None, "fit"):
            return

        if self.dataset1 is None:
            base1 = PreTokenizedDataset(self.dataset1_path)
            self.dataset1 = maybe_make_fixed_random_subset(
                base1,
                subset_size=self.subset_size_1,
                seed=self.subset_seed_1,
            )

        if self.dataset2 is None:
            base2 = PreTokenizedDataset(self.dataset2_path)
            self.dataset2 = maybe_make_fixed_random_subset(
                base2,
                subset_size=self.subset_size_2,
                seed=self.subset_seed_2,
            )

    def _current_train_dataset(self) -> Dataset:
        assert self.dataset1 is not None
        assert self.dataset2 is not None
        if self.steps_ds1 == 0:
            # can also say not to use dataset 1 at all
            return self.dataset2
        else:
            return self.dataset1 if self.current_epoch == 0 else self.dataset2

    def val_dataloader(self):
        dataset = PreTokenizedDataset(self.val_dataset_path)
        if self.subset_size_2 is not None:
            subset = maybe_make_fixed_random_subset(
                dataset,
                # proportional to k
                subset_size=max(self.subset_size_2 // 10, 100),
                seed=self.subset_seed_2,
            )
        else:
            subset = dataset

        return DataLoader(
            subset,
            batch_size=self.val_batch_size,
            num_workers=0,
            shuffle=False,
            pin_memory=True
        )

    def train_dataloader(self) -> DataLoader:
        dataset = self._current_train_dataset()

        sampler = None
        shuffle = True

        if self.trainer is not None and self.trainer.world_size > 1:
            sampler = DistributedSampler(
                dataset,
                num_replicas=self.trainer.world_size,
                rank=self.global_rank,
                shuffle=True,
                drop_last=self.sampler_drop_last,
            )
            shuffle = False

        return DataLoader(
            dataset,
            batch_size=self.per_device_batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
            drop_last=self.loader_drop_last,
        )

    def forward(self, **inputs):
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        labels = batch.pop("labels")
        outputs = self(**batch)

        logits = outputs.logits.float()  # upcast logits and compute loss in fp32
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
        self.log("train_loss", loss, prog_bar=True, logger=True)

        # phase is 0 if pretraining on n, 1 if midtraining on k
        if self.steps_ds1 == 0:
            # no steps in the first dataset, we start on phase 2
            self.log("train/phase", 2.0, on_step=True, on_epoch=False, sync_dist=True)
        else:
            # phase 1 = noisy pretrain
            # phase 2 = target midtrain
            self.log("train/phase", 1.0 if self.current_epoch == 0 else 2.0, on_step=True, on_epoch=False, sync_dist=True)

        if not self._switch_logged and self.global_step >= self.steps_ds1:
            self._switch_logged = True

        return loss

    def validation_step(self, batch, batch_idx):
        labels = batch.pop("labels")
        outputs = self(**batch)

        logits = outputs.logits.float()  # upcast logits and compute loss in fp32
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters()
                            if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters()
                            if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.hparams.learning_rate,
            betas=(0.9, 0.95)
        )

        train_dataloader = self.train_dataloader()
        if hasattr(train_dataloader, "dataset"):
            dataset_size = len(train_dataloader.dataset)
            pct_start = min(0.3, self.hparams.warmup_steps / self.total_optimizer_steps)
            scheduler = OneCycleLR(
                optimizer,
                max_lr=self.hparams.learning_rate,
                total_steps=self.total_optimizer_steps,
                pct_start=pct_start,
                div_factor=100.0,
                final_div_factor=0.1,
                anneal_strategy='cos',
                three_phase=False,
                cycle_momentum=False
            )

            scheduler_config = {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1
            }

            return [optimizer], [scheduler_config]

        return optimizer

    def save_model_checkpoint(
        self, step_dir
    ) -> None:
        assert isinstance(self.model, GPT2LMHeadModel)
        self.model.save_pretrained(
            step_dir, safe_serialization=True, max_shard_size="2GB",
        )


def build_two_phase_module(
    *,
    model: torch.nn.Module,
    dataset1_path: Path,
    dataset2_path: Path,
    val_dataset_path: Path,
    per_device_batch_size: int,
    val_batch_size: int,
    grad_accum_steps: int,
    world_size: int,
    num_workers: int = 0,
    learning_rate: float = 3e-4,
    weight_decay: float = 0.1,
    sampler_drop_last: bool = True,
    loader_drop_last: bool = True,
    subset_size_1: int | None = None,
    subset_size_2: int | None = None,
    subset_seed_1: int | None = None,
    subset_seed_2: int | None = None,
    warmup_steps: int = 0,
    num_epochs_dataset_1: int = 1,
    num_epochs_dataset_2: int = 4,
) -> TwoPhaseLMModule:
    base1 = PreTokenizedDataset(dataset1_path)
    base2 = PreTokenizedDataset(dataset2_path)

    effective_num_sequences_1 = subset_size_1 if subset_size_1 is not None else len(base1)
    effective_num_sequences_2 = subset_size_2 if subset_size_2 is not None else len(base2)

    if effective_num_sequences_1 > len(base1):
        raise ValueError(
            f"subset_size_1={effective_num_sequences_1} exceeds dataset1 size {len(base1)}"
        )
    if effective_num_sequences_2 > len(base2):
        raise ValueError(
            f"subset_size_2={effective_num_sequences_2} exceeds dataset2 size {len(base2)}"
        )

    if subset_size_1 is not None and subset_seed_1 is None:
        raise ValueError("subset_seed_1 must be provided when subset_size_1 is not None")
    if subset_size_2 is not None and subset_seed_2 is None:
        raise ValueError("subset_seed_2 must be provided when subset_size_2 is not None")

    steps_ds1 = num_epochs_dataset_1 * optimizer_steps_per_epoch(
        num_sequences=effective_num_sequences_1,
        world_size=world_size,
        per_device_batch_size=per_device_batch_size,
        grad_accum_steps=grad_accum_steps,
        sampler_drop_last=sampler_drop_last,
        loader_drop_last=loader_drop_last,
    )

    steps_ds2 = num_epochs_dataset_2 * optimizer_steps_per_epoch(
        num_sequences=effective_num_sequences_2,
        world_size=world_size,
        per_device_batch_size=per_device_batch_size,
        grad_accum_steps=grad_accum_steps,
        sampler_drop_last=sampler_drop_last,
        loader_drop_last=loader_drop_last,
    )

    total_optimizer_steps = steps_ds1 + steps_ds2

    if total_optimizer_steps <= 0:
        raise ValueError(
            "Computed total_optimizer_steps <= 0. "
            "Check subset sizes, world size, batch size, grad accumulation, and drop_last."
        )

    return TwoPhaseLMModule(
        model=model,
        dataset1_path=dataset1_path,
        dataset2_path=dataset2_path,
        val_dataset_path=val_dataset_path,
        val_batch_size=val_batch_size,
        per_device_batch_size=per_device_batch_size,
        num_workers=num_workers,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        total_optimizer_steps=total_optimizer_steps,
        steps_ds1=steps_ds1,
        pin_memory=True,
        sampler_drop_last=sampler_drop_last,
        loader_drop_last=loader_drop_last,
        subset_size_1=subset_size_1,
        subset_size_2=subset_size_2,
        subset_seed_1=subset_seed_1,
        subset_seed_2=subset_seed_2,
        warmup_steps=warmup_steps
    )

def do_training(args):
    num_devices = args.gpus_per_node
    batch_size = args.train_batch_size
    assert batch_size % num_devices == 0
    per_device_bach_size = batch_size // num_devices
    world_size = num_devices * args.num_nodes
    grad_accum_steps = args.gradient_accumulation_steps
    num_nodes = args.num_nodes

    L.seed_everything(args.seed)

    model_config = GPT2Config(
        vocab_size=55028,
        n_positions=args.seq_len,
        n_embd=args.hidden_dim,
        n_layer=args.num_layers,
        n_head=args.num_heads,
        embd_pdrop=args.embed_pdrop,
        resid_pdrop=args.resid_pdrop,
        activation_function="gelu_new",
        layer_norm_epsilon=1e-5,
        scale_attn_weights=True,
        scale_attn_by_inverse_layer_idx=True,
        use_cache=False,
    )
    model = GPT2LMHeadModel(model_config)
    checkpoint_callback = HuggingFaceCheckpoint(
        config=model.config,
        dirpath=args.output_dir,
        filename="{step}",
        save_top_k=0,
        monitor=None,
        save_last=False,
        every_n_train_steps=args.steps_per_checkpoint
    )
    checkpoint_callback.CHECKPOINT_EQUALS_CHAR = "-"

    max_epochs = args.epochs_ds1 + args.epochs_ds2
    lit_model = build_two_phase_module(
        model=model,
        dataset1_path=Path(args.dataset1_path),
        dataset2_path=Path(args.dataset2_path),
        val_dataset_path=Path(args.val_dataset_path),
        val_batch_size=args.val_batch_size,
        per_device_batch_size=per_device_bach_size,
        grad_accum_steps=grad_accum_steps,
        world_size=world_size,
        num_workers=0,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        sampler_drop_last=True,
        loader_drop_last=True,
        # number of subsets from each dataset, sequence wise
        subset_size_1=args.n_ds1,
        subset_size_2=args.k_ds2,
        subset_seed_1=args.dataset1_subset_seed,
        subset_seed_2=args.dataset2_subset_seed,
        warmup_steps=args.warmup_steps,
        num_epochs_dataset_1=args.epochs_ds1,
        num_epochs_dataset_2=args.epochs_ds2,
    )

    logger_dict = {}
    if args.use_wandb:
        if args.wandb_tag:
            tags = [str(args.wandb_tag).strip()]
        else:
            tags = []

        if tags == ["scaling"]:
            run_name = f"scaling-run-{int(time.time())}"
        else:
            num_layers = args.num_layers
            k = args.k_ds2
            n = args.n_ds1
            #time_int = int(time.time())
            run_name = f"run-{num_layers}-{k}-{n}"

        if args.wandb_resume_from_run_id:
            resume_from_run_id: str = args.wandb_resume_from_run_id
            resume_from_step: int = args.wandb_resume_from_step
            assert resume_from_step > 0, "Resume from step is unfilled or 0 - start a new run or provide a resume step."
            resume_from = f"{resume_from_run_id}?_step={resume_from_step}"
            import wandb
            _run = wandb.init(
                project=args.wandb_project,
                name=run_name,
                # UPDATE: both fork_from and resume_from are in private preview
                # email: support@wandb.com
                # to enable them on your account
                fork_from=resume_from,
                # this is in private preview, forking is the closest alternative for now
                # it starts a new run from the step you specify
                #resume_from=resume_from,
            )
            wandb_logger = WandbLogger(
                experiment=_run,
                save_dir=args.output_dir,
            )
        else:
            wandb_logger = WandbLogger(
                project=args.wandb_project,
                name=run_name,
                save_dir=args.output_dir,
                config={
                    **vars(args),
                },
                tags=tags,
            )

        logger_dict["logger"] = wandb_logger

    trainer = L.Trainer(
        accelerator="gpu",
        devices=num_devices,
        num_nodes=num_nodes,
        strategy=DDPStrategy(find_unused_parameters=False, static_graph=False),
        max_epochs=max_epochs,
        reload_dataloaders_every_n_epochs=1,
        callbacks=[
            checkpoint_callback,
            LearningRateMonitor(logging_interval="step"),
            MaxStepProgressBar(),
            OverfitStopper(
                monitor="val_loss",
                warmup_checks=20,
                window_size=1,
                # this is in terms of nats/tok
                overfit_margin=0.05,
                patience_checks=5,
            ),
        ],
        enable_progress_bar=True,
        precision="bf16-mixed" if args.bf16 else 32,
        gradient_clip_val=args.max_grad_norm,
        accumulate_grad_batches=args.gradient_accumulation_steps,
        log_every_n_steps=10,
        val_check_interval=args.steps_per_eval,
        check_val_every_n_epoch=None,
        max_steps=lit_model.total_optimizer_steps,
        **logger_dict
    )
    trainer.fit(lit_model)
    trainer.validate(lit_model)

    # save the model only, not trainer state
    #lit_model: TwoPhaseLMModule
    #lit_model.save_model_checkpoint(Path(args.output_dir))
    #print(f"saved checkpoint to: {Path(args.output_dir)}")

    # saves trainer state and model weights
    #trainer.save_checkpoint((Path(args.output_dir) / "final.ckpt"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPT-2 training script using PyTorch Lightning")

    # new for midtraining
    parser.add_argument("--dataset1_path", type=str, help="Dataset 1 (n, transcripts) numpy file")
    parser.add_argument("--dataset2_path", type=str, help="Dataset 1 (k, lakh) numpy file")
    parser.add_argument("--val_dataset_path", type=str, help="Lakh validation numpy file")

    parser.add_argument("--n_ds1", type=int, default=10, help="Number of sequences from transcripts")
    parser.add_argument("--k_ds2", type=int, default=10, help="Number of sequences from Lakh train")

    parser.add_argument("--epochs_ds1", type=int, default=1, help="Number of epochs for dataset 1")
    parser.add_argument("--epochs_ds2", type=int, default=4, help="Number of epochs for dataset 2")

    parser.add_argument("--dataset1_subset_seed", type=int, default=1234, help="Dataset 1 subset seed")
    parser.add_argument("--dataset2_subset_seed", type=int, default=5678, help="Dataset 1 subset seed")

    # --- existed before ---
    parser.add_argument("--output_dir", type=str, help="Output directory for checkpoints")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                        help="Initialize model weights from this checkpoint (not supported)")

    # Dataset parameters
    parser.add_argument("--seq_len", type=int, default=1024, help="Dataset sequence length")

    # Model parameters
    parser.add_argument("--hidden_dim", type=int, default=768, help="Model hidden dimensions")  # 768
    parser.add_argument("--num_heads", type=int, default=12, help="Model number of attention heads")  # 12
    parser.add_argument("--num_layers", type=int, default=12, help="Model number of layers")  # 12
    parser.add_argument("--embed_pdrop", type=float, default=0.1, help="Apply embedding dropout")
    parser.add_argument("--resid_pdrop", type=float, default=0.1, help="Apply residual dropout")

    # Optimization parameters
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
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16 mixed precision training")
    parser.add_argument("--steps_per_eval", type=int, default=1000, help="Number of steps between validation evals")
    parser.add_argument("--steps_per_checkpoint", type=int, default=1000,
                        help="Number of steps between checkpoints")  # set back to 1000

    # System parameters
    parser.add_argument("--num_nodes", type=int, default=1, help="Number of nodes")
    parser.add_argument("--gpus_per_node", type=int, default=1, help="Number of GPUs per node")  # 4 gpus

    # wandb stuff
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

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    do_training(args)
