#!/usr/bin/env python3
"""
LoRA Fine-Tuning Script for Anticipatory Music Transformer

This script fine-tunes the AMT model using LoRA (Low-Rank Adaptation) via PEFT.
Based on paper hyperparameters (Table 6, Thickstun et al. 2024), adapted for local GPU training.

Supports experiment matrix from docs/amt-fine-tuning.md Section 13.2 and 13.3.

Usage:
    python amt_lora_train.py --config L2 --train_data data/lmd_10pct/train.txt --valid_data data/lmd_10pct/valid.txt --output_dir outputs/exp-e2

    # Or manual override:
    python amt_lora_train.py --lora_rank 16 --target_modules qkv --lora_dropout 0.0 ...
"""

import os
import sys
import json
import math
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


LORA_CONFIGS = {
    "L1": {
        "rank": 8,
        "alpha": 16,
        "dropout": 0.0,
        "target_modules": ["q_proj", "v_proj", "k_proj"],
    },
    "L2": {
        "rank": 16,
        "alpha": 32,
        "dropout": 0.0,
        "target_modules": ["q_proj", "v_proj", "k_proj"],
    },
    "L3": {
        "rank": 32,
        "alpha": 64,
        "dropout": 0.0,
        "target_modules": ["q_proj", "v_proj", "k_proj"],
    },
    "L4": {
        "rank": 64,
        "alpha": 128,
        "dropout": 0.0,
        "target_modules": ["q_proj", "v_proj", "k_proj"],
    },
    "L5": {
        "rank": 16,
        "alpha": 32,
        "dropout": 0.0,
        "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
    },
    "L6": {
        "rank": 32,
        "alpha": 64,
        "dropout": 0.1,
        "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
    },
    "L7": {"rank": 16, "alpha": 32, "dropout": 0.0, "target_modules": "all"},
}

TARGET_MODULE_PRESETS = {
    "qkv": ["q_proj", "v_proj", "k_proj"],
    "qkvo": ["q_proj", "v_proj", "k_proj", "o_proj"],
    "all": [
        "q_proj",
        "v_proj",
        "k_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
}


class TokenDataset(Dataset):
    """Dataset that loads token sequences from text file."""

    def __init__(self, filepath: str, max_length: int = 1024):
        self.filepath = filepath
        self.max_length = max_length
        self.sequences = []
        self._load_sequences()

    def _load_sequences(self):
        logger.info(f"Loading sequences from {self.filepath}")
        file_size_mb = os.path.getsize(self.filepath) / (1024 * 1024)
        logger.info(f"File size: {file_size_mb:.1f} MB")

        with open(self.filepath, "r", encoding="utf-8") as f:
            for i, line in enumerate(tqdm(f, desc="Loading", unit=" lines")):
                tokens = [int(t) for t in line.split()]
                if len(tokens) >= 2:
                    tokens = tokens[: self.max_length]
                    self.sequences.append(tokens)
        logger.info(f"Loaded {len(self.sequences)} sequences")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        tokens = self.sequences[idx]
        return {
            "input_ids": torch.tensor(tokens, dtype=torch.long),
            "labels": torch.tensor(tokens, dtype=torch.long),
        }


class CrossEntropyLossCallback(TrainerCallback):
    """Callback to log training loss at each step."""

    def __init__(self):
        self.train_losses = []
        self.eval_losses = []
        self.train_steps = []
        self.eval_steps = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            if "loss" in logs:
                self.train_losses.append(logs["loss"])
                self.train_steps.append(state.global_step)
            elif "eval_loss" in logs:
                self.eval_losses.append(logs["eval_loss"])
                self.eval_steps.append(state.global_step)


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="Fine-tune AMT with LoRA using PEFT",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Config preset
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="LoRA config preset (L1-L7 from experiment matrix). Overrides manual settings.",
    )

    # Data arguments
    parser.add_argument(
        "--train_data",
        type=str,
        default="data/lmd_10pct/train.txt",
        help="Training data file",
    )
    parser.add_argument(
        "--valid_data",
        type=str,
        default="data/lmd_10pct/valid.txt",
        help="Validation data file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/amt-lora-default",
        help="Output directory",
    )

    # Model arguments
    parser.add_argument(
        "--model_path",
        type=str,
        default="stanford-crfm/music-large-800k",
        help="Model path or HF model ID",
    )
    parser.add_argument(
        "--max_seq_length", type=int, default=1024, help="Maximum sequence length"
    )

    # LoRA arguments (used if --config not specified)
    parser.add_argument("--lora_rank", type=int, default=16, help="LoRA rank (r)")
    parser.add_argument(
        "--lora_alpha", type=int, default=32, help="LoRA alpha (scaling)"
    )
    parser.add_argument("--lora_dropout", type=float, default=0.0, help="LoRA dropout")
    parser.add_argument(
        "--target_modules",
        type=str,
        default="qkv",
        help="Target modules: qkv, qkvo, or all (comma-separated)",
    )

    # Training arguments
    parser.add_argument(
        "--num_epochs", type=int, default=1, help="Number of training epochs"
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=4,
        help="Training batch size per device",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Eval batch size per device",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=3e-4,
        help="Learning rate (paper Medium: 3e-4)",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=100,
        help="Warmup steps (paper: 1000 for 100k steps)",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--max_steps", type=int, default=-1, help="Max training steps (-1 = full epoch)"
    )
    parser.add_argument(
        "--logging_steps", type=int, default=50, help="Log every N steps"
    )
    parser.add_argument(
        "--save_steps", type=int, default=500, help="Save checkpoint every N steps"
    )
    parser.add_argument(
        "--eval_steps", type=int, default=500, help="Evaluate every N steps"
    )

    # Other arguments
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--dataloader_num_workers", type=int, default=4, help="DataLoader workers"
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Resume from checkpoint",
    )
    parser.add_argument(
        "--experiment_id",
        type=str,
        default=None,
        help="Experiment ID (e.g., E1, E2) for tracking",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    set_seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    script_dir = Path(__file__).parent.resolve()
    data_dir = script_dir / ".." / "data"
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = script_dir / output_dir

    train_data_path = Path(args.train_data)
    if not train_data_path.is_absolute():
        train_data_path = data_dir / train_data_path

    valid_data_path = Path(args.valid_data)
    if not valid_data_path.is_absolute():
        valid_data_path = data_dir / valid_data_path

    # Apply config preset if specified
    if args.config and args.config in LORA_CONFIGS:
        cfg = LORA_CONFIGS[args.config]
        args.lora_rank = cfg["rank"]
        args.lora_alpha = cfg["alpha"]
        args.lora_dropout = cfg["dropout"]
        if isinstance(cfg["target_modules"], str) and cfg["target_modules"] == "all":
            args.target_modules = "all"
        else:
            args.target_modules = ",".join(cfg["target_modules"])
        logger.info(f"Applied config preset: {args.config}")

    # Resolve target modules
    if args.target_modules in TARGET_MODULE_PRESETS:
        target_modules = TARGET_MODULE_PRESETS[args.target_modules]
    else:
        target_modules = args.target_modules.split(",")

    # Calculate trainable parameters estimate
    # Approximate: 2 * rank * hidden_dim * num_layers * len(target_modules)
    # For 780M model: ~1024 hidden, 36 layers
    est_params = 2 * args.lora_rank * 1024 * 36 * len(target_modules)
    est_params_str = f"~{est_params / 1e6:.1f}M"

    logger.info("=" * 60)
    logger.info("AMT LoRA Fine-Tuning Configuration")
    logger.info("=" * 60)
    logger.info(f"Model: {args.model_path}")
    logger.info(f"Train data: {train_data_path}")
    logger.info(f"Valid data: {valid_data_path}")
    logger.info(f"Output dir: {output_dir}")
    logger.info(f"Max seq length: {args.max_seq_length}")
    logger.info(f"LoRA rank: {args.lora_rank}")
    logger.info(f"LoRA alpha: {args.lora_alpha}")
    logger.info(f"LoRA dropout: {args.lora_dropout}")
    logger.info(f"Target modules: {target_modules}")
    logger.info(f"Trainable params: {est_params_str}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info(f"Batch size: {args.per_device_train_batch_size}")
    logger.info(f"Gradient accumulation: {args.gradient_accumulation_steps}")
    logger.info("=" * 60)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save full config
    config = {
        "experiment_id": args.experiment_id,
        "config_preset": args.config,
        "model_path": args.model_path,
        "train_data": str(train_data_path),
        "valid_data": str(valid_data_path),
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "target_modules": target_modules,
        "learning_rate": args.learning_rate,
        "num_epochs": args.num_epochs,
        "max_seq_length": args.max_seq_length,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "seed": args.seed,
        "timestamp": datetime.now().isoformat(),
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        padding_side="left",
    )
    tokenizer.pad_token = tokenizer.eos_token

    # Load model
    logger.info("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    model.config.pad_token_id = tokenizer.eos_token_id

    # Configure LoRA
    logger.info("Configuring LoRA...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
        inference_mode=False,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load datasets
    logger.info("Loading datasets...")
    train_dataset = TokenDataset(str(train_data_path), max_length=args.max_seq_length)
    valid_dataset = TokenDataset(str(valid_data_path), max_length=args.max_seq_length)

    # Configure training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=False,
        logging_dir=str(output_dir / "logs"),
        report_to="none",
        seed=args.seed,
        dataloader_num_workers=args.dataloader_num_workers,
        remove_unused_columns=False,
        label_names=["labels"],
        optim="adamw_torch",
        max_grad_norm=1.0,
        bf16=torch.cuda.is_available(),
        fp16=False,
        logging_first_step=True,
        save_total_limit=3,
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
    )

    # Configure callbacks
    callback = CrossEntropyLossCallback()
    trainer.add_callback(callback)

    # Train
    logger.info("Starting training...")
    checkpoint = args.resume_from_checkpoint
    trainer.train(resume_from_checkpoint=checkpoint)

    # Save final model
    logger.info("Saving final model...")
    trainer.save_model(str(output_dir / "final"))
    trainer.save_state()

    # Save training metrics
    metrics = {
        "experiment_id": args.experiment_id,
        "config_preset": args.config,
        "train_steps": len(callback.train_losses),
        "train_step_values": callback.train_steps,
        "train_losses": callback.train_losses,
        "eval_steps": len(callback.eval_losses),
        "eval_step_values": callback.eval_steps,
        "eval_losses": callback.eval_losses,
    }

    if callback.train_losses:
        metrics["final_train_loss"] = callback.train_losses[-1]
    if callback.eval_losses:
        metrics["final_eval_loss"] = callback.eval_losses[-1]

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Generate experiment tracking entry
    exp_entry = generate_experiment_entry(args, config, metrics)
    with open(output_dir / "experiment_tracking.md", "w") as f:
        f.write(exp_entry)

    logger.info(f"Training complete! Output: {output_dir}")
    logger.info(f"Trainable parameters: {model.print_trainable_parameters()}")

    # Print summary
    if callback.train_losses:
        logger.info(f"Final train loss: {callback.train_losses[-1]:.4f}")
    if callback.eval_losses:
        logger.info(f"Final eval loss: {callback.eval_losses[-1]:.4f}")


def generate_experiment_entry(args, config, metrics) -> str:
    """Generate experiment tracking entry per docs template."""

    final_train_loss = metrics.get("final_train_loss", "N/A")
    final_eval_loss = metrics.get("final_eval_loss", "N/A")
    perplexity = "N/A"
    if final_eval_loss != "N/A" and final_eval_loss != "N/A":
        try:
            perplexity = f"{math.exp(float(final_eval_loss)):.2f}"
        except:
            pass

    if args.experiment_id:
        exp_id = args.experiment_id
    elif args.config:
        exp_id = f"LoRA-{args.config}"
    else:
        exp_id = f"r{args.lora_rank}-{args.target_modules}"

    entry = f"""## Experiment: {exp_id}

Date: {config.get("timestamp", datetime.now().isoformat())}
Goal: LoRA fine-tuning on S1 subset

### Setup

- Code revision (git SHA): {get_git_sha()}
- Training script: amt_lora_train.py
- Config file: {args.config or "manual"}
- Dataset variant: {config.get("train_data", "N/A")}
- Model init source: {config.get("model_path", "N/A")}
- Adaptation type: LoRA
- Hardware: {get_hardware_info()}

### Hyperparameters

- LoRA rank: {args.lora_rank}
- LoRA alpha: {args.lora_alpha}
- LoRA dropout: {args.lora_dropout}
- Target modules: {config.get("target_modules", [])}
- LR: {args.learning_rate}
- Batch: {args.per_device_train_batch_size} × {args.gradient_accumulation_steps} = {args.per_device_train_batch_size * args.gradient_accumulation_steps}
- Epochs: {args.num_epochs}
- Max steps: {args.max_steps if args.max_steps > 0 else "full epoch"}
- Seed: {args.seed}

### Outputs

- Checkpoint path: {args.output_dir}/final
- Logs path: {args.output_dir}/logs
- Metrics: {args.output_dir}/metrics.json

### Results

- Train loss trend: {format_loss_trend(metrics.get("train_losses", []))}
- Val loss: {final_eval_loss}
- Perplexity: {perplexity}
- Time: (see logs)

### A/B Against Baseline

- Baseline model: stanford-crfm/music-large-800k
- Prompt set: (TBD)
- Decoding settings: (TBD)
- Qualitative verdict: (TBD)

### Decision

- Keep / Drop / Re-run
- Next action:
"""
    return entry


def get_git_sha() -> str:
    try:
        import subprocess

        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent,
        )
        return result.stdout.strip()[:8]
    except:
        return "N/A"


def get_hardware_info() -> str:
    try:
        if torch.cuda.is_available():
            return f"GPU: {torch.cuda.get_device_name(0)}"
        return "CPU"
    except:
        return "Unknown"


def format_loss_trend(losses: List[float]) -> str:
    if not losses:
        return "N/A"
    if len(losses) <= 3:
        return " → ".join([f"{l:.4f}" for l in losses])
    return f"{losses[0]:.4f} → ... → {losses[-1]:.4f} ({len(losses)} steps)"


if __name__ == "__main__":
    main()
