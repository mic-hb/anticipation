from typing import Iterator
from pathlib import Path
import warnings

import math
import numpy as np

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, Sampler


class PreTokenizedDataset(Dataset):
    """
    Must be constructed with v2 sequence packing.
    """

    def __init__(self, path: Path) -> None:
        self.data = np.load(path, mmap_mode="r")
        assert self.data.flags["C_CONTIGUOUS"]

    @property
    def num_tokens(self) -> int:
        num_sequences = self.data.shape[0]
        num_tokens_per_sequence = self.data.shape[1]
        return num_sequences * num_tokens_per_sequence

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            input_ids = torch.from_numpy(self.data[idx]).to(dtype=torch.long)

        attention_mask = torch.ones_like(input_ids)
        labels = input_ids.clone()
        labels = torch.roll(labels, shifts=-1, dims=0)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class ResumableDistributedBatchSampler(Sampler[list[int]]):
    """
    DDP-aware batch sampler with exact mid-epoch resume semantics
    as long as num_workers=0 for the training DataLoader.

    State that gets checkpointed:
      - epoch
      - batches_seen in the current epoch

    The per-rank sample order is reconstructed deterministically from:
      seed + epoch
    """

    def __init__(
        self,
        dataset_size: int,
        batch_size: int,
        *,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = True,
        rank: int | None = None,
        world_size: int | None = None,
    ) -> None:
        super().__init__()
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")

        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last

        if world_size is None:
            if dist.is_available() and dist.is_initialized():
                world_size = dist.get_world_size()
            else:
                world_size = 1
        if rank is None:
            if dist.is_available() and dist.is_initialized():
                rank = dist.get_rank()
            else:
                rank = 0

        if not (0 <= rank < world_size):
            raise ValueError(
                f"invalid rank/world_size: rank={rank}, world_size={world_size}"
            )

        self.rank = rank
        self.world_size = world_size

        self.epoch = 0
        self.batches_seen = 0

    def state_dict(self) -> dict:
        return {
            "epoch": self.epoch,
            "batches_seen": self.batches_seen,
            "seed": self.seed,
            "shuffle": self.shuffle,
            "drop_last": self.drop_last,
            "dataset_size": self.dataset_size,
            "batch_size": self.batch_size,
            "world_size": self.world_size,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self.epoch = int(state_dict["epoch"])
        self.batches_seen = int(state_dict["batches_seen"])

        if int(state_dict["dataset_size"]) != self.dataset_size:
            raise ValueError(
                f"dataset_size mismatch: ckpt={state_dict['dataset_size']} current={self.dataset_size}"
            )
        if int(state_dict["batch_size"]) != self.batch_size:
            raise ValueError(
                f"batch_size mismatch: ckpt={state_dict['batch_size']} current={self.batch_size}"
            )
        if int(state_dict["world_size"]) != self.world_size:
            raise ValueError(
                f"world_size mismatch: ckpt={state_dict['world_size']} current={self.world_size}"
            )

    def set_epoch(self, epoch: int) -> None:
        """
        Call at the start of each epoch. If epoch changes, reset in-epoch cursor.
        """
        epoch = int(epoch)
        if epoch != self.epoch:
            self.epoch = epoch
            self.batches_seen = 0

    def _global_indices_for_epoch(self) -> list[int]:
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(self.dataset_size, generator=g).tolist()
        else:
            indices = list(range(self.dataset_size))

        # Mirror DistributedSampler semantics:
        if self.drop_last:
            total_size = (self.dataset_size // self.world_size) * self.world_size
            indices = indices[:total_size]
        else:
            total_size = (
                math.ceil(self.dataset_size / self.world_size) * self.world_size
            )
            padding = total_size - len(indices)
            if padding > 0:
                indices.extend(indices[:padding])

        return indices

    def _local_indices_for_epoch(self) -> list[int]:
        indices = self._global_indices_for_epoch()
        # rank-strided partition, matching the usual DistributedSampler pattern
        return indices[self.rank : len(indices) : self.world_size]

    def __iter__(self) -> Iterator[list[int]]:
        local_indices = self._local_indices_for_epoch()

        start = self.batches_seen * self.batch_size
        if start > len(local_indices):
            start = len(local_indices)

        for i in range(start, len(local_indices), self.batch_size):
            batch = local_indices[i : i + self.batch_size]
            if len(batch) < self.batch_size and self.drop_last:
                break

            yield batch
            self.batches_seen += 1

    def __len__(self) -> int:
        local_count = len(self._local_indices_for_epoch())

        if self.drop_last:
            total_batches = local_count // self.batch_size
        else:
            total_batches = math.ceil(local_count / self.batch_size)

        remaining = total_batches - self.batches_seen
        return max(0, remaining)
