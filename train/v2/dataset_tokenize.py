import os

import random
import shutil
import pandas as pd
import argparse
import csv
import math
import multiprocessing as mp
from functools import partial
from json import dumps
from pathlib import Path
from typing import Any, Iterable, Union, Optional

import numpy as np
from numpy.lib.format import open_memmap
from anticipation.v2.config import (
    CONFIG_ROOT,
    DATASET_ROOT,
    LAKH_MIDI_FULL_PATH,
    TOKENIZED_DATASETS_SAVE_TO_PATH,
    AnticipationV2Settings,
    Vocab,
    make_vocab,
)
from anticipation.v2.io import (
    TokenSequenceBinaryFile,
    consolidate_bins,
    buffered_shuffle_bin_to_npy,
)
from anticipation.v2.tokenize import (
    TokenizationStatSummary,
    tokenize,
)
from anticipation.v2.util import (
    get_book_keeping_info,
    iter_files,
    temporary_directory,
    AtomicDirectory,
)
from tqdm.contrib.concurrent import process_map


def _process_shard(
    shard_id_and_files_to_process: tuple[int, list[Path]],
    settings: AnticipationV2Settings,
    shards_container_path: Path,
    is_training_split: bool,
    v1_mode: bool,
    convert_all_instruments_to_code: Optional[int] = None,
) -> tuple[Path, TokenizationStatSummary]:
    shard_id, files_to_process = shard_id_and_files_to_process
    work_dir = shards_container_path / f"./{shard_id}"
    work_dir.mkdir(exist_ok=True)
    shard_artifact_path = work_dir / f"{shard_id}_shard_dataset_processed.tmp.bin"

    # tokenize code here! this is the actual logic of what we are doing,
    # everything else is coordination
    tokenized_stats_summary = tokenize(
        files_to_process,
        output=shard_artifact_path,
        settings=settings,
        shard_id=shard_id,
        is_training_split=is_training_split,
        v1_mode=v1_mode,
        convert_all_instruments_to_code=convert_all_instruments_to_code,
    )
    return shard_artifact_path, tokenized_stats_summary


def _get_dataset_shards(
    dataset_paths: Iterable[Path],
    num_shards: int,
) -> Iterable[tuple[int, list[Path]]]:
    """
    This takes a bunch of paths of files to tokenize and splits it into n disjoint lists
    of paths - one per shard.
    """
    all_files = []
    for dataset_path in sorted(dataset_paths):
        assert dataset_path.exists()
        assert dataset_path.is_dir()

        # get all files with specific file extensions
        all_files += sorted(
            list(iter_files(dataset_path, file_extensions=(".mid", ".midi")))
        )

    total_files = len(all_files)

    # number of shards is equal to number of workers
    num_files_per_shard = math.ceil(total_files / num_shards)
    shards = [
        all_files[i : i + num_files_per_shard]
        for i in range(0, total_files, num_files_per_shard)
    ]

    # give each shard an ID, just incrementing
    shards_with_id = enumerate(shards)
    return shards_with_id


def _get_dataset_file_from_paths(
    settings: AnticipationV2Settings,
    dataset_paths: Iterable[Path],
    num_workers: int,
    parent_work_dir: Path,
    shards_dir: Path,
    save_to: str,
    do_shuffle: bool,
    is_training_split: bool,
    v1_mode: bool,
    convert_all_instruments_to_code: Optional[int] = None,
) -> tuple[Path, list[tuple[Path, TokenizationStatSummary]]]:
    # get division of work
    shards = _get_dataset_shards(dataset_paths, num_workers)

    # this is where the tokenization code is actually called
    process_one_with_args = partial(
        _process_shard,
        settings=settings,
        shards_container_path=shards_dir,
        is_training_split=is_training_split,
        v1_mode=v1_mode,
        convert_all_instruments_to_code=convert_all_instruments_to_code,
    )

    # run tokenization, keep note of where results are saved
    # (to intermediate shards), as well as any files that are ignored
    records: list[tuple[Path, TokenizationStatSummary]] = process_map(
        process_one_with_args,
        shards,
        max_workers=num_workers,
        total=num_workers,
        desc="Gathering Workers",
        position=0,
    )

    # here we take all the intermediate raw binaries created by numpy and
    # consolidate them into a single file, this time with a `.npy` extension
    # so it is clearer how to load it
    bin_out_path = shards_dir / (save_to + ".bin")
    npy_out_path = parent_work_dir / (save_to + ".npy")
    dtype = TokenSequenceBinaryFile.get_dtype_for_tokens(settings.vocab.total_tokens())
    consolidate_bins(
        list(shards_dir.rglob("*.bin")),
        out_path=bin_out_path,
        dtype=dtype,
        seq_len=settings.context_size,
    )
    if do_shuffle:
        buffered_shuffle_bin_to_npy(
            bin_path=bin_out_path,
            npy_path=npy_out_path,
            dtype=dtype,
            seq_len=settings.context_size,
            seed=settings.train_data_split_shuffle_random_seed,
        )
    else:
        # -----
        loaded_arr = TokenSequenceBinaryFile.load_from_disk_to_numpy(
            bin_out_path, settings.context_size, settings.vocab.total_tokens()
        )

        # save the consolidated samples to single numpy
        np.save(npy_out_path, loaded_arr)

    return npy_out_path, records


def get_lakh_midi_splits_and_configs(
    lahk_midi_dataset_parent_path: Path,
) -> list[dict[str, Any]]:
    """
    Follow the Lakh MIDI convention of the splits being:
    - test: "f", not shuffled
    - validation: "e", not shuffled
    - train: everything else, globally shuffled
    """
    lmd_splits = [x for x in lahk_midi_dataset_parent_path.iterdir() if x.is_dir()]
    # sort all of these for determinism
    lmd_test = sorted([x for x in lmd_splits if x.name == "f"])
    lmd_valid = sorted([x for x in lmd_splits if x.name == "e"])
    lmd_train = sorted([x for x in lmd_splits if x.name not in ("f", "e")])
    all_processing_confs = [
        {
            "name": "train",
            "dataset_paths": lmd_train,
            "do_shuffle": True,
        },
        {
            "name": "valid",
            "dataset_paths": lmd_valid,
            "do_shuffle": False,
        },
        {
            "name": "test",
            "dataset_paths": lmd_test,
            "do_shuffle": False,
        },
    ]
    return all_processing_confs


def _tokenize_dataset_in_parallel(
    settings: AnticipationV2Settings,
    raw_data_enclosing_path: Path,
    save_all_dataset_files_to: Path,
    put_shards_in_tmp: bool,
    split_confs: list[dict[str, Any]],
    v1_mode: bool,
) -> dict[str, Any]:
    with temporary_directory() as td:
        td_path = Path(td)
        if put_shards_in_tmp:
            shards_dir = td_path / "shards"
            shards_dir.mkdir(exist_ok=True)
        else:
            # in this case, we don't use the temp dir that is
            # given to us by the context manager
            shards_dir = save_all_dataset_files_to / "shards"
            shards_dir.mkdir(exist_ok=True)

        # info for all splits
        ignored_files = []
        all_dataset_stats: dict[str, Union[int, float]] = {
            x: 0 for x in TokenizationStatSummary.get_int_fields()
        }

        for conf in split_confs:
            # info just for current split
            curr_split_ignored_files = []
            curr_split_dataset_stats: dict[str, Union[int, float]] = {
                x: 0 for x in TokenizationStatSummary.get_int_fields()
            }

            split_name = conf["name"]
            # create shard dir
            shards_dir_local = shards_dir / split_name
            shards_dir_local.mkdir(exist_ok=True)

            # process shard
            npy_path, file_results = _get_dataset_file_from_paths(
                settings,
                conf["dataset_paths"],
                settings.num_workers_in_dataset_construction,
                parent_work_dir=save_all_dataset_files_to,
                shards_dir=shards_dir_local,
                save_to=split_name,
                do_shuffle=conf["do_shuffle"],
                is_training_split=(split_name == "train"),
                v1_mode=v1_mode,
                convert_all_instruments_to_code=conf.get(
                    "convert_all_instruments_to_code", None
                ),
            )

            # gather any ignored file results
            for f in file_results:
                shard_path, dataset_stats = f
                for k in all_dataset_stats:
                    all_dataset_stats[k] += getattr(dataset_stats, k)
                    curr_split_dataset_stats[k] += getattr(dataset_stats, k)

                # handle ignored files
                for reason, files_list in dataset_stats.ignored_files.items():
                    for file in files_list:
                        ignored_file = {
                            "split": conf["name"],
                            "shard": shard_path.name,
                            # e.g. TOO_FEW_EVENTS
                            "reason": reason.name,
                            # e.g. f9aad86bfb384b22875d40ef15be023d.mid
                            "file": str(file.relative_to(raw_data_enclosing_path)),
                        }
                        curr_split_ignored_files.append(ignored_file)
                        ignored_files.append(ignored_file)

            split_stat_path = Path(
                save_all_dataset_files_to / f"stats_{split_name}.json"
            )
            curr_split_dataset_stats = _add_more_info_to_dataset_stats(
                curr_split_dataset_stats, settings, curr_split_ignored_files
            )
            split_stat_path.write_text(
                dumps(curr_split_dataset_stats, sort_keys=True, indent=4)
            )

        # write all the ignored files to disk for awareness
        if ignored_files:
            field_names = list(ignored_files[0].keys())
            with open(
                save_all_dataset_files_to / "ignored_files.csv", "w", newline=""
            ) as file:
                writer = csv.DictWriter(file, fieldnames=field_names)  # noqa
                writer.writeheader()
                writer.writerows(ignored_files)

        # write dataset stats
        stat_path = Path(save_all_dataset_files_to / "stats.json")
        all_dataset_stats = _add_more_info_to_dataset_stats(
            all_dataset_stats, settings, ignored_files
        )
        stat_path.write_text(dumps(all_dataset_stats, sort_keys=True, indent=4))

    return all_dataset_stats


def _add_more_info_to_dataset_stats(
    all_dataset_stats: dict, settings: AnticipationV2Settings, ignored_files: list[dict]
) -> dict:
    all_dataset_stats["total_tokens"] = (
        settings.context_size * all_dataset_stats["num_sequences"]
    )
    all_dataset_stats["total_time_in_sec"] = (
        all_dataset_stats["total_time_in_midi_ticks"] / settings.time_resolution
    )
    all_dataset_stats["total_time_in_minutes"] = (
        all_dataset_stats["total_time_in_sec"] / 60
    )
    all_dataset_stats["total_time_in_sec_before_augmentation"] = (
        all_dataset_stats["total_time_in_midi_ticks_before_augmentation"]
        / settings.time_resolution
    )
    all_dataset_stats["total_time_in_minutes_before_augmentation"] = (
        all_dataset_stats["total_time_in_sec_before_augmentation"] / 60
    )
    all_dataset_stats["total_ignored_files"] = len(ignored_files)
    return all_dataset_stats


def _write_book_keeping_info_and_get_dataset_enclosing_path(
    settings: AnticipationV2Settings,
    save_tokenized_dataset_to: Path,
) -> Path:
    """
    This function writes some bookkeeping information to an enclosing directory.

    Each tokenized dataset has some metadata associated with it, mainly
    - the global project settings used during generation (settings_<md5>.json)
    - process information (process_info.json)
        - the time tokenization code started to run
        - the git branch and the commit that was used to create dataset
        - a UUID associated with the dataset. We made a UUID each time this is run
          to mitigate any overwriting of existing datasets
    - any files from the dataset that were ignored and roughly why (ignored_files.csv)

    At the end of running this script, the data directory should look approximately
    like this:

    ├── lmd_full (the lakh midi dataset)
    │ ├── 0
    │ ├── 1
    ...
    │ └── f
    └── tokenized_data (where we save our tokenized data)
        └── <UUID> (e.g. 030fb457e4884ad487ef5b4edd415414)
            ├── ignored_files.csv
            ├── process_info.json
            ├── settings_<md5 hash>.json
            ├── test.npy
            ├── train.npy
            └── valid.npy

    This function handles:
    - writing the settings and process info
    - generating a UUID, creating a directory using this UUID and returns it
    """
    dataset_generation_info = get_book_keeping_info()
    work_dir = save_tokenized_dataset_to

    # save bookkeeping info
    dataset_generation_info_save_to = work_dir / "book_keeping_info.json"
    dataset_generation_info_save_to.write_text(
        dumps(dataset_generation_info, sort_keys=True, indent=4)
    )

    # save project settings
    settings.save_to_disk(work_dir)
    return work_dir


def unique_name(p: Path, src_root: Path) -> str:
    rel = p.relative_to(src_root)
    return "__".join(rel.parts)


def split_files(files, train_pct=0.8, val_pct=0.1, test_pct=0.1, seed=None):
    if not (0 <= train_pct <= 1 and 0 <= val_pct <= 1 and 0 <= test_pct <= 1):
        raise ValueError("All split percentages must be in [0, 1].")

    total = train_pct + val_pct + test_pct
    if abs(total - 1.0) > 1e-8:
        raise ValueError(f"Split percentages must sum to 1. Got {total}")

    files = list(files)
    rng = random.Random(seed)
    rng.shuffle(files)

    n = len(files)
    n_train = int(n * train_pct)
    n_val = int(n * val_pct)

    train = files[:n_train]
    val = files[n_train : n_train + n_val]
    test = files[n_train + n_val :]

    return train, val, test


def get_splits(raw_data_enclosing_path: Path) -> list[dict[str, Any]]:
    if (
        raw_data_enclosing_path == LAKH_MIDI_FULL_PATH
        or raw_data_enclosing_path.parts[-1] == "lmd_full"
    ):
        # LAKH MIDI
        return get_lakh_midi_splits_and_configs(raw_data_enclosing_path)
    elif raw_data_enclosing_path.parts[-1] == "giga_midi":
        # GIGA MIDI
        # get these files by running scripts/v2/giga_midi_to_files.py
        return [
            {
                "name": "train",
                "dataset_paths": [raw_data_enclosing_path / "train"],
                "do_shuffle": True,
            },
            {
                "name": "valid",
                "dataset_paths": [raw_data_enclosing_path / "validation"],
                "do_shuffle": False,
            },
            {
                "name": "test",
                "dataset_paths": [raw_data_enclosing_path / "test"],
                "do_shuffle": False,
            },
        ]
    elif raw_data_enclosing_path.parts[-1] == "maestro-v3.0.0":
        dataset_info = raw_data_enclosing_path / "maestro-v3.0.0.csv"
        df = pd.read_csv(dataset_info)

        train_df = df[df["split"] == "train"]
        train_files = train_df["midi_filename"].tolist()
        train_path = raw_data_enclosing_path / "train"
        if not (train_path.exists() and train_path.is_dir()):
            train_path.mkdir()
            for fname in train_files:
                shutil.copy2(
                    raw_data_enclosing_path / fname, train_path / Path(fname).parts[-1]
                )

        valid_df = df[df["split"] == "validation"]
        valid_files = valid_df["midi_filename"].tolist()
        valid_path = raw_data_enclosing_path / "validation"
        if not (valid_path.exists() and valid_path.is_dir()):
            valid_path.mkdir()
            for fname in valid_files:
                shutil.copy2(
                    raw_data_enclosing_path / fname, valid_path / Path(fname).parts[-1]
                )

        test_df = df[df["split"] == "test"]
        test_files = test_df["midi_filename"].tolist()
        test_path = raw_data_enclosing_path / "test"
        if not (test_path.exists() and test_path.is_dir()):
            test_path.mkdir()
            for fname in test_files:
                shutil.copy2(
                    raw_data_enclosing_path / fname, test_path / Path(fname).parts[-1]
                )

        return [
            {
                "name": "train",
                "dataset_paths": [raw_data_enclosing_path / "train"],
                "do_shuffle": True,
            },
            {
                "name": "valid",
                "dataset_paths": [raw_data_enclosing_path / "validation"],
                "do_shuffle": True,
            },
            {
                "name": "test",
                "dataset_paths": [raw_data_enclosing_path / "test"],
                "do_shuffle": False,
            },
        ]
    elif raw_data_enclosing_path.parts[-1] == "adl-piano-midi":
        splits_path = raw_data_enclosing_path / "splits"
        if not (splits_path.exists() and splits_path.is_dir()):
            splits_path.mkdir()
            all_files = sorted(
                list(
                    iter_files(
                        raw_data_enclosing_path, file_extensions=(".mid", ".midi")
                    )
                )
            )
            train, val, test = split_files(
                all_files, train_pct=0.8, val_pct=0.1, test_pct=0.1, seed=42
            )
            assert len(train) + len(val) + len(test) == len(all_files)

            # careful here, there are non-unique filenames, so we need to prefix
            # the directory they came from to preserve it
            train_path = splits_path / "train"
            train_path.mkdir()
            for p in train:
                # copy2 will silently overwrite files of same name
                shutil.copy2(p, train_path / unique_name(p, raw_data_enclosing_path))

            val_path = splits_path / "validation"
            val_path.mkdir()
            for p in val:
                # copy2 will silently overwrite files of same name
                shutil.copy2(p, val_path / unique_name(p, raw_data_enclosing_path))

            test_path = splits_path / "test"
            test_path.mkdir()
            for p in test:
                # copy2 will silently overwrite files of same name
                shutil.copy2(p, test_path / unique_name(p, raw_data_enclosing_path))

            num_moved = (
                len(list(iter_files(train_path, file_extensions=(".mid", ".midi"))))
                + len(list(iter_files(val_path, file_extensions=(".mid", ".midi"))))
                + len(list(iter_files(test_path, file_extensions=(".mid", ".midi"))))
            )
            assert num_moved == len(all_files), f"{num_moved}"

        return [
            {
                "name": "train",
                "dataset_paths": [splits_path / "train"],
                "do_shuffle": True,
                # piano
                "convert_all_instruments_to_code": 0,
            },
            {
                "name": "valid",
                "dataset_paths": [splits_path / "validation"],
                "do_shuffle": True,
                "convert_all_instruments_to_code": 0,
            },
            {
                "name": "test",
                "dataset_paths": [splits_path / "test"],
                "do_shuffle": False,
                "convert_all_instruments_to_code": 0,
            },
        ]
    else:
        return [
            {
                "name": "train",
                "dataset_paths": [raw_data_enclosing_path],
                "do_shuffle": True,
            }
        ]


def main(
    settings_path: Path,
    put_shards_in_tmp: bool,
    raw_data_enclosing_path: Path,
    v1_mode: bool = False,
) -> Path:
    dataset_enclosing_path = raw_data_enclosing_path.parts[-1]
    settings = AnticipationV2Settings.load_from_disk(settings_path)

    put_tokenized_datasets_in_dir = (
        TOKENIZED_DATASETS_SAVE_TO_PATH / dataset_enclosing_path
    ) / settings.md5_hash()

    with AtomicDirectory(
        put_tokenized_datasets_in_dir, overwrite=False, keep_temp_on_error=False
    ) as txn:
        # do the work now, no more config past this point
        _tokenize_dataset_in_parallel(
            settings,
            raw_data_enclosing_path,
            _write_book_keeping_info_and_get_dataset_enclosing_path(settings, txn.path),
            put_shards_in_tmp,
            get_splits(raw_data_enclosing_path),
            v1_mode=v1_mode,
        )

    return put_tokenized_datasets_in_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dataset Tokenization Script")
    parser.add_argument(
        "--dataset_type",
        type=str,
        default="lakh",
        choices=["lakh", "aria", "transcripts", "giga_midi", "maestro3", "adl_piano"],
        help=(
            "Which dataset to tokenize. These are expected to be in specific locations in the ./data/ folder"
        ),
    )
    parser.add_argument(
        "--v1_mode",
        action="store_true",
        help="use v1 tokenization mode (AR only supported)",
    )
    _args = parser.parse_args()
    return _args


if __name__ == "__main__":
    """
    Example:

        PYTHONPATH=. python train/v2/dataset_tokenize.py --dataset_type lakh --v1_mode

    """
    mp.set_start_method("spawn", force=True)
    args = parse_args()

    # --- create settings and vocabulary ---
    max_note_duration_in_seconds = 10
    time_resolution = 100
    tick_token_every_n_ticks = 0
    num_workers = 16

    n = len(os.sched_getaffinity(0))

    my_vocab: Vocab = make_vocab(
        tick_token_every_n_ticks=tick_token_every_n_ticks,
        time_resolution=time_resolution,
        max_note_duration_in_seconds=max_note_duration_in_seconds,
        use_controls=False,
    )
    if args.dataset_type == "lakh":
        max_instr = 16
    else:
        # no limit
        max_instr = 10_000

    if args.dataset_type == "maestro3":
        # too small - if 16 workers, some won't have any work to do
        num_workers = 1
        max_instr = 10_000

    if args.dataset_type == "adl_piano":
        num_workers = 4

    assert n >= num_workers, f"not enough CPUs. Need: {num_workers}, Have: {n}"
    to_create = AnticipationV2Settings(
        vocab=my_vocab,
        delta=5,
        context_size=1024,
        # filter settings
        max_track_instruments=max_instr,
        max_note_duration_in_seconds=max_note_duration_in_seconds,
        # data mixture and augmentation settings
        num_autoregressive_seq_per_midi_file=1,
        num_span_anticipation_augmentations_per_midi_file=0,
        num_instrument_anticipation_augmentations_per_midi_file=0,
        # system-like settings
        num_workers_in_dataset_construction=num_workers,
        do_clip_overlapping_durations_in_midi_conversion=False,
        # time settings
        tick_token_every_n_ticks=tick_token_every_n_ticks,
        time_resolution=time_resolution,
    )
    settings_file_path = to_create.save_to_disk(CONFIG_ROOT)
    configs = {
        "lakh": {
            "settings": settings_file_path,
            "raw_data_enclosing_path": DATASET_ROOT / "lmd_full",
        },
        "transcripts": {
            "settings": settings_file_path,
            "raw_data_enclosing_path": DATASET_ROOT / "transcripts",
        },
        "aria": {
            # can use the same config as local lakh
            "settings": settings_file_path,
            "raw_data_enclosing_path": DATASET_ROOT / "aria-midi-v1-pruned-ext",
        },
        "adl_piano": {
            "settings": settings_file_path,
            "raw_data_enclosing_path": DATASET_ROOT / "adl-piano-midi",
        },
        "maestro3": {
            "settings": settings_file_path,
            "raw_data_enclosing_path": DATASET_ROOT / "maestro-v3.0.0",
        },
        "giga_midi": {
            "settings": settings_file_path,
            "raw_data_enclosing_path": DATASET_ROOT / "giga_midi",
        },
    }
    dataset_choice = configs[args.dataset_type]
    output_place = main(
        settings_path=dataset_choice["settings"],
        put_shards_in_tmp=True,
        raw_data_enclosing_path=dataset_choice["raw_data_enclosing_path"],
        v1_mode=args.v1_mode,
    )
    print(f"SUCCESS. Saved {args.dataset_type} to: {output_place}")
