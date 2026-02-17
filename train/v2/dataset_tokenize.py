import csv
import math
from pathlib import Path
from typing import Iterable, Any, Union
from functools import partial
from json import dumps
import multiprocessing as mp
import tempfile

import numpy as np
from tqdm.contrib.concurrent import process_map

from anticipation.v2.config import (
    AnticipationV2Settings,
    CONFIG_ROOT,
    LAKH_MIDI_FULL_PATH,
    TOKENIZED_DATASETS_SAVE_TO_PATH,
)
from anticipation.v2.tokenize import (
    tokenize,
    TokenizationStatSummary,
)
from anticipation.v2.util import (
    iter_files,
    get_book_keeping_info,
)
from anticipation.v2.io import TokenSequenceBinaryFile, consolidate_bins


def _process_shard(
    shard_id_and_files_to_process: tuple[int, list[Path]],
    settings: AnticipationV2Settings,
    shards_container_path: Path,
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
) -> tuple[Path, list[tuple[Path, TokenizationStatSummary]]]:
    # get division of work
    shards = _get_dataset_shards(dataset_paths, num_workers)

    # this is where the tokenization code is actually called
    process_one_with_args = partial(
        _process_shard, settings=settings, shards_container_path=shards_dir
    )

    # run tokenization, keep note of where results are saved
    # (to intermediate shards), as well as any files that are ignored
    records: list[tuple[Path, TokenizationStatSummary]] = process_map(
        process_one_with_args,
        shards,
        max_workers=num_workers,
        total=num_workers,
        desc="Gathering Workers",
    )

    # here we take all the intermediate raw binaries created by numpy and
    # consolidate them into a single file, this time with a `.npy` extension
    # so it is clearer how to load it
    bin_out_path = shards_dir / (save_to + ".bin")
    npy_out_path = parent_work_dir / (save_to + ".npy")
    consolidate_bins(
        list(shards_dir.rglob("*.bin")),
        out_path=bin_out_path,
        dtype=TokenSequenceBinaryFile.get_dtype_for_tokens(
            settings.vocab.total_tokens()
        ),
        seq_len=settings.context_size,
    )
    loaded_arr = TokenSequenceBinaryFile.load_from_disk_to_numpy(
        bin_out_path, settings.context_size, settings.vocab.total_tokens()
    )
    if do_shuffle:
        # NB: this won't work for huge datasets that don't fit in ram, might need
        # to fix later
        np.random.seed(settings.train_data_split_shuffle_random_seed)
        loaded_arr = loaded_arr[np.random.permutation(loaded_arr.shape[0])]

    # save the consolidated samples to single numpy
    np.save(npy_out_path, loaded_arr)
    return npy_out_path, records


def _get_lakh_midi_splits_and_configs(
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
) -> dict[str, int]:
    all_dataset_stats: dict[str, Union[int, float]] = {
        x: 0 for x in TokenizationStatSummary.get_int_fields()
    }
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        if put_shards_in_tmp:
            shards_dir = td_path / "shards"
            shards_dir.mkdir(exist_ok=True)
        else:
            # in this case, we don't use the temp dir that is
            # given to us by the context manager
            shards_dir = save_all_dataset_files_to / "shards"
            shards_dir.mkdir(exist_ok=True)

        ignored_files = []
        for conf in split_confs:
            # create shard dir
            shards_dir_local = shards_dir / conf["name"]
            shards_dir_local.mkdir(exist_ok=True)

            # process shard
            npy_path, file_results = _get_dataset_file_from_paths(
                settings,
                conf["dataset_paths"],
                settings.num_workers_in_dataset_construction,
                parent_work_dir=save_all_dataset_files_to,
                shards_dir=shards_dir_local,
                save_to=conf["name"],
                do_shuffle=conf["do_shuffle"],
            )

            # gather any ignored file results
            for f in file_results:
                shard_path, dataset_stats = f
                for k in all_dataset_stats:
                    all_dataset_stats[k] += getattr(dataset_stats, k)

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
                        ignored_files.append(ignored_file)

        # write all the ignored files to disk for awareness
        field_names = list(ignored_files[0].keys())
        with open(
            save_all_dataset_files_to / "ignored_files.csv", "w", newline=""
        ) as file:
            writer = csv.DictWriter(file, fieldnames=field_names)  # noqa
            writer.writeheader()
            writer.writerows(ignored_files)

        # write dataset stats
        stat_path = Path(save_all_dataset_files_to / "stats.json")
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
        stat_path.write_text(dumps(all_dataset_stats, sort_keys=True, indent=4))

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

    # save all subsequent dataset files to `work_dir` - do not allow
    # overwriting folder of same name! That means it was already generated
    work_dir = save_tokenized_dataset_to / settings.md5_hash()
    work_dir.mkdir(exist_ok=False)

    # save bookkeeping info
    dataset_generation_info_save_to = work_dir / "book_keeping_info.json"
    dataset_generation_info_save_to.write_text(
        dumps(dataset_generation_info, sort_keys=True, indent=4)
    )

    # save project settings
    settings.save_to_disk(work_dir)
    return work_dir


def get_splits(raw_data_enclosing_path: Path):
    if (
        raw_data_enclosing_path == LAKH_MIDI_FULL_PATH
        or raw_data_enclosing_path.parts[-1] == "lmd_full"
    ):
        return _get_lakh_midi_splits_and_configs(raw_data_enclosing_path)
    else:
        return [
            {
                "name": "train",
                "dataset_paths": [raw_data_enclosing_path],
                "do_shuffle": True,
            }
        ]


def main(
    settings_path: Path, put_shards_in_tmp: bool, raw_data_enclosing_path: Path
) -> None:
    dataset_enclosing_path = raw_data_enclosing_path.parts[-1]
    put_tokenized_datasets_in_dir = (
        TOKENIZED_DATASETS_SAVE_TO_PATH / dataset_enclosing_path
    )
    put_tokenized_datasets_in_dir.mkdir(exist_ok=True, parents=True)

    settings = AnticipationV2Settings.load_from_disk(settings_path)

    # do the work now, no more config past this point
    _tokenize_dataset_in_parallel(
        settings,
        raw_data_enclosing_path,
        _write_book_keeping_info_and_get_dataset_enclosing_path(
            settings, put_tokenized_datasets_in_dir
        ),
        put_shards_in_tmp,
        get_splits(raw_data_enclosing_path),
    )


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    # TODO: do argparse thing
    ar_only_settings = (
        CONFIG_ROOT
        / "ar_only_local_midi_settings_b1f4b64911a603018ed67a154db6fb16.json"
    )
    tokenize_data_at_path = Path("/Users/admin/Documents/RESEARCH/anticipation_v2/anticipation/data/lmd_full")
    # tokenize_data_at_path = Path(
    #     "/Users/admin/Documents/RESEARCH/anticipation_v2/anticipation/data/transcripts"
    # )
    main(
        settings_path=ar_only_settings,
        put_shards_in_tmp=True,
        raw_data_enclosing_path=tokenize_data_at_path,
    )
