import math
from pathlib import Path
from typing import Iterable
from functools import partial
from json import dumps
import multiprocessing as mp
import tempfile

import numpy as np
from tqdm.contrib.concurrent import process_map

from anticipation.v2.config import (
    AnticipationV2Settings,
    Vocab,
    DATASET_ROOT,
)
from anticipation.v2.tokenize import tokenize, MIDIFileIgnoredReason
from anticipation.v2.util import (
    iter_files,
    get_uuid_string,
    get_git_info,
    get_time_info,
)
from anticipation.v2.io import TokenSequenceBinaryFile, consolidate_bins


def _process_shard(
    shard_id_and_files_to_process: tuple[int, list[Path]],
    settings: AnticipationV2Settings,
    shards_container_path: Path,
) -> tuple[Path, dict[MIDIFileIgnoredReason, int]]:
    shard_id, files_to_process = shard_id_and_files_to_process
    work_dir = shards_container_path / f"./{shard_id}"
    work_dir.mkdir(exist_ok=True)

    shard_artifact_path = work_dir / f"{shard_id}_shard_dataset_processed.tmp.bin"
    try:
        ignored_files_summary = tokenize(
            files_to_process,
            output=shard_artifact_path,
            settings=settings,
        )
        return shard_artifact_path, ignored_files_summary
    except Exception as p:
        raise p
        # print(p)


def _get_dataset_shards(
    dataset_paths: Iterable[Path],
    num_shards: int,
) -> Iterable[tuple[int, list[Path]]]:
    all_files = []
    for dataset_path in sorted(dataset_paths):
        assert dataset_path.exists()
        assert dataset_path.is_dir()

        # get all files with specific file extensions
        all_files += list(iter_files(dataset_path, file_extensions=(".mid", ".midi")))

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


def get_dataset_file_from_paths(
    settings,
    dataset_paths,
    num_workers,
    parent_work_dir,
    shards_dir,
    save_to,
    do_shuffle: bool = False,
):
    process_one_with_args = partial(
        _process_shard, settings=settings, shards_container_path=shards_dir
    )
    shards = _get_dataset_shards(dataset_paths, num_workers)
    # --- multiprocess ---
    records: list[int] = process_map(
        process_one_with_args,
        shards,
        max_workers=num_workers,
        total=num_workers,
    )

    # single process
    bin_out_path = shards_dir / (save_to + ".bin")
    npy_out_path = parent_work_dir / (save_to + ".npy")
    consolidate_bins(
        list(shards_dir.rglob("*.bin")),
        out_path=bin_out_path,
        dtype=TokenSequenceBinaryFile.get_dtype_for_tokens(settings.vocab.VOCAB_SIZE),
        seq_len=settings.context_size,
    )
    loaded_arr = TokenSequenceBinaryFile.load_from_disk_to_numpy(
        bin_out_path, settings.context_size, settings.vocab.VOCAB_SIZE
    )
    if do_shuffle:
        np.random.seed(settings.train_data_split_shuffle_random_seed)
        loaded_arr = loaded_arr[np.random.permutation(loaded_arr.shape[0])]

    np.save(npy_out_path, loaded_arr)
    return npy_out_path, records


def main(put_shards_in_tmp: bool = True) -> None:
    # job settings
    dataset_path = DATASET_ROOT

    # --- settings and book-keeping ---
    lmd_dataset_path = dataset_path / "lmd_full"
    parent_work_dir = dataset_path / "tokenized_data"
    parent_work_dir.mkdir(exist_ok=True)
    uuid = get_uuid_string()
    parent_work_dir = parent_work_dir / uuid
    parent_work_dir.mkdir()
    dataset_generation_info = {
        "git_info": get_git_info(),
        "started_time": get_time_info(),
    }
    dataset_generation_info_save_to = parent_work_dir / "process_info.json"
    dataset_generation_info_save_to.write_text(
        dumps(dataset_generation_info, sort_keys=True, indent=4)
    )

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        if put_shards_in_tmp:
            shards_dir = td_path / "shards"
            shards_dir.mkdir(exist_ok=True)
        else:
            shards_dir = parent_work_dir / "shards"
            shards_dir.mkdir(exist_ok=True)

        settings = AnticipationV2Settings(
            vocab=Vocab(),
            num_autoregressive_seq_per_midi_file=1,
            num_span_anticipation_augmentations_per_midi_file=0,
            num_instrument_anticipation_augmentations_per_midi_file=0,
            num_random_anticipation_augmentations_per_midi_file=0,
            debug=False,
            num_workers_in_dataset_construction=10,
            train_data_split_shuffle_random_seed=42,
        )
        settings.save_to_disk(parent_work_dir)
        # -------------------

        lmd_splits = [x for x in lmd_dataset_path.iterdir() if x.is_dir()]
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
        for conf in all_processing_confs:
            shards_dir_local = shards_dir / conf["name"]
            shards_dir_local.mkdir(exist_ok=True)
            get_dataset_file_from_paths(
                settings,
                conf["dataset_paths"],
                settings.num_workers_in_dataset_construction,
                parent_work_dir=parent_work_dir,
                shards_dir=shards_dir_local,
                save_to=conf["name"],
                do_shuffle=conf["do_shuffle"],
            )


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
