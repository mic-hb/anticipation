from typing import Any
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

from symusic import Score

# see: https://huggingface.co/datasets/Metacreation/GigaMIDI
GIGA_MIDI_ENCLOSING_PATH = (Path(__file__).parent.parent.parent / "data") / "giga_midi"

if __name__ == "__main__":
    """
    From repo root, run:
    
        PYTHONPATH=. python scripts/v2/giga_midi_to_files.py
        
    If not yet downloaded, one might need to run: `hf auth login` because the dataset
    requires huggingface authentication. 
    """
    splits = [
        "train", "validation", "test"
    ]

    GIGA_MIDI_ENCLOSING_PATH.mkdir(parents=True, exist_ok=True)

    for split in splits:
        # if dataset is gated, run `hf auth login`
        dataset = load_dataset("Metacreation/GigaMIDI", split=split)
        split_enclosing_path = (GIGA_MIDI_ENCLOSING_PATH / split)
        split_enclosing_path.mkdir(parents=True, exist_ok=True)

        iter_obj = tqdm(dataset, mininterval=1.0, desc=f"Saving files in split: {split}")
        for i, sample in enumerate(iter_obj):
            sample: dict[str, Any]
            sample_file_name = sample["md5"]
            save_to = Path(split_enclosing_path/ f"{sample_file_name}.midi")
            try:
                score = Score.from_midi(sample["music"], sanitize_data=True)
                score.dump_midi(save_to)
            except RuntimeError:
                # can't save this file for some reason...
                pass
