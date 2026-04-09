import io
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from anticipation.v2.config import AnticipationV2Settings
from anticipation.v2.sample import generate_ar_simple
from train.v2.hf_gpt2_rewrite import GPT2LMHeadModelLite

SAVE_RESULTS_TO = Path(__file__).parent / "results"
SAVE_RESULTS_TO.mkdir(exist_ok=True)


def main():
    # load an anticipatory music transformer
    checkpoint_dir = "output/slurm_logs/2632/checkpoints/step-100000"
    data_dir_lmd = "data/tokenized_datasets/lmd_full/6fb2094dfa7c0d16278dfaa4a401e3b8"
    settings = AnticipationV2Settings.load_from_disk(
        Path(data_dir_lmd) / "settings_6fb2094dfa7c0d16278dfaa4a401e3b8.json"
    )
    model = GPT2LMHeadModelLite.from_pretrained(checkpoint_dir).to("cuda").to(torch.float32)
    model = model.eval()
    print("Starting to generate...")
    midi_file = generate_ar_simple(
        model,
        settings,
        num_events_to_generate=500,
        top_p=0.98,
    )
    midi_file.save(str((SAVE_RESULTS_TO / "example_midi.mid").resolve()))

if __name__ == "__main__":
    """

        PYTHONPATH=. python scripts/v2/sample.py

    """
    main()
