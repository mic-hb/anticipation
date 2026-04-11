from datetime import datetime, timezone
import io
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from anticipation.v2.config import AnticipationV2Settings
from anticipation.v2.sample import generate_ar_simple as v2_generate_ar_simple
from anticipation.sample import generate_ar as v1_generate_ar
from anticipation.convert import events_to_midi as v1_events_to_midi
from train.v2.hf_gpt2_rewrite import GPT2LMHeadModelLite

SAVE_RESULTS_TO = Path(__file__).parent / "results"
SAVE_RESULTS_TO.mkdir(exist_ok=True)


def get_time_as_string() -> str:
    dt = datetime.now(timezone.utc)
    ts = dt.strftime("%Y_%m_%d_%H_%M_%S")
    return f"{ts}"


def main():
    # load an anticipatory music transformer
    #checkpoint_dir = "output/slurm_logs/2632/checkpoints/step-100000"
    checkpoint_dir = "/home/mf867/anticipation_isolated/anticipation/output/slurm_logs/232666/checkpoints/step-20000"
    #data_dir_lmd = "data/tokenized_datasets/lmd_full/6fb2094dfa7c0d16278dfaa4a401e3b8"
    data_dir_lmd_base = "data/tokenized_datasets/lakh_baseline/b0d0dbce322fc3318387b6cc12cf096a"
    # /home/mf867/anticipation_isolated/anticipation/data/tokenized_datasets/lakh_baseline/b0d0dbce322fc3318387b6cc12cf096a
    settings = AnticipationV2Settings.load_from_disk(
        Path(data_dir_lmd_base) / "settings_b0d0dbce322fc3318387b6cc12cf096a.json"
    )
    model = GPT2LMHeadModelLite.from_pretrained(checkpoint_dir).to("cuda").to(torch.float32)
    model = model.eval()
    print("Starting to generate...")
    if settings.tick_token_every_n_ticks != 0:
        midi_file = v2_generate_ar_simple(
            model,
            settings,
            num_events_to_generate=500,
            top_p=0.98,
        )
    else:
        outputs = v1_generate_ar(model, 0, 5, top_p=0.98)
        midi_file = v1_events_to_midi(outputs)

    midi_file.save(
        str((SAVE_RESULTS_TO / f"sample_{get_time_as_string()}_{settings.md5_hash()}.mid").resolve())
    )

if __name__ == "__main__":
    """

        PYTHONPATH=. python scripts/v2/sample.py

    """
    main()
