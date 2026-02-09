from json import loads
import uuid
from pathlib import Path
import sys,time

import transformers
from transformers import AutoModelForCausalLM

from anticipation import ops
from anticipation.sample import generate, generate_ar
from anticipation.tokenize import extract_instruments
from anticipation.convert import events_to_midi, midi_to_events, lm_to_midi, lm_to_event
from anticipation.vocabs.localmidi import vocab as lmv
#from anticipation.config import *
#from anticipation.vocab import *

from datetime import datetime, timezone

def get_unique_midi_filename() -> str:
    dt = datetime.now(timezone.utc)
    ts = dt.strftime("%Y_%m_%d_%H_%M_%S")
    return f"{ts}_{uuid.uuid4().hex}"

def tokens_to_midi_file(tokens, save_to: Path):
    #mid = events_to_midi(tokens)
    mid = lm_to_midi(tokens, vocab=lmv)
    mid.save(save_to)
    print(f"Saved MIDI to: {save_to}")

if __name__ == "__main__":
    # pip install "numpy<2"
    # in seconds
    length = 10
    save_output_to_path = f"/home/mf867/anticipation/output/checkpoints/step-100000/{get_unique_midi_filename()}.mid"
    model_checkpoint_path = "/home/mf867/anticipation/output/checkpoints/step-100000/"
    model = AutoModelForCausalLM.from_pretrained(model_checkpoint_path).cuda()
    #unconditional_tokens = generate(model, start_time=0, end_time=length, top_p=.98, debug=True)
    tokens = generate_ar(model, start_time=0, end_time=length, inputs=None, controls=None, top_p=.98, debug=False)
    #p = Path("tokensfrozen.txt")
    #tokens = loads(p.read_text())
    #assert isinstance(tokens, list)
    #assert all(isinstance(x, int) for x in tokens)
    tokens_to_midi_file(tokens, save_output_to_path)
