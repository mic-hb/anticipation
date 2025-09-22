from argparse import ArgumentParser
from anticipation.convert import events_to_compound, lm_to_event
from anticipation.vocabs.localmidi import vocab as local_vocab
from anticipation.vocabs.tripletmidi import vocab as triplet_vocab
import importlib.util
import sys
from pathlib import Path

# load prepare functions from train/tokenize-new.py
module_path = Path("train/tokenize-new.py")
spec = importlib.util.spec_from_file_location("tokenize_new", module_path)
tokenize_new = importlib.util.module_from_spec(spec)
sys.modules["tokenize_new"] = tokenize_new
spec.loader.exec_module(tokenize_new)

prepare_local_midi = tokenize_new.prepare_local_midi
prepare_triplet_midi = tokenize_new.prepare_triplet_midi


if __name__ == '__main__':
    parser = ArgumentParser(description='compare local-midi and triplet-midi round-trips')
    parser.add_argument('filename',
        help='file containing a compound representation (.compound.txt)')
    args = parser.parse_args()

    # === local-midi parsing ===
    local_tokens, _, _, status = prepare_local_midi(
        args.filename, local_vocab, task="autoregress", transcript=False
    )
    if status > 0:
        print(f"local-midi had status={status}")

    # === triplet-midi parsing ===
    triplet_tokens, _, _, status = prepare_triplet_midi(
        args.filename, triplet_vocab, task="autoregress", transcript=False
    )
    if status > 0:
        print(f"triplet-midi had status={status}")

    # === convert back to compound ===
    local_events = lm_to_event(local_tokens, local_vocab)
    local_compound = events_to_compound(local_events)
    triplet_compound = events_to_compound(triplet_tokens)

    # === compare ===
    if local_compound == triplet_compound:
        print("Local-midi and Triplet-midi round-trip to the same compound representation.")
    else:
        print("Difference detected between local and triplet compound forms.")
        print(f"Triplet length: {len(triplet_compound)}, Local length: {len(local_compound)}")
        print("Triplet tail:", triplet_compound[-15:])
        print("Local tail:  ", local_compound[-15:])
