import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import mido
from transformers import AutoModelForCausalLM

from anticipation import ops
from anticipation.convert import events_to_midi, midi_to_events
from anticipation.sample import generate
from anticipation.tokenize import extract_instruments
from anticipation.vocab import CONTROL_OFFSET, NOTE_OFFSET


DEFAULT_SMALL_MODEL = "stanford-crfm/music-small-800k"
DEFAULT_MEDIUM_MODEL = "stanford-crfm/music-medium-800k"
DEFAULT_LARGE_MODEL = "stanford-crfm/music-large-800k"


class SessionState:
    def __init__(self) -> None:
        self.source_track_names: dict[int, str] = {}
        self.unconditional_tokens: Optional[List[int]] = None
        self.prompted_history: Optional[List[int]] = None
        self.prompted_length: float = 0.0
        self.loaded_events: Optional[List[int]] = None
        self.loaded_segment: Optional[List[int]] = None
        self.span_history: Optional[List[int]] = None
        self.span_anticipated: Optional[List[int]] = None
        self.span_inpainted: Optional[List[int]] = None
        self.accompaniment_events: Optional[List[int]] = None
        self.accompaniment_melody: Optional[List[int]] = None
        self.accompaniment_history: Optional[List[int]] = None
        self.accompaniment_output: Optional[List[int]] = None
        self.ctrl_events: Optional[List[int]] = None
        self.ctrl_segment: Optional[List[int]] = None
        self.ctrl_melody: Optional[List[int]] = None
        self.ctrl_prompt: Optional[List[int]] = None
        self.ctrl_length: float = 0.0
        self.ctrl_proposal: Optional[List[int]] = None


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


GM_PROGRAM_NAMES = [
    "Acoustic Grand Piano", "Bright Acoustic Piano", "Electric Grand Piano", "Honky-tonk Piano",
    "Electric Piano 1", "Electric Piano 2", "Harpsichord", "Clavinet",
    "Celesta", "Glockenspiel", "Music Box", "Vibraphone",
    "Marimba", "Xylophone", "Tubular Bells", "Dulcimer",
    "Drawbar Organ", "Percussive Organ", "Rock Organ", "Church Organ",
    "Reed Organ", "Accordion", "Harmonica", "Tango Accordion",
    "Acoustic Guitar (nylon)", "Acoustic Guitar (steel)", "Electric Guitar (jazz)", "Electric Guitar (clean)",
    "Electric Guitar (muted)", "Overdriven Guitar", "Distortion Guitar", "Guitar Harmonics",
    "Acoustic Bass", "Electric Bass (finger)", "Electric Bass (pick)", "Fretless Bass",
    "Slap Bass 1", "Slap Bass 2", "Synth Bass 1", "Synth Bass 2",
    "Violin", "Viola", "Cello", "Contrabass",
    "Tremolo Strings", "Pizzicato Strings", "Orchestral Harp", "Timpani",
    "String Ensemble 1", "String Ensemble 2", "SynthStrings 1", "SynthStrings 2",
    "Choir Aahs", "Voice Oohs", "Synth Voice", "Orchestra Hit",
    "Trumpet", "Trombone", "Tuba", "Muted Trumpet",
    "French Horn", "Brass Section", "SynthBrass 1", "SynthBrass 2",
    "Soprano Sax", "Alto Sax", "Tenor Sax", "Baritone Sax",
    "Oboe", "English Horn", "Bassoon", "Clarinet",
    "Piccolo", "Flute", "Recorder", "Pan Flute",
    "Blown Bottle", "Shakuhachi", "Whistle", "Ocarina",
    "Lead 1 (square)", "Lead 2 (sawtooth)", "Lead 3 (calliope)", "Lead 4 (chiff)",
    "Lead 5 (charang)", "Lead 6 (voice)", "Lead 7 (fifths)", "Lead 8 (bass+lead)",
    "Pad 1 (new age)", "Pad 2 (warm)", "Pad 3 (polysynth)", "Pad 4 (choir)",
    "Pad 5 (bowed)", "Pad 6 (metallic)", "Pad 7 (halo)", "Pad 8 (sweep)",
    "FX 1 (rain)", "FX 2 (soundtrack)", "FX 3 (crystal)", "FX 4 (atmosphere)",
    "FX 5 (brightness)", "FX 6 (goblins)", "FX 7 (echoes)", "FX 8 (sci-fi)",
    "Sitar", "Banjo", "Shamisen", "Koto",
    "Kalimba", "Bag pipe", "Fiddle", "Shanai",
    "Tinkle Bell", "Agogo", "Steel Drums", "Woodblock",
    "Taiko Drum", "Melodic Tom", "Synth Drum", "Reverse Cymbal",
    "Guitar Fret Noise", "Breath Noise", "Seashore", "Bird Tweet",
    "Telephone Ring", "Helicopter", "Applause", "Gunshot",
]


def gm_name_from_instrument(instr: int) -> str:
    if instr == 128:
        return "Drums"
    if 0 <= instr < len(GM_PROGRAM_NAMES):
        return GM_PROGRAM_NAMES[instr]
    return f"Instrument {instr}"


def extract_source_track_names(midi_path: Path) -> dict[int, str]:
    midi = mido.MidiFile(str(midi_path))
    instrument_name_map: dict[int, str] = {}

    for track in midi.tracks:
        track_name = None
        instrument_name = None
        channel_program: dict[int, int] = {}
        channels_used: set[int] = set()
        for msg in track:
            if msg.type == "track_name" and track_name is None and msg.name.strip():
                track_name = msg.name.strip()
            elif msg.type == "instrument_name" and instrument_name is None and msg.name.strip():
                instrument_name = msg.name.strip()
            elif msg.type == "program_change":
                channel_program[msg.channel] = msg.program
                channels_used.add(msg.channel)
            elif msg.type in ("note_on", "note_off"):
                channels_used.add(msg.channel)

        preferred_name = track_name or instrument_name
        for channel in channels_used:
            instr = 128 if channel == 9 else channel_program.get(channel, 0)
            if preferred_name and instr not in instrument_name_map:
                instrument_name_map[instr] = preferred_name

    return instrument_name_map


def apply_track_names(mid: mido.MidiFile, source_name_map: Optional[dict[int, str]] = None) -> None:
    source_name_map = source_name_map or {}
    name_counts: dict[str, int] = {}

    for track in mid.tracks:
        channel = None
        program = 0
        for msg in track:
            if hasattr(msg, "channel") and channel is None:
                channel = msg.channel
            if msg.type == "program_change":
                program = msg.program
                if channel is None:
                    channel = msg.channel

        instr = 128 if channel == 9 else program
        base_name = source_name_map.get(instr) or gm_name_from_instrument(instr)

        count = name_counts.get(base_name, 0) + 1
        name_counts[base_name] = count
        final_name = base_name if count == 1 else f"{base_name} {count}"

        filtered = [msg for msg in track if msg.type not in ("track_name", "instrument_name")]
        track.clear()
        track.append(mido.MetaMessage("track_name", name=final_name, time=0))
        track.append(mido.MetaMessage("instrument_name", name=gm_name_from_instrument(instr), time=0))
        track.extend(filtered)


def save_midi(events: List[int], out_path: Path, source_name_map: Optional[dict[int, str]] = None) -> None:
    mid = events_to_midi(events)
    apply_track_names(mid, source_name_map)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mid.save(str(out_path))
    print(f"[OK] Saved MIDI: {out_path}")


def prompt_float(message: str, default: float) -> float:
    raw = input(f"{message} [{default}]: ").strip()
    if raw == "":
        return default
    return float(raw)


def prompt_int(message: str, default: int) -> int:
    raw = input(f"{message} [{default}]: ").strip()
    if raw == "":
        return default
    return int(raw)


def prompt_str(message: str, default: str) -> str:
    raw = input(f"{message} [{default}]: ").strip()
    return default if raw == "" else raw


def trim_and_translate(events: List[int], start_sec: float, length_sec: float) -> List[int]:
    clipped = ops.clip(events, start_sec, start_sec + length_sec)
    return ops.translate(clipped, -ops.min_time(clipped, seconds=False))


def summarize_instruments(events: List[int], label: str) -> None:
    print(f"[INFO] {label} instruments:", dict(ops.get_instruments(events)))


def task_unconditional_generation(model, state: SessionState, out_dir: Path, top_p: float) -> None:
    length = prompt_float("Unconditional generation length in seconds", 10.0)
    tokens = generate(model, start_time=0, end_time=length, top_p=top_p)
    state.unconditional_tokens = tokens
    out_path = out_dir / "01_unconditional.mid"
    save_midi(tokens, out_path, state.source_track_names)
    print(f"[INFO] Generated {len(tokens)//3} note events.")


def task_prompted_generation(model, state: SessionState, out_dir: Path, top_p: float) -> None:
    if state.unconditional_tokens is None:
        print("[WARN] No unconditional tokens found. Generating them first.")
        task_unconditional_generation(model, state, out_dir, top_p)
    if state.unconditional_tokens is None:
        return

    history = state.unconditional_tokens.copy()
    length = prompt_float("Current history length in seconds", 10.0)
    n = prompt_float("Seconds to generate per iteration", 5.0)
    rounds = prompt_int("How many interactive rounds?", 2)

    for idx in range(rounds):
        proposal = generate(
            model,
            start_time=length,
            end_time=length + n,
            inputs=history,
            top_p=top_p,
        )
        preview_path = out_dir / f"02_prompted_round_{idx+1}.mid"
        save_midi(proposal, preview_path, state.source_track_names)
        decision = prompt_str(
            f"Round {idx+1}: accept proposal? (y/n/stop)",
            "y",
        ).lower()
        if decision == "y":
            history = proposal
            length += n
        elif decision == "stop":
            break

    state.prompted_history = history
    state.prompted_length = length
    save_midi(history, out_dir / "02_prompted_final.mid", state.source_track_names)


def task_loading_own_midi(state: SessionState, out_dir: Path, midi_path: Path) -> None:
    state.source_track_names = extract_source_track_names(midi_path)
    events = midi_to_events(str(midi_path))
    state.loaded_events = events
    summarize_instruments(events, "Loaded MIDI")

    clip_len = prompt_float("Preview clip length (seconds)", 30.0)
    preview = ops.clip(events, 0, clip_len)
    save_midi(preview, out_dir / "03_loaded_preview.mid", state.source_track_names)

    seg_start = prompt_float("Segment start for editing (seconds)", 41.0)
    seg_len = prompt_float("Segment length for editing (seconds)", 16.0)
    segment = trim_and_translate(events, seg_start, seg_len)
    state.loaded_segment = segment
    save_midi(segment, out_dir / "03_loaded_segment.mid", state.source_track_names)


def task_span_infilling(model, state: SessionState, out_dir: Path, top_p: float) -> None:
    if state.loaded_segment is None:
        print("[WARN] No segment loaded. Run 'Loading your own MIDI' first.")
        return

    segment = state.loaded_segment
    history = ops.clip(segment, 0, 6, clip_duration=False)
    anticipated = [CONTROL_OFFSET + tok for tok in ops.clip(segment, 11, 16, clip_duration=False)]
    inpainted = generate(model, 6, 11, inputs=history, controls=anticipated, top_p=top_p)

    state.span_history = history
    state.span_anticipated = anticipated
    state.span_inpainted = inpainted

    save_midi(ops.combine(history, anticipated), out_dir / "04_span_controls_only.mid", state.source_track_names)
    save_midi(ops.combine(inpainted, anticipated), out_dir / "04_span_inpainted.mid", state.source_track_names)


def task_accompaniment(model, state: SessionState, out_dir: Path, top_p: float) -> None:
    if state.loaded_events is None:
        print("[WARN] No loaded MIDI found. Run 'Loading your own MIDI' first.")
        return

    events = state.loaded_events
    segment = trim_and_translate(events, 41, 20)
    summarize_instruments(segment, "Accompaniment source segment")

    non_melody, melody = extract_instruments(segment, [53])
    history = ops.clip(non_melody, 0, 5, clip_duration=False)
    accompaniment = generate(model, 5, 20, inputs=history, controls=melody, top_p=top_p, debug=False)
    output = ops.clip(ops.combine(accompaniment, melody), 0, 20, clip_duration=True)

    state.accompaniment_events = non_melody
    state.accompaniment_melody = melody
    state.accompaniment_history = history
    state.accompaniment_output = output

    save_midi(melody, out_dir / "05_accompaniment_melody.mid", state.source_track_names)
    save_midi(output, out_dir / "05_accompaniment_output.mid", state.source_track_names)


def task_control_loop_setup(state: SessionState, out_dir: Path, midi_path: Path) -> None:
    state.source_track_names = extract_source_track_names(midi_path)
    events = midi_to_events(str(midi_path))
    segment = trim_and_translate(events, 41, 45)
    summarize_instruments(segment, "Control-loop segment")
    non_melody, melody = extract_instruments(segment, [53])

    length = prompt_float("Initial control-loop prompt length (seconds)", 5.0)
    prompt = ops.clip(non_melody, 0, length, clip_duration=False)

    state.ctrl_events = non_melody
    state.ctrl_segment = segment
    state.ctrl_melody = melody
    state.ctrl_prompt = prompt
    state.ctrl_length = length
    state.ctrl_proposal = None

    save_midi(prompt, out_dir / "06_control_loop_prompt.mid", state.source_track_names)
    save_midi(melody, out_dir / "06_control_loop_melody.mid", state.source_track_names)


def _delete_instrument(events: List[int], instr: int) -> List[int]:
    return ops.delete(events, lambda token: (token[2] - NOTE_OFFSET) // (2 ** 7) == instr)


def _preview_combined(events: List[int], melody: List[int], start: float, end: float) -> List[int]:
    output = ops.clip(ops.combine(events, melody), start, end, clip_duration=True)
    return ops.translate(output, -ops.min_time(output, seconds=False))


def task_control_loop_run(model, state: SessionState, out_dir: Path, top_p: float) -> None:
    if state.ctrl_prompt is None or state.ctrl_melody is None:
        print("[WARN] Control loop not initialized. Run setup first.")
        return

    print("[INFO] Entering control loop. Actions: generate, inspect, accept, revise_instr, revert, save, quit")
    while True:
        action = prompt_str("Action", "generate").lower()

        if action == "generate":
            n = prompt_float("Seconds to generate", 5.0)
            nucleus_p = prompt_float("Nucleus top_p", top_p)
            proposal = generate(
                model,
                start_time=state.ctrl_length,
                end_time=state.ctrl_length + n,
                inputs=state.ctrl_prompt,
                controls=state.ctrl_melody,
                top_p=nucleus_p,
            )
            state.ctrl_proposal = proposal
            preview = _preview_combined(proposal, state.ctrl_melody, 0, state.ctrl_length + n)
            save_midi(preview, out_dir / "07_control_loop_preview.mid", state.source_track_names)
            print(f"[INFO] Generated proposal to t={state.ctrl_length + n:.2f}s")

        elif action == "inspect":
            if state.ctrl_proposal is None:
                print("[WARN] No proposal yet.")
                continue
            summarize_instruments(state.ctrl_proposal, "Current proposal")

        elif action == "accept":
            if state.ctrl_proposal is None:
                print("[WARN] No proposal to accept.")
                continue
            new_end = ops.max_time(state.ctrl_proposal)
            state.ctrl_prompt = state.ctrl_proposal
            state.ctrl_length = float(new_end)
            print(f"[INFO] Accepted proposal. New prompt length={state.ctrl_length:.2f}s")

        elif action == "revise_instr":
            if state.ctrl_proposal is None:
                print("[WARN] No proposal to revise.")
                continue
            instr = prompt_int("Instrument id to delete (e.g. 128 for drums)", 128)
            candidate = _delete_instrument(state.ctrl_proposal, instr)
            state.ctrl_proposal = candidate
            preview = _preview_combined(candidate, state.ctrl_melody, 0, state.ctrl_length + 5)
            save_midi(preview, out_dir / "07_control_loop_revise_instr.mid", state.source_track_names)
            print(f"[INFO] Deleted instrument {instr} from proposal.")

        elif action == "revert":
            if state.ctrl_proposal is None:
                print("[WARN] No proposal to revert.")
                continue
            reversion = prompt_float("Revert to timepoint (seconds)", state.ctrl_length + 2)
            candidate = ops.clip(state.ctrl_proposal, 0, reversion, clip_duration=False)
            state.ctrl_prompt = candidate
            state.ctrl_length = reversion
            state.ctrl_proposal = candidate
            save_midi(candidate, out_dir / "07_control_loop_reverted_prompt.mid", state.source_track_names)
            print(f"[INFO] Reverted and accepted prompt at t={reversion:.2f}s")

        elif action == "save":
            if state.ctrl_proposal is None:
                print("[WARN] Nothing to save yet.")
                continue
            save_midi(state.ctrl_proposal, out_dir / "07_control_loop_final.mid", state.source_track_names)

        elif action == "quit":
            print("[INFO] Leaving control loop.")
            break

        else:
            print("[WARN] Unknown action.")


def load_model(model_name: str, device: str):
    model = AutoModelForCausalLM.from_pretrained(model_name)
    if device == "cuda":
        model = model.cuda()
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Single interactive script for all getting_started.ipynb AMT examples/tasks."
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_LARGE_MODEL,
        help=(
            f"HuggingFace model name. Options include: {DEFAULT_SMALL_MODEL}, "
            f"{DEFAULT_MEDIUM_MODEL}, {DEFAULT_LARGE_MODEL}."
        ),
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Run model on cuda or cpu.",
    )
    parser.add_argument(
        "--midi",
        default="examples/strawberry.mid",
        help="Default MIDI file used in loading/infilling/accompaniment/control-loop tasks.",
    )
    parser.add_argument(
        "--top-p",
        default=0.98,
        type=float,
        help="Default nucleus sampling probability.",
    )
    parser.add_argument(
        "--out-dir",
        default="outputs/getting_started_interactive",
        help="Directory for generated MIDI files.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    midi_path = Path(args.midi)
    if not midi_path.is_absolute():
        midi_path = Path.cwd() / midi_path
    if not midi_path.exists():
        raise FileNotFoundError(f"MIDI not found: {midi_path}")

    print("[INFO] Loading model...")
    model = load_model(args.model, args.device)
    print("[INFO] Model ready.")

    state = SessionState()

    menu = """
=== Anticipatory Music Transformer Interactive Script ===
I. Simple Interaction
  1 - Unconditional generation
  2 - Prompted generation
  3 - Loading your own MIDI
  4 - Span Infilling
  5 - Accompaniment

II. Richer Interactive Control Flow
  6 - Loading initial note history and controls
  7 - The control loop

Other
  8 - Run all tasks once in order
  9 - Exit
"""

    while True:
        print(menu)
        choice = prompt_str("Choose an action", "9")
        if choice == "1":
            task_unconditional_generation(model, state, out_dir, args.top_p)
        elif choice == "2":
            task_prompted_generation(model, state, out_dir, args.top_p)
        elif choice == "3":
            task_loading_own_midi(state, out_dir, midi_path)
        elif choice == "4":
            task_span_infilling(model, state, out_dir, args.top_p)
        elif choice == "5":
            task_accompaniment(model, state, out_dir, args.top_p)
        elif choice == "6":
            task_control_loop_setup(state, out_dir, midi_path)
        elif choice == "7":
            task_control_loop_run(model, state, out_dir, args.top_p)
        elif choice == "8":
            task_unconditional_generation(model, state, out_dir, args.top_p)
            task_prompted_generation(model, state, out_dir, args.top_p)
            task_loading_own_midi(state, out_dir, midi_path)
            task_span_infilling(model, state, out_dir, args.top_p)
            task_accompaniment(model, state, out_dir, args.top_p)
            task_control_loop_setup(state, out_dir, midi_path)
            task_control_loop_run(model, state, out_dir, args.top_p)
        elif choice == "9":
            print("Done.")
            break
        else:
            print("[WARN] Unknown selection.")


if __name__ == "__main__":
    main()
