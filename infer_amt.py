import argparse
from pathlib import Path

import mido
import torch
from transformers import AutoModelForCausalLM

from anticipation import ops
from anticipation.sample import generate
from anticipation.convert import midi_to_events, events_to_midi
from anticipation.tokenize import extract_instruments
from anticipation.vocab import DUR_OFFSET, NOTE_OFFSET, SEPARATOR, TIME_OFFSET


def list_instruments(events):
    instr_counts = ops.get_instruments(events)
    return dict(sorted(instr_counts.items(), key=lambda x: x[0]))


def keep_only_instruments(events, allowed_instr):
    allowed_instr = set(allowed_instr)

    def drop(token_triplet):
        _, _, note = token_triplet
        # skip non-musical special/control tokens conservatively
        if note >= 30000:
            return True
        # decode instrument for normal events
        from anticipation.vocab import NOTE_OFFSET, CONTROL_OFFSET
        if note >= CONTROL_OFFSET:
            note = note - CONTROL_OFFSET
        note = note - NOTE_OFFSET
        instr = note // 128
        return instr not in allowed_instr

    return ops.delete(events, drop)


def load_model(model_name, device):
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name)
    except OSError as exc:
        if "does not appear to have a file named pytorch_model.bin" in str(exc):
            raise RuntimeError(
                f"Failed to load '{model_name}': this checkpoint may be safetensors-only. "
                "Install safetensors in this environment with: pip install safetensors"
            ) from exc
        raise RuntimeError(
            f"Failed to load model '{model_name}'. "
            "Try --model stanford-crfm/music-medium-800k (the README default)."
        ) from exc
    model = model.to(device)
    model.eval()
    return model


def sec_from_bar(bar_index_1_based, bpm, beats_per_bar=4):
    # bar 1 starts at t=0
    bar0 = bar_index_1_based - 1
    return bar0 * (60.0 / bpm) * beats_per_bar


def midi_has_tempo_event(mid_path):
    mid = mido.MidiFile(mid_path)
    for track in mid.tracks:
        for msg in track:
            if msg.type == "set_tempo":
                return True
    return False


def rescale_event_timing(events, scale):
    if abs(scale - 1.0) < 1e-9:
        return events

    scaled = []
    for time_tok, dur_tok, note_tok in zip(events[0::3], events[1::3], events[2::3]):
        if note_tok == SEPARATOR:
            scaled.extend([time_tok, dur_tok, note_tok])
            continue

        raw_time = time_tok - TIME_OFFSET
        raw_dur = dur_tok - DUR_OFFSET
        new_time = TIME_OFFSET + int(round(raw_time * scale))
        new_dur = DUR_OFFSET + max(0, int(round(raw_dur * scale)))

        scaled.extend([new_time, new_dur, note_tok])

    return scaled


def max_note_end_time(events):
    max_end = 0.0
    for time_tok, dur_tok, note_tok in zip(events[0::3], events[1::3], events[2::3]):
        if note_tok == SEPARATOR:
            continue
        onset_s = (time_tok - TIME_OFFSET) / 100.0
        dur_s = (dur_tok - DUR_OFFSET) / 100.0
        max_end = max(max_end, onset_s + dur_s)
    return max_end


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["continuation", "drum_from_controls"], required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="stanford-crfm/music-large-800k")
    parser.add_argument("--bpm", type=float, default=120.0)
    parser.add_argument("--beats-per-bar", type=int, default=4)
    parser.add_argument("--start-bar", type=int, required=True)
    parser.add_argument("--end-bar", type=int, required=True)
    parser.add_argument("--top-p", type=float, default=0.98)
    parser.add_argument("--control-instr", type=int, nargs="*", default=[])
    parser.add_argument("--drum-only", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument(
        "--source-bpm",
        type=float,
        default=None,
        help=(
            "Original BPM used by the MIDI timing when tempo metadata is missing. "
            "If omitted and no tempo event exists, defaults to 120 for auto-rescaling."
        ),
    )
    parser.add_argument(
        "--disable-auto-tempo-rescale",
        action="store_true",
        help=(
            "Disable automatic timing rescale when MIDI has no tempo event and --bpm differs "
            "from source/default 120."
        ),
    )
    parser.add_argument(
        "--allow-leading-silence",
        action="store_true",
        help=(
            "Allow start bar/time to be significantly after the last input note onset. "
            "Disabled by default to prevent accidental long silent gaps from BPM/bar mismatch."
        ),
    )
    args = parser.parse_args()

    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    print(f"[INFO] device={device}")

    model = load_model(args.model, device)

    has_tempo_event = midi_has_tempo_event(args.input)
    events = midi_to_events(args.input)

    if not has_tempo_event and not args.disable_auto_tempo_rescale:
        source_bpm = args.source_bpm if args.source_bpm is not None else 120.0
        if abs(source_bpm - args.bpm) > 1e-9:
            scale = source_bpm / args.bpm
            events = rescale_event_timing(events, scale)
            print(
                "[INFO] no tempo metadata found; auto-rescaled input timing "
                f"by factor {scale:.3f} (source_bpm={source_bpm:.3f} -> target_bpm={args.bpm:.3f})"
            )

    print(f"[INFO] loaded events={len(events)//3}")
    print(f"[INFO] instruments={list_instruments(events)}")
    input_max_time = ops.max_time(events)
    input_max_end_time = max_note_end_time(events)
    print(f"[INFO] input max onset time: {input_max_time:.3f}s")
    print(f"[INFO] input max note end time: {input_max_end_time:.3f}s")

    start_sec = sec_from_bar(args.start_bar, args.bpm, args.beats_per_bar)
    end_sec = sec_from_bar(args.end_bar + 1, args.bpm, args.beats_per_bar)
    print(f"[INFO] generation window: {start_sec:.3f}s -> {end_sec:.3f}s")
    seconds_per_bar = (60.0 / args.bpm) * args.beats_per_bar

    # Safety check: if start time is far after the input's last note onset,
    # generation can produce long leading silence that usually indicates wrong BPM/bar mapping.
    reference_time = max(input_max_time, input_max_end_time)
    if start_sec > reference_time + 0.5 * seconds_per_bar and not args.allow_leading_silence:
        gap = start_sec - reference_time
        raise ValueError(
            "Start time is significantly after the end of the input musical content. "
            f"Requested start={start_sec:.3f}s, input_end={reference_time:.3f}s, gap={gap:.3f}s. "
            "This usually means BPM or bar indexing is mismatched for this MIDI. "
            "Double-check --bpm/--start-bar (or pass --allow-leading-silence if intentional)."
        )

    if args.mode == "continuation":
        # only keep prompt up to start time for true continuation
        prompt_events = ops.clip(events, 0, start_sec, clip_duration=False)
        generated = generate(
            model,
            start_time=start_sec,
            end_time=end_sec,
            inputs=prompt_events,
            controls=None,
            top_p=args.top_p,
        )
        out_events = generated

    elif args.mode == "drum_from_controls":
        if not args.control_instr:
            raise ValueError("For drum_from_controls, pass --control-instr (e.g. 0 33).")
        # controls = specified instruments (e.g., piano+bass)
        non_controls, controls = extract_instruments(events, args.control_instr)

        # history up to start time from non-controls
        history = ops.clip(non_controls, 0, start_sec, clip_duration=False)

        generated = generate(
            model,
            start_time=start_sec,
            end_time=end_sec,
            inputs=history,
            controls=controls,
            top_p=args.top_p,
        )

        if args.drum_only:
            generated = keep_only_instruments(generated, [128])

        out_events = ops.combine(generated, controls)

    out_mid = events_to_midi(out_events)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out_mid.save(args.output)
    print(f"[OK] saved {args.output}")


if __name__ == "__main__":
    main()