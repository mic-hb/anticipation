import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from anticipation import ops
from anticipation.sample import generate
from anticipation.convert import midi_to_events, events_to_midi
from anticipation.tokenize import extract_instruments


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
    args = parser.parse_args()

    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    print(f"[INFO] device={device}")

    model = load_model(args.model, device)

    events = midi_to_events(args.input)
    print(f"[INFO] loaded events={len(events)//3}")
    print(f"[INFO] instruments={list_instruments(events)}")

    start_sec = sec_from_bar(args.start_bar, args.bpm, args.beats_per_bar)
    end_sec = sec_from_bar(args.end_bar + 1, args.bpm, args.beats_per_bar)
    print(f"[INFO] generation window: {start_sec:.3f}s -> {end_sec:.3f}s")

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