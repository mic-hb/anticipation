import argparse
from pathlib import Path
from collections import Counter

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


def midi_tempo_and_timesig_info(mid_path):
    mid = mido.MidiFile(mid_path)
    tempos = []
    timesigs = []
    for track_idx, track in enumerate(mid.tracks):
        abs_ticks = 0
        for msg in track:
            abs_ticks += msg.time
            if msg.type == "set_tempo":
                tempos.append((track_idx, abs_ticks, msg.tempo, mido.tempo2bpm(msg.tempo)))
            if msg.type == "time_signature":
                timesigs.append((track_idx, abs_ticks, msg.numerator, msg.denominator))
    return tempos, timesigs


def midi_length_seconds(mid_path):
    return mido.MidiFile(mid_path).length


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


def first_note_onset_time(events):
    onset = None
    for time_tok, _, note_tok in zip(events[0::3], events[1::3], events[2::3]):
        if note_tok == SEPARATOR:
            continue
        t = (time_tok - TIME_OFFSET) / 100.0
        onset = t if onset is None else min(onset, t)
    return 0.0 if onset is None else onset


def seconds_to_bar_beat(seconds, bpm, beats_per_bar):
    beat_len = 60.0 / bpm
    total_beats = seconds / beat_len if beat_len > 0 else 0.0
    bar = int(total_beats // beats_per_bar) + 1
    beat_in_bar = (total_beats % beats_per_bar) + 1.0
    return bar, beat_in_bar


def write_tempo_metadata(mid, bpm, beats_per_bar):
    # Ensure the exported MIDI carries explicit musical timing metadata.
    # Without this, many DAWs/tools assume defaults (often 120 BPM, 4/4).
    if len(mid.tracks) == 0:
        mid.tracks.append(mido.MidiTrack())

    # IMPORTANT:
    # Event times in this codebase are quantized at TIME_RESOLUTION (10ms => 100 ticks/second).
    # To preserve absolute timing when setting BPM, we must set ticks_per_beat so that:
    #   seconds_per_tick = tempo_us_per_beat / 1e6 / ticks_per_beat = 1 / TIME_RESOLUTION
    tempo_us = mido.bpm2tempo(bpm)
    target_tpb = max(1, int(round(tempo_us / 10000.0)))  # because 1/TIME_RESOLUTION = 0.01s
    mid.ticks_per_beat = target_tpb

    # Remove existing global timing metas to avoid conflicts.
    for track in mid.tracks:
        filtered = [
            msg
            for msg in track
            if msg.type not in ("set_tempo", "time_signature")
        ]
        track.clear()
        track.extend(filtered)

    tempo_msg = mido.MetaMessage("set_tempo", tempo=tempo_us, time=0)
    ts_msg = mido.MetaMessage("time_signature", numerator=beats_per_bar, denominator=4, time=0)
    mid.tracks[0].insert(0, ts_msg)
    mid.tracks[0].insert(0, tempo_msg)


def summarize_midi_file(path, label):
    mid = mido.MidiFile(path)
    tempos, timesigs = midi_tempo_and_timesig_info(path)
    print(f"[INFO] [{label}] midi summary: tracks={len(mid.tracks)}, ticks_per_beat={mid.ticks_per_beat}, length={mid.length:.3f}s")
    if tempos:
        bpm_list = [f"{bpm:.3f}" for _, _, _, bpm in tempos]
        print(f"[INFO] [{label}] tempo events: count={len(tempos)}, bpms={bpm_list}")
    else:
        print(f"[INFO] [{label}] tempo events: none")
    if timesigs:
        sigs = [f"{num}/{den}" for _, _, num, den in timesigs]
        print(f"[INFO] [{label}] time signatures: count={len(timesigs)}, values={sigs}")
    else:
        print(f"[INFO] [{label}] time signatures: none")

    channel_counter = Counter()
    for idx, track in enumerate(mid.tracks):
        note_on = 0
        channels = set()
        programs = {}
        track_name = None
        for msg in track:
            if msg.type == "track_name":
                track_name = msg.name
            if hasattr(msg, "channel"):
                channels.add(msg.channel)
                channel_counter[msg.channel] += 1
            if msg.type == "program_change":
                programs[msg.channel] = msg.program
            if msg.type == "note_on" and msg.velocity > 0:
                note_on += 1
        print(
            f"[INFO] [{label}] track#{idx}: name={track_name!r}, note_on={note_on}, "
            f"channels={sorted(channels)}, programs={dict(sorted(programs.items()))}"
        )
    print(f"[INFO] [{label}] channels observed (all tracks): {sorted(channel_counter.keys())}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["continuation", "drum_from_controls"], required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="stanford-crfm/music-large-800k")
    parser.add_argument("--bpm", type=float, default=120.0)
    parser.add_argument("--beats-per-bar", type=int, default=4)
    parser.add_argument("--start-bar", type=int, default=None)
    parser.add_argument("--end-bar", type=int, default=None)
    parser.add_argument(
        "--start-from",
        choices=["bar", "active_end"],
        default="bar",
        help=(
            "Generation start reference. "
            "'bar' uses --start-bar as usual. "
            "'active_end' starts at the input's last note end time."
        ),
    )
    parser.add_argument(
        "--generate-bars",
        type=float,
        default=None,
        help=(
            "Number of bars to generate from the chosen start point. "
            "Useful with --start-from active_end. If set, this overrides --end-bar."
        ),
    )
    parser.add_argument("--top-p", type=float, default=0.98)
    parser.add_argument("--control-instr", type=int, nargs="*", default=[])
    parser.add_argument("--drum-only", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument(
        "--auto-tempo-rescale",
        action="store_true",
        help=(
            "When input MIDI has no tempo metadata, rescale tokenized event timing from source BPM "
            "to --bpm. Disabled by default because it may overcorrect on some human-played MIDI files."
        ),
    )
    parser.add_argument(
        "--source-bpm",
        type=float,
        default=None,
        help=(
            "Original BPM used by the MIDI timing when tempo metadata is missing. "
            "If omitted and --auto-tempo-rescale is enabled, defaults to 120."
        ),
    )
    parser.add_argument(
        "--skip-write-tempo-meta",
        action="store_true",
        help="Do not write set_tempo/time_signature metadata to output MIDI.",
    )
    parser.add_argument(
        "--align-bars-to-input-length",
        action="store_true",
        help=(
            "Map bar positions to the input MIDI timeline length instead of fixed --bpm conversion. "
            "Useful when tempo metadata is missing or unreliable."
        ),
    )
    parser.add_argument(
        "--input-bars",
        type=int,
        default=None,
        help=(
            "Total bar count represented by the input clip. Required when using "
            "--align-bars-to-input-length."
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
    parser.add_argument(
        "--snap-start-to-next-bar",
        action="store_true",
        help=(
            "When using --start-from active_end, snap generation start to the next full bar boundary "
            "at --bpm/--beats-per-bar. Helps keep continuation on DAW grid."
        ),
    )
    args = parser.parse_args()
    if args.start_from == "bar" and args.start_bar is None:
        raise ValueError("--start-bar is required when --start-from bar.")
    if args.snap_start_to_next_bar and args.start_from != "active_end":
        raise ValueError("--snap-start-to-next-bar is only valid with --start-from active_end.")
    if args.generate_bars is not None and args.generate_bars <= 0:
        raise ValueError("--generate-bars must be > 0.")
    if args.generate_bars is None and args.end_bar is None:
        raise ValueError("Provide either --end-bar or --generate-bars.")

    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    print(f"[INFO] device={device}")
    summarize_midi_file(args.input, "input")

    model = load_model(args.model, device)

    tempos, timesigs = midi_tempo_and_timesig_info(args.input)
    has_tempo_event = midi_has_tempo_event(args.input)
    events = midi_to_events(args.input)

    if not has_tempo_event:
        if args.auto_tempo_rescale:
            source_bpm = args.source_bpm if args.source_bpm is not None else 120.0
            if abs(source_bpm - args.bpm) > 1e-9:
                scale = source_bpm / args.bpm
                events = rescale_event_timing(events, scale)
                print(
                    "[INFO] no tempo metadata found; auto-rescaled input timing "
                    f"by factor {scale:.3f} (source_bpm={source_bpm:.3f} -> target_bpm={args.bpm:.3f})"
                )
        else:
            print(
                "[WARN] input MIDI has no tempo metadata. "
                "Bar-to-second mapping may not match your DAW project tempo. "
                "If boundaries feel shifted, try either: "
                "(1) set --bpm to 120 for this run, or "
                "(2) enable --auto-tempo-rescale --source-bpm <timeline bpm>."
            )

    print(f"[INFO] loaded events={len(events)//3}")
    print(f"[INFO] instruments={list_instruments(events)}")
    if tempos:
        bpms = [f"{tempo_bpm:.3f}" for _, _, _, tempo_bpm in tempos]
        print(f"[INFO] detected tempo events: count={len(tempos)}, bpms={bpms}")
    else:
        print("[INFO] detected tempo events: none")
    if timesigs:
        sigs = [f"{num}/{den}" for _, _, num, den in timesigs]
        print(f"[INFO] detected time signatures: count={len(timesigs)}, values={sigs}")
    else:
        print("[INFO] detected time signatures: none")

    first_onset = first_note_onset_time(events)
    input_max_time = ops.max_time(events)
    input_max_end_time = max_note_end_time(events)
    input_midi_length = midi_length_seconds(args.input)
    bar, beat = seconds_to_bar_beat(first_onset, args.bpm, args.beats_per_bar)
    active_end_bar, active_end_beat = seconds_to_bar_beat(input_max_end_time, args.bpm, args.beats_per_bar)
    active_span = max(0.0, input_max_end_time - first_onset)
    active_bars = active_span / ((60.0 / args.bpm) * args.beats_per_bar)
    trailing_silence = max(0.0, input_midi_length - input_max_end_time)
    print(f"[INFO] first note onset: {first_onset:.3f}s (approx bar {bar}, beat {beat:.2f} at {args.bpm} BPM)")
    print(f"[INFO] input max onset time: {input_max_time:.3f}s")
    print(f"[INFO] input max note end time: {input_max_end_time:.3f}s")
    print(f"[INFO] active musical span: {active_span:.3f}s (~{active_bars:.2f} bars at {args.bpm} BPM)")
    print(f"[INFO] input midi file length: {input_midi_length:.3f}s")
    print(
        f"[INFO] last note end position: approx bar {active_end_bar}, "
        f"beat {active_end_beat:.2f} at {args.bpm} BPM"
    )
    if trailing_silence > ((60.0 / args.bpm) * args.beats_per_bar):
        print(
            "[WARN] large trailing silence detected: "
            f"{trailing_silence:.3f}s after last note end."
        )

    if args.start_from == "active_end":
        start_sec = input_max_end_time
        default_sec_per_bar = (60.0 / args.bpm) * args.beats_per_bar
        if args.snap_start_to_next_bar:
            bar_index = int(start_sec // default_sec_per_bar)
            snapped_start = (bar_index + 1) * default_sec_per_bar
            print(
                "[INFO] snapped active_end start to next bar boundary: "
                f"{start_sec:.3f}s -> {snapped_start:.3f}s"
            )
            start_sec = snapped_start
        if args.generate_bars is not None:
            end_sec = start_sec + args.generate_bars * default_sec_per_bar
        elif args.end_bar is not None:
            end_sec = sec_from_bar(args.end_bar + 1, args.bpm, args.beats_per_bar)
            if end_sec <= start_sec:
                raise ValueError(
                    "Computed end time is not after active_end start. "
                    "Increase --end-bar or use --generate-bars."
                )
        print(
            "[INFO] start-from active_end enabled: "
            f"start_sec={start_sec:.3f}s, generate_bars={args.generate_bars}"
        )
    elif args.align_bars_to_input_length:
        if args.input_bars is None or args.input_bars <= 0:
            raise ValueError("--input-bars must be provided and > 0 when using --align-bars-to-input-length.")
        # Use the observed input timeline so bar mapping remains consistent with this file's own timing basis.
        observed_span = max(input_midi_length, input_max_end_time)
        seconds_per_bar = observed_span / float(args.input_bars)
        start_sec = (args.start_bar - 1) * seconds_per_bar
        end_sec = args.end_bar * seconds_per_bar
        print(
            "[INFO] bar mapping aligned to input length: "
            f"observed_span={observed_span:.3f}s, input_bars={args.input_bars}, "
            f"seconds_per_bar={seconds_per_bar:.3f}s"
        )
    else:
        start_sec = sec_from_bar(args.start_bar, args.bpm, args.beats_per_bar)
        if args.generate_bars is not None:
            end_sec = start_sec + args.generate_bars * ((60.0 / args.bpm) * args.beats_per_bar)
        else:
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
    if not args.skip_write_tempo_meta:
        write_tempo_metadata(out_mid, args.bpm, args.beats_per_bar)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out_mid.save(args.output)
    print(f"[OK] saved {args.output}")
    summarize_midi_file(args.output, "output")


if __name__ == "__main__":
    main()