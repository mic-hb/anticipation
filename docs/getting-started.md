# Getting Started with Anticipatory Music Transformer (AMT)

This guide is a complete, beginner-friendly walkthrough for running AMT generation/inference with this repository.

It covers:

- What AMT does and how it differs from plain continuation models
- Full environment setup
- A reusable inference script
- Three practical scenarios:
  - 1 piano track, 4 bars -> continue to bars 5-8
  - 2 tracks (piano+bass) -> generate drums conditioned on both
  - 5-track verse -> generate a chorus continuation
- Common errors, debugging, and quality tips

---

## 1) What AMT Is (Simple Explanation)

AMT (Anticipatory Music Transformer) is a symbolic music model that generates MIDI-like events. It can work in two important ways:

1. **Continuation (prompted generation)**  
   Give the model a beginning section (prompt), and it continues the piece forward in time.

2. **Controlled generation (infilling/accompaniment-like behavior)**  
   Give the model one or more fixed tracks/events as controls (for example melody, piano, bass), and ask it to generate additional material that matches those controls.

In the AMT paper, this control behavior is done through an _anticipation_ mechanism: control events are interleaved so the model can account for near-future controls while generating current events.

---

## 2) Repository APIs You Will Use

From this repository, the key inference APIs are:

- `anticipation.convert.midi_to_events(mid_path)`  
  Convert MIDI file -> AMT event tokens.

- `anticipation.sample.generate(model, start_time, end_time, inputs=None, controls=None, top_p=...)`  
  Generate events for the target time window.

- `anticipation.convert.events_to_midi(events)`  
  Convert AMT event tokens -> MIDI.

- `anticipation.tokenize.extract_instruments(events, instruments)`  
  Split events into:
  - controls: selected instrument IDs
  - events: everything else

- `anticipation.ops.combine(events, controls)`  
  Merge generated events and controls back to one event stream.

---

## 3) Prerequisites

- Python 3.9+ recommended
- (Optional but recommended) CUDA GPU
- Internet access to download Hugging Face model checkpoint

Recommended model for this guide:

- `stanford-crfm/music-large-800k`

Important compatibility note:

- `stanford-crfm/music-large-800k` ships as `model.safetensors` (not `pytorch_model.bin`).
- You must install `safetensors` in the same virtual environment.
- `stanford-crfm/music-medium-800k` uses `pytorch_model.bin`, so it may load even without `safetensors`.

---

## 4) Environment Setup (Step-by-step)

Run these commands from the repository submodule:

```bash
cd /home/michb/dev/01-ISTTS/auto-midi/lib/anticipation
python -m venv .venv
source .venv/bin/activate
pip install .
pip install -r requirements.txt
pip install transformers huggingface_hub
pip install safetensors
```

### Typical output (example)

```text
Processing /home/michb/dev/01-ISTTS/auto-midi/lib/anticipation
Preparing metadata (setup.py) ... done
Installing collected packages: anticipation
Running setup.py install for anticipation ... done
Successfully installed anticipation-1.0
...
Collecting torch>=2.0.1
Downloading torch-...whl
Successfully installed ...
```

Notes:

- The deprecation warning about `setup.py install` can appear and does not block usage.
- If installation is very slow, it is usually `torch` download size.
- `safetensors` is required for loading `stanford-crfm/music-large-800k`.

---

## 5) Time, Bars, and BPM (Very Important)

AMT generation API uses **seconds**, not bars.

You usually think in bars (music structure), so convert bars to seconds:

```text
seconds_per_bar = (60 / BPM) * beats_per_bar
```

Example (4/4, 120 BPM):

- 1 beat = 0.5 s
- 1 bar = 4 \* 0.5 = 2 s
- 4 bars = 8 s
- 8 bars = 16 s

So:

- bar 1 starts at 0 s
- bar 5 starts at 8 s
- bar 9 starts at 16 s

---

## 6) Create a Reusable Inference Script

Create file: `lib/anticipation/infer_amt.py`

```python
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
            "Try --model stanford-crfm/music-medium-800k as fallback."
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
```

---

## 7) Scenario A: Piano 4 Bars -> Continue to Bars 5-8

### Goal

Input MIDI has one piano track and 4 bars.  
You want to generate bars 5-8.

### Command

```bash
cd /home/michb/dev/01-ISTTS/auto-midi/lib/anticipation
source .venv/bin/activate

python infer_amt.py \
  --mode continuation \
  --input data/inference/continuation/piano-4bars.mid \
  --output data/inference/out/piano-8bars.mid \
  --start-bar 5 \
  --end-bar 8 \
  --bpm 120 \
  --beats-per-bar 4 \
  --top-p 0.98
```

### Example output

```text
[INFO] device=cuda
[INFO] loaded events=412
[INFO] instruments={0: 412}
[INFO] generation window: 8.000s -> 16.000s
100%|████████████████████████████████| 800/800 [00:06<00:00, 129.30it/s]
[OK] saved data/inference/out/piano-8bars.mid
```

### Explanation

- `--start-bar 5` means generation starts right after the first 4 bars.
- `--end-bar 8` means generation includes bars 5, 6, 7, and 8.
- `top_p=0.98` gives some variety while keeping coherence.

---

## 8) Scenario B: Piano + Bass -> Generate Drum Track

### Goal

Input MIDI has 2 tracks (piano and bass).  
You want drum track generation conditioned on both.

### Step 1: Check instrument IDs in your MIDI

```bash
python - <<'PY'
from anticipation.convert import midi_to_events
from anticipation import ops
events = midi_to_events("data/inference/your_piano_bass.mid")
print(dict(ops.get_instruments(events)))
PY
```

### Example output

```text
{0: 530, 33: 290}
```

In this example:

- `0` is piano
- `33` is bass

### Step 2: Generate drums with controls

```bash
python infer_amt.py \
  --mode drum_from_controls \
  --input data/inference/your_piano_bass.mid \
  --output data/inference/out/piano_bass_plus_drums.mid \
  --start-bar 1 \
  --end-bar 8 \
  --bpm 120 \
  --beats-per-bar 4 \
  --control-instr 0 33 \
  --drum-only \
  --top-p 0.95
```

### Example output

```text
[INFO] device=cuda
[INFO] loaded events=820
[INFO] instruments={0: 530, 33: 290}
[INFO] generation window: 0.000s -> 16.000s
100%|████████████████████████████████| 1600/1600 [00:13<00:00, 120.40it/s]
[OK] saved data/inference/out/piano_bass_plus_drums.mid
```

### Explanation

- `extract_instruments(..., [0,33])` marks piano+bass as controls.
- Model generates around those controls.
- `--drum-only` filters generated events to drum instrument ID `128`.
- `ops.combine(...)` merges generated drums and fixed controls.

---

## 9) Scenario C: 5-Track Verse (8 Bars) -> Continue as Chorus

### Goal

Input MIDI is full-band verse (8 bars): for example piano melody, strings, piano harmony, bass, drums.  
You want to continue into chorus.

### Command (example: generate bars 9-16)

```bash
python infer_amt.py \
  --mode continuation \
  --input data/inference/fullband_verse_8bars.mid \
  --output data/inference/out/fullband_with_chorus.mid \
  --start-bar 9 \
  --end-bar 16 \
  --bpm 120 \
  --beats-per-bar 4 \
  --top-p 0.97
```

### Example output

```text
[INFO] device=cuda
[INFO] loaded events=1450
[INFO] instruments={0: 620, 48: 180, 1: 210, 34: 210, 128: 230}
[INFO] generation window: 16.000s -> 32.000s
100%|████████████████████████████████| 1600/1600 [00:19<00:00, 83.95it/s]
[OK] saved data/inference/out/fullband_with_chorus.mid
```

### Practical advice

- Chorus generation usually benefits from trying multiple outputs:
  - `--top-p 0.95`
  - `--top-p 0.97`
  - `--top-p 0.99`
- Keep 3-5 variants and select by listening.

---

## 10) Optional: Quick Baseline Test from Python REPL

If you want a very short sanity check:

```bash
python - <<'PY'
from transformers import AutoModelForCausalLM
from anticipation.sample import generate
from anticipation.convert import events_to_midi

model = AutoModelForCausalLM.from_pretrained("stanford-crfm/music-large-800k").to("cuda")
events = generate(model, start_time=0, end_time=10, top_p=0.98)
mid = events_to_midi(events)
mid.save("data/inference/out/sanity_10s.mid")
print("saved data/inference/out/sanity_10s.mid")
PY
```

Expected output:

```text
100%|████████████████████████████████| 1000/1000 [00:..<?, ?it/s]
saved data/inference/out/sanity_10s.mid
```

---

## 11) Common Problems and Fixes

### A) CUDA out-of-memory

Symptoms:

- crash while loading model or during generation.

Fix:

- Run with CPU: add `--cpu` to script command.
- Reduce generation duration.

### B) Wrong bar boundaries in output

Cause:

- Wrong BPM/time signature assumption in command.

Fix:

- Verify MIDI tempo/time signature in your DAW.
- Recalculate `start-bar`/`end-bar`.

### C) No/poor drum generation in Scenario B

Try:

- increase randomness: `--top-p 0.98` or `0.99`
- ensure `--control-instr` IDs are correct from `ops.get_instruments`
- generate multiple candidates and curate manually

### D) Instrument ID confusion

Always inspect first:

```bash
python - <<'PY'
from anticipation.convert import midi_to_events
from anticipation import ops
events = midi_to_events("YOUR.mid")
print(dict(ops.get_instruments(events)))
PY
```

### E) `pip uninstall anticipation-1.0` warning

Use:

```bash
pip uninstall anticipation
```

Package name is `anticipation`, not `anticipation-1.0`.

### F) `music-large-800k` fails with "no pytorch_model.bin"

Cause:

- `music-large-800k` uses `model.safetensors`.
- Your environment is missing `safetensors`.

Fix:

```bash
pip install safetensors
```

Then re-run the same `infer_amt.py` command.

---

## 12) Quality and Workflow Best Practices

1. Keep prompts clean and musically coherent.
2. Use fixed tempo sections for easiest bar-based control.
3. Generate multiple versions and pick best sections.
4. Post-edit in DAW (velocity, voicing, drum groove cleanup).
5. Save all candidates with clear naming.

Suggested naming:

- `songA_chorus_v01.mid`
- `songA_chorus_v02.mid`
- `songA_chorus_v03.mid`

---

## 13) Relationship to the Paper

From the AMT paper:

- Music is represented as a temporal point process.
- AMT introduces an anticipation mechanism for asynchronous controls.
- This allows controllable generation (including accompaniment/infilling) while preserving strong prompted generation quality.

This repository’s `generate(...)` function implements that anticipatory inference logic, while still supporting plain continuation usage patterns.

---

## 14) Fast Command Reference

### Continuation

```bash
python infer_amt.py \
  --mode continuation \
  --input INPUT.mid \
  --output OUTPUT.mid \
  --start-bar 5 \
  --end-bar 8 \
  --bpm 120 \
  --beats-per-bar 4 \
  --top-p 0.98
```

### Controlled drum generation from piano+bass

```bash
python infer_amt.py \
  --mode drum_from_controls \
  --input INPUT.mid \
  --output OUTPUT.mid \
  --start-bar 1 \
  --end-bar 8 \
  --bpm 120 \
  --beats-per-bar 4 \
  --control-instr 0 33 \
  --drum-only \
  --top-p 0.95
```

---

## 15) Verified Environment Matrix

The following environment was tested and confirmed to load and run `stanford-crfm/music-large-800k` successfully:

- Python: `3.9.25`
- PyTorch: `2.8.0+cu128`
- Transformers: `4.29.2`
- safetensors: `0.7.0`
- huggingface_hub: `0.36.2`

Use this command any time you want to quickly print your active environment versions:

```bash
python -c "import sys, torch, transformers, safetensors, huggingface_hub; print('python', sys.version.split()[0]); print('torch', torch.__version__); print('transformers', transformers.__version__); print('safetensors', safetensors.__version__); print('huggingface_hub', huggingface_hub.__version__)"
```

If your environment differs significantly and the model fails to load, first verify that:

- you are inside the correct venv (`source .venv/bin/activate`)
- `safetensors` is installed in that exact venv

---

## 16) Next Recommended Improvements

- Add batch generation option (`--n 5`) to create multiple versions automatically.
- Add automatic tempo extraction from MIDI (instead of fixed BPM input).
- Add per-track stem export (write generated drums as separate MIDI file).
- Add objective filters (note density, rhythmic consistency) before manual listening.
