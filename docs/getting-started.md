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

## 14) Deep Dive: Context Size and Context Time in AMT

This section answers a very important practical question:

- "How many bars can AMT use as context for continuation?"

The short answer is:

- AMT has a **fixed token context window**, not a fixed bar window.
- So the number of bars depends on event density (how many note events exist per bar).
- Sparse melody can fit many bars; dense full-band can fit fewer bars.

The detailed answer is below.

### 14.1 Context in AMT is token-based, not bar-based

In this repository, AMT operates on event triplets:

- time token
- duration token
- note token

So one musical event consumes 3 tokens.

From `anticipation/config.py`:

- `CONTEXT_SIZE = 1024`
- `EVENT_SIZE = 3`
- `M = 341` where `1024 = 1 + 3 * 341`

Interpretation:

- The model sees up to 1024 tokens per forward pass.
- One token is used for mode (`AUTOREGRESS` or `ANTICIPATE`).
- The remaining 1023 tokens are event/control content.
- 1023 tokens / 3 = 341 triplets.

That is why you should mentally think:

- "My effective context is roughly **341 events** (triplets), not N bars."

### 14.2 Why bars vary even with same context size

A "bar" is a musical concept tied to tempo and meter.
A "context window" here is a model concept tied to token count.
These two do not map 1:1.

If your melody is simple (few notes per bar), each bar uses fewer events, so many bars fit.
If your melody is dense (ornaments, arpeggios, fast runs), each bar uses more events, so fewer bars fit.

This is the key reason there is no single universal number like:

- "AMT can use exactly 16 bars of context."

That sentence is only true for a specific MIDI density and arrangement style.

### 14.3 The generation loop uses a rolling recent-history window

In sampling, AMT does not keep appending unlimited past tokens into model input forever.
It keeps a moving lookback window near context length:

- `lookback = max(len(tokens) - 1017, 0)`
- `history = history[lookback:]`

This means:

- As generation advances, old events fall out of active model input.
- The model conditions on the **most recent** chunk of musical history.
- Long-range recall beyond that window is indirect (through generated continuation), not direct full attention.

Why this design exists:

- Transformer inference cost grows with context length.
- Fixed-size rolling context keeps inference tractable and stable.
- This is standard behavior for causal generation pipelines.

### 14.4 Formula to estimate bars of usable context

Let:

- `E` = average number of event triplets per bar in your prompt segment
- `C` = usable triplets in context (use ~339 to ~341 as practical number)

Then:

- `estimated_bars_in_context = C / E`

Rule-of-thumb table:

- sparse monophonic melody (`E ≈ 4`) -> about `339/4 ≈ 84` bars
- moderate melody (`E ≈ 8`) -> about `339/8 ≈ 42` bars
- dense melodic line (`E ≈ 16`) -> about `339/16 ≈ 21` bars
- dense multi-track arrangement (`E ≈ 32`) -> about `339/32 ≈ 10` bars
- very dense piano/full-band (`E ≈ 48`) -> about `339/48 ≈ 7` bars

These are estimation ranges, not hard guarantees.

### 14.5 Why your prompt bars can be lower than expected

Even if your melody seems short, context can be consumed by:

- accompaniment tracks (if included in prompt events)
- controls (in anticipatory mode)
- rest/padding behavior around clipping boundaries
- high note density at same timestamps (e.g. chords across tracks)

So "7 bars" in one MIDI might behave like "20 bars" in another in token usage terms.

### 14.6 Important distinction: context time vs generation time

Two separate ideas often get mixed:

1. **Context time**
   - How far back the model can directly attend during next-token prediction.
   - Limited by rolling token window (~1024 tokens total, ~341 triplets).

2. **Generation time**
   - How long the generated output can be.
   - Controlled by your `end_time` / `--end-bar`.
   - Not the same thing as context window size.

You can generate long outputs, but at each step the model still only "sees" a recent token window.

### 14.7 How anticipation (`delta`) affects practical context usefulness

AMT can run in anticipatory mode where controls are interleaved ahead of local event time.
Default `DELTA` is 5 seconds.

Practical implication:

- Some context budget is shared between event history and anticipated controls.
- With heavy controls, fewer pure event-history triplets fit.
- In exchange, generation follows conditioning constraints better.

This is a trade-off:

- more control fidelity vs. less raw historical event span in window.

### 14.8 How to measure context for your exact MIDI (recommended)

If you want accurate numbers (instead of estimates), compute event density from your own input:

1. Convert MIDI to events
2. Clip to prompt bars/time
3. Count events
4. Divide by bars
5. Compute `339 / events_per_bar`

Example helper command:

```bash
python - <<'PY'
from anticipation.convert import midi_to_events
from anticipation import ops

mid = "data/inference/continuation/02-10000-reasons-7bars-input.mid"
bpm = 80
beats_per_bar = 4
bars = 7
end_sec = bars * (60.0 / bpm) * beats_per_bar

events = midi_to_events(mid)
clip = ops.clip(events, 0, end_sec, clip_duration=False)
n_events = len(clip) // 3
events_per_bar = n_events / bars if bars else 0
approx_context_bars = (339 / events_per_bar) if events_per_bar else float("inf")

print("events:", n_events)
print("events_per_bar:", round(events_per_bar, 2))
print("approx_context_bars:", round(approx_context_bars, 2))
PY
```

How to interpret:

- If `approx_context_bars` is much larger than your prompt bars, AMT can likely see all prompt bars directly.
- If it is near or below your prompt bars, oldest prompt bars may be truncated from active context during generation.

### 14.9 Practical recommendations for melody continuation

If your goal is "preserve melodic style and phrase memory":

- Keep prompt focused (only tracks that matter for style memory).
- Avoid unnecessary dense accompaniment in prompt when not needed.
- Continue in chunks (e.g., 4-8 bars at a time), curate, then extend.
- For highly dense inputs, expect less direct long-range memory.

If your goal is "follow controls tightly":

- Use explicit controls via `extract_instruments(...)`.
- Accept that some context budget shifts from past events to controls.

### 14.10 Final answer to "How many bars can AMT use as context?"

Best accurate statement:

- AMT uses about **341 event triplets** of active context.
- Converted to bars, context bars = `341 / (events per bar)`.
- For simple melody-only prompts, this can be dozens of bars.
- For dense multi-track arrangements, it can drop to single-digit or low-teens bars.

So always estimate from your own MIDI density rather than relying on one fixed bar number.

---

## 15) Deep Dive: Output Behavior, Generation Length, and Practical Max Bars

This section answers the second major question:

- "What is the maximum number of bars that AMT can generate?"

The short answer:

- In the current AMT sampler, there is no explicit hard-coded "max output tokens = N" like many LLM APIs expose.
- Generation continues until your requested `end_time` is reached.
- So "max bars" is mostly a practical limit (time, compute, quality drift), not a single strict software ceiling in this script.

Now let's unpack this in detail.

### 15.1 How generation actually stops

In `anticipation.sample.generate(...)`, the stopping behavior is time-driven:

- `start_time` and `end_time` are converted from seconds to internal ticks.
- The loop keeps sampling event triplets.
- It exits when newly sampled event time reaches or exceeds `end_time`.

Conceptually:

1. You request an interval: generate from `start_time` to `end_time`.
2. Model proposes next event triplet `(time, dur, note)`.
3. If `new_time >= end_time`, stop.
4. Otherwise append and continue.

This is fundamentally different from "stop after K tokens" APIs. Here, stop criterion is **musical time**, not token count.

### 15.2 Why this feels different from LLM max output tokens

Many LLM interfaces enforce a fixed generation cap such as:

- `max_new_tokens=512`

AMT generation in this repo does not expose that style cap in the top-level API. Instead:

- output duration is controlled by `end_time`
- note/event density determines how many tokens are produced inside that duration

So two runs with same bars can produce different token counts:

- dense rhythmic output -> more events/tokens
- sparse output -> fewer events/tokens

That variability is expected and musically natural.

### 15.3 Is there any hidden upper bound?

There are three kinds of bounds to understand:

1. **Per-step model context bound** (fixed)
   - The model only attends to a recent rolling context window.
   - This affects memory depth, not total generated length.

2. **Time vocabulary/window mechanics** (implementation detail)
   - The tokenizer has quantized time/duration vocab ranges.
   - The sampler keeps recent history relativized in a rolling window.
   - This allows long generation in chunks of local time context.

3. **Practical runtime/system bound** (real-world)
   - GPU/CPU time
   - memory pressure
   - user patience
   - quality degradation over long unbroken free generation

So: no obvious "you can never exceed X bars" constant in your inference wrapper, but there are practical ceilings for useful quality.

### 15.4 The true practical "max bars" question

A better framing is:

- "How many bars can AMT generate before quality or structure becomes unacceptable for my use case?"

Because musically, you can often generate long continuations technically, but:

- thematic drift grows
- harmonic consistency may weaken
- groove identity may fluctuate
- section-level form (verse/chorus logic) is not guaranteed unless you provide strong conditioning

That is why experienced workflows do staged generation instead of one giant run.

### 15.5 Bar-length conversion and what it implies

Your script uses bar -> seconds conversion:

- `sec_from_bar(bar_index, bpm, beats_per_bar)`

This means the generated duration in seconds is:

- `(end_bar - start_bar + 1) * (60 / bpm) * beats_per_bar`

Example at 80 BPM, 4/4:

- 1 bar = 3 sec
- 8 bars = 24 sec
- 32 bars = 96 sec

Implication:

- slow BPM dramatically increases wall-clock generation time for same bar count
- fast BPM shortens seconds for same bar count

So "max bars" and "max seconds" are not interchangeable.

### 15.6 Runtime behavior as output length increases

As requested duration increases:

- number of sampled events grows roughly with musical density
- generation latency increases accordingly
- the active context remains local/rolling (old context falls out)

This means:

- long generation is feasible
- but it is not "global-memory long generation"
- it is "iterative local continuation"

This distinction explains why very long single-pass generations may gradually drift stylistically.

### 15.7 Why long-form quality drifts (the why behind the behavior)

Quality drift is not a bug; it is a consequence of autoregressive local continuation:

1. model only sees recent window directly
2. each new event depends on prior generated events
3. small deviations accumulate over time
4. without explicit high-level controls, form can wander

This is analogous to language models writing very long passages:

- coherence is high locally
- global structure requires scaffolding/constraints

AMT gives you tools for scaffolding via controls; use them for long-form outputs.

### 15.8 Recommended strategy for "long output" generation

Instead of generating very long segments in one pass, use iterative staged generation:

1. Generate 4-8 bars.
2. Listen/select best candidate.
3. Use selected output as next prompt context.
4. Repeat.

Benefits:

- tighter quality control
- easier correction of drift
- better section design (verse -> pre-chorus -> chorus)
- lower risk of wasting long compute runs

For production composition pipelines, this is usually better than a monolithic 64-bar pass.

### 15.9 Conditioning to extend useful output horizon

If you need longer coherent output, add structure via controls:

- keep anchor melody/harmony controls
- constrain instrumentation
- generate per-section (with known section boundary bars)
- recondition each new section using selected stems/events

Effect:

- less unconstrained drift
- stronger continuity across longer total form

### 15.10 Edge cases and failure modes in long generation

Watch for:

- repetitive loops (model gets stuck in local motif)
- harmonic flattening (less functional progression over long spans)
- rhythmic over-regularization or instability
- instrument crowding in dense runs

Mitigations:

- adjust `top_p` down for stability (`0.95-0.97`)
- regenerate alternative takes
- generate shorter chunks
- enforce controls on key tracks

### 15.11 Practical guidance: choosing target bars by objective

If your objective is:

- quick idea extension -> 2-4 bars per pass
- phrase-level continuation -> 4-8 bars per pass
- section drafting -> 8-16 bars per pass (with curation)
- full-song roughing -> multi-pass section-by-section, not one-shot

These are not hard limits, but reliability-oriented operating ranges.

### 15.12 Final answer to "maximum number of bars"

Most accurate statement:

- AMT in this repo does not enforce a simple fixed "max bars" output cap at the user API level.
- Generation stops when your requested end time is reached.
- Therefore, maximum bars are primarily constrained by practical runtime and musical quality retention, not a single hard constant in `infer_amt.py`.

For best real-world results, treat AMT as a strong section-level generator and compose long forms through staged, controlled continuation.

---

## 16) Deep Dive: Conditioning, Controls, and Tuning Parameters

This section answers the third major question:

- "What are the various ways to condition AMT generation?"
- "What parameters and controls are available to tweak?"

If you are new to this area, think of AMT like this:

- AMT is a sequence generator.
- It predicts the next musical event repeatedly.
- What it predicts depends on what you feed it as context and controls.

So the core question is not only "what model checkpoint are you using?"  
It is also:

- what events do you provide as past history?
- what events do you hold fixed as controls?
- over what time window do you ask it to generate?
- how much randomness do you allow?

### 16.1 Two big conditioning channels: `inputs` and `controls`

In this codebase, the main sampler function is:

- `generate(model, start_time, end_time, inputs=None, controls=None, top_p=..., delta=...)`

Conceptually:

1. `inputs` = the baseline musical material (prompt/history)
2. `controls` = explicit constraints AMT should respect (melody/track anchors/etc.)

These are not the same thing.

- `inputs` tells AMT "what happened so far."
- `controls` tells AMT "what must be considered/anticipated around target times."

Why this matters:

- If you only use `inputs`, AMT does continuation.
- If you also use `controls`, AMT does constrained/conditioned generation.

### 16.2 How `inputs` are interpreted internally

Inside `generate(...)`, `inputs` is split by time boundary:

- events up to `start_time` become prompt history
- events after `start_time` are treated as future constraints (converted to controls)

This is subtle and very important.

Practical effect:

- If you pass a full MIDI as `inputs` and set `start_time` in the middle, AMT uses the earlier part as context and can treat later part as conditioning signal.
- If you want clean continuation, clip `inputs` to prompt only.

Why this design is useful:

- One input stream can encode both history and future anchors.
- This naturally supports infilling-like use cases.

### 16.3 How explicit `controls` work

`controls` is a separate stream of events marked as control tokens.
AMT interleaves these with event tokens in anticipatory order.

What this gives you:

- local future-aware constraints
- better accompaniment/infilling behavior than plain left-to-right continuation

Simple mental model:

- normal continuation = "finish this sentence from left to right"
- anticipatory conditioning = "finish this sentence, but you are allowed to peek at important future hints"

### 16.4 Instrument-level conditioning: the workhorse pattern

The most practical conditioning method in this repo is:

- `extract_instruments(events, [instrument_ids])`

It splits one MIDI into:

- controls = selected instruments
- events = everything else

Common examples:

- melody as control, generate accompaniment
- piano+bass as controls, generate drums
- keep harmonic skeleton fixed, generate ornamentation

Why this is powerful:

- It gives deterministic "anchors" while letting AMT fill creative gaps.
- It is easy to reason about musically.

### 16.5 Conditioning mode selection in your workflow

You can think in three operating modes:

1. **Unconstrained continuation**
   - use prompt in `inputs`
   - no explicit `controls`
   - most freedom, most drift risk over long spans

2. **Partially constrained generation**
   - prompt + selected controls (e.g., melody track)
   - good balance of freedom and structure

3. **Strongly constrained generation**
   - dense controls (multiple tracks, many events)
   - highest structural compliance, lowest novelty

There is no universally "best" mode; choose based on objective.

### 16.6 Time-window conditioning parameters (`start_time`, `end_time`)

These define *where* generation happens.

- `start_time` says when generation begins.
- `end_time` says when generation ends.

Why these are conditioning parameters too:

- They decide which part of input is history vs future.
- They decide the temporal scope over which controls can influence output.

Music-structure impact:

- wrong boundary placement can make the model condition on the wrong section.
- accurate bar-to-second conversion is crucial for stable results.

### 16.7 Sampling randomness: `top_p` and what it does musically

`top_p` is nucleus sampling threshold.
It controls diversity/coherence trade-off.

Intuition:

- lower `top_p` (e.g., 0.90-0.95): safer, more conservative, less surprising
- medium `top_p` (e.g., 0.96-0.98): balanced, commonly best for many tasks
- high `top_p` (e.g., 0.99+): more surprising ideas, higher instability risk

Why:

- higher `top_p` allows more low-probability choices.
- low-probability choices can be creative or noisy.

Best practice:

- start around `0.95-0.98`
- sample multiple variants
- curate by ear

### 16.8 Anticipation span: `delta`

AMT has anticipation interval `delta` (default from config: 5 seconds).

Interpretation:

- how far ahead controls are brought into local context
- larger delta = earlier awareness of upcoming controls

Trade-off:

- larger delta can improve compliance with future controls
- but may reduce tight locality and consume contextual focus

For most users:

- keep default unless you have a concrete reason to change it and can evaluate systematically

### 16.9 Hidden-but-important conditioning constraints in sampler logic

Beyond user-facing parameters, sampler logic also constrains outputs:

- only valid token type can appear at each position in triplet (time/dur/note)
- cannot generate control tokens directly as ordinary events
- cannot generate notes in the past relative to current time
- instrument count limits are enforced during generation dynamics

Why this matters:

- model sampling is guided toward valid musical event structure
- this reduces degenerate outputs from raw unconstrained sampling

### 16.10 Conditioning budget: controls consume context capacity

Remember from context deep-dive:

- context window is finite

Controls and events both occupy that budget.

So as control density increases:

- direct history span may shrink
- control adherence may improve

This is one of AMT's core practical balancing acts:

- historical memory vs control strength

### 16.11 Practical control design patterns (copy from production habits)

Pattern A: Melody-anchored accompaniment

- controls: melody track
- generated: harmony/rhythm/backing
- use when melody identity must remain fixed

Pattern B: Rhythm-section completion

- controls: piano + bass harmonic/rhythmic guide
- generated: drums/percussion
- use for groove generation tied to harmonic motion

Pattern C: Section continuation with anchors

- controls: sparse chord tones at section boundaries
- generated: full texture
- use for verse->chorus transitions with form guidance

Pattern D: Regenerate one stem while freezing others

- controls: all stems except target stem
- generated: target stem
- use for iterative arrangement polishing

### 16.12 How to choose parameters by objective

If your priority is **stability and musical correctness**:

- tighter controls
- moderate/lower `top_p` (`0.93-0.97`)
- shorter generation chunks

If your priority is **novel ideation**:

- lighter controls
- higher `top_p` (`0.98-0.995`)
- multi-sample and curate

If your priority is **section-level coherence over length**:

- chunked generation (4-8 bars)
- re-anchor each step with controls
- avoid one huge unconstrained pass

### 16.13 Beginner-friendly "parameter map" for `infer_amt.py`

In your current wrapper script:

- `--mode`
  - `continuation`: prompt-led continuation
  - `drum_from_controls`: instrument-split conditioned generation

- `--input`
  - source MIDI to tokenize and condition from

- `--output`
  - resulting MIDI path

- `--model`
  - checkpoint choice (`music-large-800k` default)

- `--bpm`, `--beats-per-bar`
  - bar->seconds conversion for time boundaries

- `--start-bar`, `--end-bar`
  - generation window in bar units (converted internally to seconds)

- `--top-p`
  - diversity/coherence control

- `--control-instr`
  - instrument IDs to treat as controls (in `drum_from_controls`)

- `--drum-only`
  - post-filter generated events to drum instrument ID

- `--cpu`
  - force CPU inference (slower, useful fallback)

### 16.14 Debugging conditioning mistakes (most common)

If output ignores intended structure, check:

1. wrong instrument IDs passed to `--control-instr`
2. wrong `start-bar`/`end-bar` mapping (tempo mismatch)
3. controls are too sparse to constrain desired behavior
4. `top_p` too high causing unstable branching
5. prompt includes unintended dense tracks that dominate context

Diagnostic workflow:

- print instrument counts first (`ops.get_instruments`)
- run one short test window (2-4 bars)
- compare 2-3 `top_p` values
- inspect control split quality before full run

### 16.15 A very practical mental model to keep

Think of AMT conditioning like a studio session with session notes:

- `inputs` = what the band already played
- `controls` = producer constraints ("keep this melody", "follow this bass line")
- `top_p` = how adventurous musicians are allowed to be
- `start/end` = exact section you are recording

If constraints are weak, performance is freer but less predictable.  
If constraints are strong, performance is tighter but less surprising.

That is exactly the quality-control trade-off you tune with AMT.

### 16.16 Final answer to "ways to condition + tweakable controls"

Most complete practical answer:

- AMT conditions on:
  - prompt/history (`inputs` up to `start_time`)
  - future anchors inferred from `inputs` beyond `start_time`
  - explicit control streams (`controls`)
  - model-selected next-token probabilities shaped by `top_p`
  - temporal scope (`start_time`, `end_time`)
  - anticipatory alignment (`delta`)
  - instrument-level control design (`extract_instruments`)

- In your wrapper, this appears as:
  - mode selection (`continuation` vs `drum_from_controls`)
  - bar/time boundaries
  - control instrument list
  - randomness setting
  - optional drum-only filtering

The strongest outcomes usually come from combining:

- clear section boundaries
- musically meaningful controls
- moderate sampling randomness
- iterative generation + curation rather than one-shot long output

---

## 17) Fast Command Reference

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

## 18) Verified Environment Matrix

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

## 19) Next Recommended Improvements

- Add batch generation option (`--n 5`) to create multiple versions automatically.
- Add automatic tempo extraction from MIDI (instead of fixed BPM input).
- Add per-track stem export (write generated drums as separate MIDI file).
- Add objective filters (note density, rhythmic consistency) before manual listening.
