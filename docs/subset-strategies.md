# GigaMIDI Subset Strategies for AMT Fine-Tuning

## Motivation

Fine-tuning a large-scale transformer like AMT on the full GigaMIDI dataset is infeasible given compute constraints. The full expressive corpus contains **758,197 files / 1,386,160 tracks** — an estimated ~7× the token volume of the Lakh MIDI Dataset (LMD). We need a **representative subset** (~50% LMD tokens) that preserves instrument diversity while avoiding domination by over-represented groups.

This document traces the evolution of the subset strategy — from the initial "Option B" heuristic through the exploration of principled alternatives, to the final three strategies selected for empirical comparison.

---

## 1. Background: The Dataset Imbalance Problem

### Instrument Group Distribution (Full Expressive GigaMIDI)

| Group                | Tracks        | % of Total |
| -------------------- | ------------- | ---------- |
| Drums                | 544,403       | 39.3%      |
| Piano                | 264,308       | 19.1%      |
| Guitar               | 165,820       | 12.0%      |
| Ensemble             | 98,417        | 7.1%       |
| Bass                 | 52,035        | 3.8%       |
| Brass                | 48,939        | 3.5%       |
| Reed                 | 42,102        | 3.0%       |
| Strings              | 27,156        | 2.0%       |
| Pipe                 | 25,165        | 1.8%       |
| Synth Lead           | 25,847        | 1.9%       |
| Organ                | 25,410        | 1.8%       |
| Chromatic Percussion | 18,018        | 1.3%       |
| Synth Pad            | 17,724        | 1.3%       |
| Synth Effects        | 9,170         | 0.7%       |
| Percussive           | 8,751         | 0.6%       |
| Sound Effects        | 7,732         | 0.6%       |
| Ethnic               | 5,163         | 0.4%       |
| **Total**            | **1,386,160** | **100%**   |

Three instrument groups — **Drums (39.3%), Piano (19.1%), and Guitar (12.0%)** — together account for **70.3%** of all expressive tracks. At the other extreme, **Ethnic** has only 5,163 tracks (0.4%), and **Sound Effects** has 7,732 (0.6%).

A naively proportional subset would reproduce this imbalance: the model would see vastly more piano/drums/guitar examples than ethnic or sound effects, likely biasing its generations toward those instrumentations and underperforming on minority groups.

---

## 2. Option B: The Original Heuristic (Big3-Equalized)

The first attempt at a balanced subset — **Option B** — emerged from the initial `analyze_dataset.py` analysis. Its core idea:

- **Big3** (Drums, Piano, Guitar): assign each an **equal fixed cap** of tracks, replacing the proportional allocation
- **Other groups**: sample at a **uniform percentage** of their available tracks

### Initial Parameters (Pre-Calibration)

- BIG3_CAP = 44,444 tracks each
- OTHER_PCT = 13.68% of each minor group's available tracks

These initial parameters were chosen to match the proportional Option A's total volume at roughly 50% LMD. However, the greedy selection algorithm — which picks largest-files-first within each group's cap — overshot badly:

> **First pass: 127% of LMD tokens** (844M vs. 332M target)

### Calibration by Binary Search

A dedicated `calibrate_subset_size.py` script was introduced. It binary-searches a single **scale factor** applied to both BIG3_CAP and OTHER_PCT simultaneously, targeting exactly 50% LMD tokens.

**Calibrated parameters (scale=0.2688):**

| Parameter | Pre-Calibration | Calibrated |
| --------- | --------------- | ---------- |
| BIG3_CAP  | 44,444          | 11,944     |
| OTHER_PCT | 13.68%          | 3.68%      |

**Calibrated Option B profile (50% LMD):**

| Metric             | Value          |
| ------------------ | -------------- |
| Scale factor       | 1.0068         |
| Files              | 22,843         |
| Expressive tracks  | 51,314         |
| Estimated tokens   | ~332M          |
| Big3 total         | 36,075 (70.3%) |
| Other total        | 15,239 (29.7%) |
| Big3 each (capped) | 12,025         |

### Limitation

While Option B prevents Big3 dominance, it still preserves the **original proportion among the 14 non-Big3 groups** (each at 3.68% of its availability). The tiniest groups remain tiny: **Ethnic gets only 189 tracks** (3.7% of 5,163). This might be insufficient for the model to learn meaningful representations of these instrumentations.

---

## 3. Systematic Strategy Exploration

Before committing to Option B, we ran a broad exploration of alternative balancing approaches in `explore_balanced_strategies.py`.

### Strategies Considered

| ID  | Name                 | Logic                                                        |
| --- | -------------------- | ------------------------------------------------------------ |
| B   | Big3-equalized       | Cap Big3 equally, proportional % for others                  |
| C1  | Uniform tracks       | All 15 groups receive the same track count                   |
| C2  | Per-group %          | Each group gets exactly N% of its available tracks           |
| C3  | Sqrt-weighted        | Target proportion ~ sqrt(current proportion)                 |
| C4  | Log-weighted         | Target proportion ~ log(1 + current %)                       |
| C5  | Compressed range     | Squeeze toward uniform: uniform + (current - uniform) × 0.25 |
| C6  | Floor + proportional | Each group gets a minimum floor, rest proportional           |

### Insights from the Exploration

- **C3 (sqrt-weighted)** and **C4 (log-weighted)** both compress the range but in different ways — sqrt heavily boosts minorities, log moderately boosts them
- **C5 (25% compressed)** is a tunable middle ground: it pulls each group 75% of the way toward uniform distribution
- **C6 (floor-based)** guarantees minimum representation but is sensitive to the floor choice
- **C1 (uniform)** is the most aggressive rebalancer — the bottleneck is Ethnic at 5,163 tracks, so uniform caps are limited by Ethnic availability
- **C2 (per-group %)** is the simplest to understand and implement: `target[g] = avail[g] × pct / 100`

### Selection of Final Three

We selected **three strategies covering a spectrum of rebalancing intensity** for empirical comparison:

| Strategy               | Rebalancing | Philosophy                                                                                             |
| ---------------------- | ----------- | ------------------------------------------------------------------------------------------------------ |
| **B** (Big3-equalized) | Mild        | Correct the single biggest imbalance (Big3 vs. rest), preserve natural distribution among minor groups |
| **C1** (Uniform)       | Strong      | Every instrument group gets equal representation regardless of real-world prevalence                   |
| **C2** (Per-group %)   | Moderate    | Preserve natural distribution ratios exactly, just shrink proportionally                               |

---

## 4. Final Three Strategies — Detailed Specification

All three are calibrated to ~50% LMD tokens (~332M tokens) via binary search over a scale factor applied to each strategy's base targets.

### Strategy B: Big3-Equalized (Mild Rebalance)

**Base targets (before calibration):**

| Group             | Base Target    | Logic                            |
| ----------------- | -------------- | -------------------------------- |
| Drums             | 11,944         | Fixed cap (equal among Big3)     |
| Piano             | 11,944         | Fixed cap (equal among Big3)     |
| Guitar            | 11,944         | Fixed cap (equal among Big3)     |
| Other (14 groups) | 3.68% of avail | Uniform % of each group's tracks |

**After calibration (scale=1.0068):**
- 22,843 files, 51,314 tracks, ~332M tokens
- Big3: 12,025 each (equalized at 23.4% of total each)
- Other groups: retain their natural distribution at 3.7% sampling rate
- Smallest group (Ethnic): ~189 tracks before calibration, ~191 after

**Selection algorithm:**
1. Scan all files containing non-Big3 groups (="other files"), sorted by total_notes descending
2. Greedily assign files to their groups' caps
3. Big3 tracks found in "other files" count toward Big3 caps
4. After other-file pass, fill remaining Big3 capacity from Big3-only files (piano+guitar, guitar-only, piano-only, drums-only) in that priority order

### Strategy C1: Uniform Tracks (Strong Rebalance)

**Base targets (before calibration):**

| Group         | Base Target | Logic                                       |
| ------------- | ----------- | ------------------------------------------- |
| All 15 groups | 3,098       | Cap = 60% of Ethnic available (5,163 × 0.6) |

Every group is capped at the same number of tracks, determined by the **bottleneck group** (Ethnic, the smallest at 5,163 tracks). With `--pct 60`, the cap is 3,098 tracks per group — but groups with fewer than 3,098 available tracks are capped at their full availability (minor adjustment for Sound Effects, Percussive, etc.).

**After calibration (scale=1.1-1.3 range, exact depends on pct):**
- Dramatically different distribution from the full dataset:
  - Large groups (Drums, Piano, Guitar) shrink from 70.3% to ~7% each
  - Small groups (Ethnic, Sound Effects) rise from 0.4% to ~7% each
- Max-to-min ratio drops from ~105× (544,403 / 5,163) to ~2-3×

**Effect on model training:** The model sees equal exposure to every instrument group. Risk: over-representation of minority instrument patterns that appear in few unique files.

### Strategy C2: Proportional % (Moderate Rebalance)

**Base targets (before calibration):**

Each group gets exactly `pct%` of its available tracks. E.g., with `--pct 5`:

| Group         | Available | Target (5%) |
| ------------- | --------- | ----------- |
| Drums         | 544,403   | 27,220      |
| Piano         | 264,308   | 13,215      |
| Guitar        | 165,820   | 8,291       |
| Ensemble      | 98,417    | 4,921       |
| Ethnic        | 5,163     | 258         |
| Sound Effects | 7,732     | 387         |

**After calibration (scale=0.7041):**
- 26,984 files, 48,785 tracks, ~332M tokens
- Big3: 34,306 (70.3%) — same proportion as the full dataset
- Ethnic: ~181 tracks (0.37%) — same proportion as the full dataset

**Key property:** C2 **preserves the exact natural distribution** of instrument groups. It is the most conservative strategy — it only shrinks the dataset size, not the balance. Whatever biases exist in GigaMIDI are faithfully reproduced in the subset.

---

## 5. Selection Algorithm: Greedy Largest-Files-First

All three strategies use the same selection engine (in `analyze_subset.py`'s `evaluate()`):

1. **Target computation**: per-group track caps from the strategy's formula
2. **Other-file pass**: iterate all files containing any non-Big3 group, sorted by total_notes descending. For each file, check if adding its tracks would exceed any group's cap. If not, select it and update running counts. Big3 tracks found in these files count toward Big3 caps.
3. **Big3-only pass**: iterate Big3-only files (piano+guitar, guitar-only, piano-only, drums-only — in that priority order), sorted by total_notes descending. Assign remaining Big3 capacity.
4. **Calibration loop**: binary search a uniform scale factor over all base targets until the total estimated tokens hit ±0.5% of the target.

The **largest-files-first** ordering is deliberate: it maximizes the notes-per-file ratio, minimizing the number of files needed to reach the token target. This reduces total data-loading overhead during training.

---

## 6. Cross-Strategy Comparison

| Metric        | Full Expressive | Strategy B       | Strategy C1 (60%)    | Strategy C2 (5%)  |
| ------------- | --------------- | ---------------- | -------------------- | ----------------- |
| Files         | 758,197         | 22,843           | ~26,000              | 26,984            |
| Tracks        | 1,386,160       | 51,314           | ~39,000              | 48,785            |
| Est. tokens   | ~2.3B           | ~332M            | ~332M                | ~332M             |
| Big3 %        | 70.3%           | 70.3%            | ~21%                 | 70.3%             |
| Drums %       | 39.3%           | 23.4%            | ~7%                  | 39.3%             |
| Ethnic %      | 0.4%            | 0.4%             | ~7%                  | 0.4%              |
| Max/min ratio | 105×            | 63×              | ~3×                  | 105×              |
| Rebalancing   | None            | Mild (Big3 only) | Aggressive (uniform) | None (pure scale) |

---

## 7. Scripts Reference

| Script                               | Purpose                                                  |
| ------------------------------------ | -------------------------------------------------------- |
| `analyze_dataset.py`                 | Original overview (superseded)                           |
| `analyze_subset.py`                  | **Current**: analysis + calibration for all 3 strategies |
| `explore_balanced_strategies.py`     | Broad strategy exploration (B, C1-C6)                    |
| `gigamidi_create_balanced_subset.py` | **Current**: creation for all 3 strategies               |
| `gigamidi_create_balanced_lmd.py`    | Original Option-B creation (superseded)                  |
| `calibrate_subset_size.py`           | Original Option-B calibration (superseded)               |
| `analyze_option_b_subset.py`         | Original Option-B analysis (superseded)                  |

### Workflow

```bash
# 1. Explore targets (quick, no calibration)
uv run python scripts/analyze_subset.py --strategy C1 --pct 60 --dry-run

# 2. Full analysis + calibration
uv run python scripts/analyze_subset.py --strategy B

# 3. Create the subset (local copy mode, fastest)
uv run python scripts/gigamidi_create_balanced_subset.py \
  --strategy C2 --pct 5 \
  --output data/gigamidi_balanced_C2/ \
  --local_path /path/to/Final_GigaMIDI_V2.0_Final/
```

---

## 8. Next Steps: Empirical Comparison

The three strategies now represent a controlled experiment:

- **B** asks: does equalizing the top-3 groups improve AMT's performance on non-Big3 instruments, compared to the natural distribution?
- **C1** asks: does uniform instrument representation help the model handle rare instrumentations?
- **C2** asks: is the natural distribution already fine, and simply shrinking the dataset is sufficient?

Each strategy should be taken through the full fine-tuning pipeline:

```
create subset → preprocess → tokenize → define splits → shuffle → fine-tune AMT → evaluate on held-out test set
```

The held-out test set should preserve instrument-group balance to enable per-group evaluation. Key questions:
- Which strategy yields the lowest overall loss?
- Which strategy yields the best per-group performance on minority instruments?
- Is there a trade-off between major-group and minor-group performance?
