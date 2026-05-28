# HeteroFlow-DST

*Motion-aware CV pipeline for bacterial segmentation & heteroresistance detection.*

## Overview

A Python pipeline combining classical computer vision with deep learning for
analysing bacterial time-lapse microscopy.  Designed for *Mycobacterium
smegmatis* NCTC 8159 antibiotic-susceptibility testing, with an
`--organism M_bovis_BCG` profile available for slow growers.  The primary
goal is to detect **heteroresistance** — a small subpopulation of resistant
cells inside an otherwise susceptible culture — from dense optical-flow
signals, and to validate the call against population-level area-growth
shape.

The segmentation pipeline refines Omnipose masks with Gaussian smoothing,
Sobel edge detection and a continuous memory mask for temporal stability.
On top of that, `heteroresistance_detector.py` runs:

1. **Strategy C — flow-field divergence variability** (per-chamber).
   Dense Farnebäck optical flow is computed inside the segmentation mask;
   the divergence ∇·v of the flow field is measured frame by frame; the
   spatial coefficient of variation (CV) of divergence inside a fixed
   pre-drug ROI is tracked over time.  Four criteria (C1–C4) — including
   `cv_trend < 1.0` (chaotic death resolving into coherent resistant
   expansion) — decide the per-chamber verdict.
2. **`pop_score` — population-level area-curve cosine score** (across-chamber).
   Each post-drug area curve is compared (cosine similarity) against the
   mean A_REF (untreated) and mean B_RIF10 (100 % susceptible) curves.  A
   leave-one-out (LOO) ranking flags chambers whose growth shape is
   incompatible with both control modes.

The two signals are **orthogonal and complementary**: Strategy C catches an
*"active expansion"* phenotype via divergence dynamics, while `pop_score`
catches an *"area-maintained"* phenotype via curve shape.  Together they
cover all 5/5 C_HETERO chambers in our validation set with at least one
signal.

### Removed components (kept here as a deliberate trail)

Earlier prototypes are gone for empirical, not aesthetic, reasons:

* **Sparse Lucas–Kanade optical flow** — strictly redundant with dense
  Farnebäck.
* **Watershed mask refinement** — degraded Omnipose output in dense
  microchambers (advisor's recommendation).
* **Quadtree spatial-growth analysis (Strategy A)** — empirically redundant
  with Strategy C.
* **Flow-magnitude spatial CV (Strategy B)** — produced false-positive
  DETECTED verdicts on 100 %-susceptible controls (uniform cell death is
  not actually spatially uniform at pixel scale).  Divergence
  (Strategy C) is the magnitude-independent replacement.

## Quick Start

### 1. Installation

```bash
pip install numpy scipy scikit-image opencv-python matplotlib pandas
pip install pytest                    # optional, for the test suite
```

### 2. Smoke-test the install

```bash
python test_installation.py
# or
pytest test_installation.py -v
```

### 3. Configure paths

Edit `config.py`:

```python
BASE_DIR = "./data"   # Point at your data root
```

Or call `apply_organism_profile("M_bovis_BCG")` at the start of your script
if your data is slow-grower BCG at 10 min/frame.

### 4. Run the headline workflow

```bash
# Per-chamber + population-level heteroresistance verdict on the
# verdict set defined in test_heteroresistance_real_data.py
python test_heteroresistance_real_data.py
```

Outputs land in `results/heteroresistance_test/`:

* `summary_report.txt` — per-chamber verdicts (`NOT_DETECTED` /
  `UNCERTAIN` / `DETECTED`), the population-level LOO ranking and a
  paper cross-check (which chambers our Strategy C agrees with Tran
  et al. 2025 on).
* `dashboard_*.png` — per-chamber 9-panel dashboards.
* `loo_strip_*.png` — mini-movie strips for chambers flagged by
  `pop_score` (★ or ⚠).

## Project Structure

```
HeteroFlow-DST/
├── config.py                          # All universal parameters + organism profiles
├── hybrid_pipeline.py                 # Segmentation + GrowthAnalyzer
├── heteroresistance_detector.py       # Strategy C + population-level pop_score + LOO
├── analysis.py                        # Full-experiment workflow + TTD statistics
├── sample_position.py                 # Interactive single-position frame viewer
├── parameter_tuner.py                 # Gaussian/Sobel interactive tuner
├── test_heteroresistance_real_data.py # Verdict-set runner (multiprocessing)
├── test_installation.py               # Smoke tests (deps + import + tiny synthetic runs)
├── data/                              # (gitignored) your time-lapse data
└── results/                           # (gitignored) outputs land here
```

## Data Layout

The verdict-set runner expects three families of data directories under
`data/`:

```
data/
├── REF_raw_data101_110/                          # Original A_REF (untreated)
│   └── Pos101..Pos110/aphase/img_*.tiff
├── REF_masks101_110/
│   └── Pos101..Pos110/PreprocessedPhaseMasks/MASK_*.tif
├── REF_raw_data111_130_for_testing/              # Expanded A_REF pool
│   └── Pos111..Pos130/aphase/img_*.tiff
├── REF_masks111_131_for_testing/
│   └── Pos111..Pos130/PreprocessedPhaseMasks/MASK_*.tif
├── RIF10_raw_data201_210/                        # Original B_RIF10 (100% susceptible)
│   └── Pos201..Pos210/aphase/img_*.tiff
├── RIF10_masks201_210/
│   └── Pos201..Pos210/PreprocessedPhaseMasks/MASK_*.tif
├── RIF10_raw_data211_217_for_testing/            # Expanded B_RIF10 pool
│   └── Pos211..Pos217/aphase/img_*.tiff
├── RIF10_masks211_217_for_testing/
│   └── Pos211..Pos217/PreprocessedPhaseMasks/MASK_*.tif
├── TREAT_raw_data/                               # C_HETERO (1% resistant mix)
│   └── Pos201..Pos205/PreprocessedPhase/img_*.tiff
└── TREAT_masks/
    └── Pos201..Pos205/PreprocessedPhaseMasks/MASK_*.tif
```

The current `TEST_POSITIONS` (verdict set, 17 chambers) is:

| Group     | Positions               | Expected verdict          |
|-----------|-------------------------|---------------------------|
| A_REF     | Pos120–Pos125 (6)       | `NOT_DETECTED`            |
| B_RIF10   | Pos211–Pos216 (6)       | `NOT_DETECTED` / `UNCERTAIN` |
| C_HETERO  | Pos201–Pos205 (5)       | ≥1 of Strategy C / pop_score should fire |

Plus 21 reference-pool chambers (anchors for `pop_score`, do not vote).

## Usage

### Heteroresistance verdict on the real-data set

```bash
# Default: M. smegmatis profile (2 min/frame, HET_ROLLING_WINDOW=16)
python test_heteroresistance_real_data.py

# Slow-grower BCG profile (10 min/frame, HET_ROLLING_WINDOW=12)
python test_heteroresistance_real_data.py --organism M_bovis_BCG
```

Runtime ≈ 5 min with 4 workers on the 17-chamber verdict set + 21-chamber
reference pool.

### Segmentation + growth-curve API

```python
from hybrid_pipeline import HybridSegmentationPipeline, GrowthAnalyzer
from heteroresistance_detector import HeteroresistanceDetector
import config

# Segmentation refinement
seg = HybridSegmentationPipeline(gaussian_sigma=config.GAUSSIAN_SIGMA)
refined_masks, edges = seg.process_sequence(frames, omnipose_masks, use_memory=True)

# Area-based growth curve
ga = GrowthAnalyzer()
areas = ga.compute_area_growth(refined_masks)

# Per-chamber heteroresistance detection (Strategy C)
detector = HeteroresistanceDetector()
results = detector.run(refined_masks, frames, drug_start_frame=30)
# results is a HeterogeneityTimeSeries dataclass with:
#   .div_cv_time, .div_cv_trend, .div_hotspots, .mean_divergence
#   .detection_flags_C, .combined_score, .detection_combined
#   .clump_warning, .baseline_cell_area_um2, .div_persistent_hotspot_map
```

### Interactive frame viewer

```bash
python sample_position.py
# Type a position id (101..130 = REF, 201..217 = RIF10, 201..205 (TREAT) = HETERO)
# Slide / arrow-key through frames; baseline vs hybrid masks shown side by side
```

### Interactive parameter tuner

```bash
python parameter_tuner.py
# Sliders for Gaussian sigma + Sobel kernel; live preview
```

### Full-experiment analysis (TTD baselines)

```bash
python analysis.py
# Output includes both a stricter bootstrap + t-test TTD and the
# paper-faithful Tran-et-al-2025 SEM-overlap TTD on the same data:
#
#   TIME-TO-DETECTION RESULTS (with 95% Confidence Intervals):
#     Baseline (Omnipose + Area)   ...
#     Hybrid (Area-based)          ...
#   PAPER-STYLE BASELINE (Tran et al. 2025 — SEM-overlap criterion):
#     Baseline (paper SEM)         ...
#     Hybrid   (paper SEM)         ...
```

The SEM-overlap criterion lives in `analysis.detect_separation_sem_overlap()`
and reproduces the paper's "separation of normalized SEM values between
treatment and reference" definition.

## Configuration

`config.py` is the single source for all tuning knobs.  Highlights:

| Group                    | Constants                                         |
|--------------------------|---------------------------------------------------|
| Time resolution          | `INTERVAL_MINUTES`, `PIXEL_SIZE_UM`               |
| Rolling windows          | `ROLLING_WINDOW` (area), `HET_ROLLING_WINDOW` (detector) |
| Drug artefact            | `DRUG_ARTIFACT_FRAMES`                            |
| Outlier QC               | `AREA_OUTLIER_THRESHOLD = 0.05`                   |
| Strategy C — flow        | `DIV_FLOW_SMOOTH_SIGMA`, `DIV_MAP_SMOOTH_SIGMA`   |
| Strategy C — hotspots    | `DIV_HOTSPOT_SIGMA_THRESH`, `DIV_HOTSPOT_MIN_AREA_PX`, `DIV_PERSISTENCE_FRAMES` |
| Strategy C — CV gate     | `DIV_CV_THRESHOLD = 0.8`                          |
| Clumping QC              | `CLUMP_WARN_CELL_AREA_UM2 = 25.0`                 |
| Statistical              | `DIVERGENCE_ALPHA`, `DIVERGENCE_MIN_CONSECUTIVE`  |

To switch organism profile in-script:

```python
import config
config.apply_organism_profile("M_bovis_BCG")
```

## Verdict Semantics

### Per-chamber (Strategy C)

The Strategy C criteria C1–C4 each fire individually; the final verdict is:

* **`DETECTED`** — all four criteria fire, including `cv_trend < 1.0`.
* **`UNCERTAIN`** — some but not all criteria fire (e.g. heterogeneous
  death in a fully susceptible chamber).  This is the correct conservative
  call for borderline cases, not a failure mode.
* **`NOT_DETECTED`** — no criteria fire (clean susceptible kill, or
  untreated control).

### Population-level (`pop_score`)

LOO score is signed distance to the A_REF cluster centre, in normalised
cosine-similarity units:

* **★ candidate** — LOO score in an intermediate band (≈ +0.04 to +0.06).
  Growth shape "doesn't look untreated, doesn't look killed".  Flag the
  chamber for inspection.
* **⚠ outlier** — LOO score outside both A_REF and B_RIF10 clusters in an
  unexpected direction.
* No mark — sits inside its expected cluster.

## Two Heteroresistance Phenotypes

Validated on the current verdict set (17 fresh chambers, none used for
threshold tuning):

| Phenotype             | Signal source       | Caught chambers (C_HETERO Pos201–205) |
|-----------------------|---------------------|---------------------------------------|
| Active expansion      | Strategy C verdict  | Pos202, Pos205                        |
| Area-maintained       | `pop_score` ★       | Pos201, Pos203, Pos204                |

The two methods are not redundant — each catches chambers the other misses.
Together they flag all 5/5 C_HETERO chambers with at least one signal.

## Pipeline Components

### 1. `HybridSegmentationPipeline` (`hybrid_pipeline.py`)

* Gaussian blur preprocessing.
* Sobel edge detection.
* Continuous memory mask for temporal stability.

### 2. `GrowthAnalyzer` (`hybrid_pipeline.py`)

* Area-based growth curves.
* Rolling-window log-linear growth rate (via `scipy.stats.linregress`).
* Time-to-detection computation feeding `analysis.py`.

### 3. `HeteroresistanceDetector` (`heteroresistance_detector.py`)

* Single shared dense Farnebäck pass per frame pair.
* Divergence ∇·v of the flow field; spatial CV of divergence inside a
  fixed pre-drug ROI is the primary time series.
* Four criteria C1–C4 produce per-chamber `detection_flags_C`; `c4 =
  cv_trend < 1.0` (chaotic death resolving into coherent resistant
  expansion → CV drops).
* `score_heteroresistance_population()` + `loo_population_score()`
  produce the across-chamber `pop_score` ranking.
* Per-chamber `max_frames` override is supported (kept for future BCG
  experiments where the paper-correct window is 6–12 h).

## Testing

```bash
python test_installation.py
```

Test coverage (after LK / watershed / Strategy B removals):

* Dependencies (numpy, scipy, scikit-image, opencv-python, matplotlib, pandas)
* Pipeline imports (`hybrid_pipeline`, `HybridSegmentationPipeline`,
  `GrowthAnalyzer`, `config`)
* Class instantiation
* Preprocessing (Gaussian + Sobel sanity checks)
* Memory mask (reset + accumulation)
* Growth analysis (length, monotonicity, rolling-rate type)
* Configuration (path + parameter presence + types)

End-to-end validation lives in `test_heteroresistance_real_data.py`; running
it is the canonical "everything still works" check.

## Output

`results/heteroresistance_test/` (test runner):

* `summary_report.txt` — per-chamber verdict table, LOO ranking, paper
  cross-check.
* `dashboard_<Group>_<Pos>.png` — 9-panel per-chamber visualisation
  (divergence time series, persistent hotspot map, net expansion vs
  contraction, area curve, etc.).
* `loo_strip_<Pos>.png` — frame strips for `pop_score`-flagged chambers.

`results/full_hybrid_output/` and `results/sample_analysis_output/`
(`analysis.py` / `sample_position.py`):

* `baseline_area_growth.png`, `hybrid_area_growth.png`.
* `comprehensive_comparison.png` — 4-panel comparison.
* `analysis_summary.pickle` — full numeric results.

## Design Notes

A handful of decisions that aren't obvious from the code alone:

* **Paper-alignment defaults.** The drug-addition artefact window
  (`DRUG_ARTIFACT_FRAMES = 3`), 5 % outlier threshold for the smoothing
  fit, organism-specific rolling windows, clumping QC gate and the
  SEM-overlap TTD baseline are all set to match Tran et al. (2025).
* **Reference pool size matters for `pop_score`.** The cosine-similarity
  separation between A_REF and B_RIF10 cluster centres collapses with
  only a handful of anchor chambers; the current 23 + 10 reference-pool
  layout gives a clean three-tier separation.
* **`c4 = cv_trend < 1.0` is empirical, not theoretical.** A naïve
  reading expects heteroresistance to *increase* spatial CV of divergence
  over time.  On M. smeg + RIF10, the opposite is observed: B_RIF10
  (100 % susceptible) chambers keep CV growing (chaotic asynchronous
  death never resolves), while C_HETERO chambers see CV *drop* as a
  resistant patch takes over and produces coherent expansion.  Re-validate
  this direction on BCG before trusting it there.
* **Validation chambers are kept disjoint from threshold-tuning chambers.**
  Thresholds were tuned on the original Pos101–Pos103 / Pos201–Pos203 set;
  those chambers have since been demoted to the reference pool and the
  verdict set rebuilt from fresh chambers (Pos120–125, Pos211–216,
  Pos201–205) that never participated in tuning.  On the fresh set:
  A 6/6 `NOT_DETECTED`, B no `DETECTED` false positives, C 5/5 flagged
  by at least one of Strategy C / `pop_score`.
* **Longer post-drug window is not always better.** Extending the
  C_HETERO window from 121 frames (~3 h post-drug) to 240 frames (~7 h)
  *dilutes* the `cv_trend < 1` signature, because the resistant-outgrowth
  "coherent expansion" phase is transient (1–2 h post-drug for RIF10 on
  M. smeg).  The per-chamber `max_frames` override is kept for future
  BCG experiments where the paper reports separation only at 6 h (RIF)
  or 12 h (INH).
