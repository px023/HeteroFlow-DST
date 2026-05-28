"""
test_heteroresistance_real_data.py

Real-data validation of Strategy C (flow divergence heteroresistance detection)
against three experimental groups with known ground truth.

Experimental context (Nature Communications, 2025 — Tran et al.):
  Organism    : M. smegmatis (fast-growing model for M. tuberculosis)
  Chip        : static microchambers, 50 x 60 x 1 µm
  Protocol    : 1 + 3 h  (1 h drug-free baseline, then 3 h RIF treatment)
  Frame rate  : 2 min / frame  =>  drug_start_frame = 30 (1 h / 2 min)
  Drug        : Rifampicin 10 mg/L = 10x MIC  (folder name "RIF10")
  Heterores.  : 99:1 susceptible:resistant mix  =>  1% resistant bacteria

Three test groups (known ground truth):
  A  REF (no drug)             — expected NOT_DETECTED  (negative control)
  B  RIF10 (100% susceptible)  — expected NOT_DETECTED  (uniform suppression)
  C  Heteroresistant           — expected DETECTED in >= 1 position

Detection parameters are tuned for the 1% resistance signal, which is subtle:
  - tiny spatial hotspot (~1-5% of mask area)
  - modest divergence CV increase (~1.3-2x baseline)

Usage:
    python test_heteroresistance_real_data.py
"""

import argparse
import os
import re
import sys
import glob
import math
import time
import warnings
from multiprocessing import Pool, cpu_count
warnings.filterwarnings("ignore")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
from typing import Dict, List, Optional, Tuple

import numpy as np
import skimage.io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

import config
from hybrid_pipeline import HybridSegmentationPipeline, GrowthAnalyzer
from heteroresistance_detector import (
    HeteroresistanceDetector,
    HeterogeneityTimeSeries,
    plot_heteroresistance_dashboard,
    score_heteroresistance_population,
    loo_population_score,
)


# ── Experiment constants ────────────────────────────────────────────────────

DRUG_START_FRAME = 30     # absolute frame number when drug is introduced
                          # = 1 h baseline / 2 min per frame (paper protocol)
MAX_FRAMES       = 121    # frames loaded per position: 1 h pre + 3 h post-drug
                          # 121 is the data limit of REF_*101_110 / RIF10_*201_210.
INTERVAL_MINUTES = 2.0    # minutes between consecutive frames

# ── Paths ───────────────────────────────────────────────────────────────────

DATA_DIR = "data"
EXP_DIR  = os.path.join(DATA_DIR, "EXP-25-CB5663_cropped_Heteroresistant")
OUT_DIR  = os.path.join("results", "heteroresistance_test")

# ── Detection parameters tuned for 1 % heteroresistance ────────────────────
# The 1 % resistant subpopulation creates a spatially localised divergence
# hotspot that is smaller and weaker than the default config values expect.
SENSITIVE_PARAMS: Dict[str, float] = {
    "DIV_CV_THRESHOLD":         0.5,   # default 0.8
    "DIV_HOTSPOT_SIGMA_THRESH": 1.2,   # default 1.5
    "DIV_HOTSPOT_MIN_AREA_PX":  20,    # default 50
    "DIV_PERSISTENCE_FRAMES":   3,     # default 4
}

# ── Visual style (dark-background, consistent with pipeline) ─────────────────

STYLE: Dict[str, Dict] = {
    "A_REF":    {"color": "#009ADE", "label": "Group A — REF (no drug)"},
    "B_RIF10":  {"color": "#FF6B35", "label": "Group B — RIF10 (100% susceptible)"},
    "C_HETERO": {"color": "#FF1F5B", "label": "Group C — Heteroresistant (1% resistant)"},
}

# ── Test positions ────────────────────────────────────────────────────────────

def _reg(sub, pos, folder):
    return os.path.join(DATA_DIR, sub, pos, folder)

def _exp(sub, pos, folder):
    return os.path.join(EXP_DIR, sub, pos, folder)

# ── TEST_POSITIONS: verdict set ────────────────────────

# Verdict set (17 chambers):
#   A_REF    : Pos120-125 (6, from REF_*_111_130_for_testing)
#   B_RIF10  : Pos211-216 (6, from RIF10_*_211_217_for_testing)
#   C_HETERO : Pos201-205 (all 5 TREAT chambers)
#
# Original verdict chambers (Pos101-103, Pos201-203) are demoted to the
# reference pool so they still anchor pop_score's mean curves but no longer
# decide the validation outcome.
TEST_POSITIONS: List[Dict] = [
    # ── Group A: no drug, negative control ──────────────────────────────────
    *[{"group": "A_REF", "id": f"Pos{_i}",
       "raw_dir":  _reg("REF_raw_data111_130_for_testing", f"Pos{_i}", "aphase"),
       "mask_dir": _reg("REF_masks111_131_for_testing",     f"Pos{_i}", "PreprocessedPhaseMasks")}
      for _i in range(120, 126)],

    # ── Group B: RIF10, all bacteria susceptible ─────────────────────────────
    *[{"group": "B_RIF10", "id": f"Pos{_i}",
       "raw_dir":  _reg("RIF10_raw_data211_217_for_testing", f"Pos{_i}", "aphase"),
       "mask_dir": _reg("RIF10_masks211_217_for_testing",    f"Pos{_i}", "PreprocessedPhaseMasks")}
      for _i in range(211, 217)],

    # ── Group C: heteroresistant experiment (1% resistant cells) ─────────────
    # All 5 TREAT chambers
    #
    # Note: we tried bumping C_HETERO to max_frames=240
    # (~7 h post-drug) to give cv_trend a longer late window.  Result was
    # *negative*: cv_trend dilutes back toward 1.0 over a longer window
    # because resistant outgrowth's "coherent expansion" signature is
    # transient (1-2 h post-drug for RIF10 on M. smeg), and later frames
    # mix in new asynchronous events.  Pop_score was unaffected (it uses
    # whole-curve shape, not late-window trend).  Kept the per-chamber
    # max_frames CAPABILITY in load_position/run_position/compute_pool_area_curve
    # for future BCG experiments (paper says RIF separates at 6 h, INH at 12 h)
    # but reverted the C_HETERO override here.
    *[{"group": "C_HETERO", "id": f"Pos{_p}",
       "raw_dir":  _exp("TREAT_raw_data", f"Pos{_p}", "PreprocessedPhase"),
       "mask_dir": _exp("TREAT_masks",    f"Pos{_p}", "PreprocessedPhaseMasks")}
      for _p in (201, 202, 203, 204, 205)],
]


# ── REFERENCE_POOL_POSITIONS: anchor pool for pop_score (NOT verdict) ───────
# Demoted from previous verdict set + the ref-pool positions
# verdict set didn't claim.
REFERENCE_POOL_POSITIONS: List[Dict] = []
# Original verdict A_REF chambers, now demoted to anchors
for _i in (101, 102, 103):
    REFERENCE_POOL_POSITIONS.append({
        "group": "A_REF", "id": f"Pos{_i}",
        "raw_dir":  _reg("REF_raw_data101_110", f"Pos{_i}", "aphase"),
        "mask_dir": _reg("REF_masks101_110",    f"Pos{_i}", "PreprocessedPhaseMasks"),
    })
# Ref-pool A_REF chambers verdict didn't pull (Pos111-119 + Pos126-130)
for _i in list(range(111, 120)) + list(range(126, 131)):
    REFERENCE_POOL_POSITIONS.append({
        "group": "A_REF", "id": f"Pos{_i}",
        "raw_dir":  _reg("REF_raw_data111_130_for_testing", f"Pos{_i}", "aphase"),
        "mask_dir": _reg("REF_masks111_131_for_testing",     f"Pos{_i}", "PreprocessedPhaseMasks"),
    })
# Original verdict B_RIF10 chambers, now demoted to anchors
for _i in (201, 202, 203):
    REFERENCE_POOL_POSITIONS.append({
        "group": "B_RIF10", "id": f"Pos{_i}",
        "raw_dir":  _reg("RIF10_raw_data201_210", f"Pos{_i}", "aphase"),
        "mask_dir": _reg("RIF10_masks201_210",    f"Pos{_i}", "PreprocessedPhaseMasks"),
    })
# Ref-pool B_RIF10 chambers verdict didn't pull (just Pos217)
REFERENCE_POOL_POSITIONS.append({
    "group": "B_RIF10", "id": "Pos217",
    "raw_dir":  _reg("RIF10_raw_data211_217_for_testing", "Pos217", "aphase"),
    "mask_dir": _reg("RIF10_masks211_217_for_testing",    "Pos217", "PreprocessedPhaseMasks"),
})


# ════════════════════════════════════════════════════════════════════════════
# 1.  Parameter management
# ════════════════════════════════════════════════════════════════════════════

_original_config: Dict = {}


def apply_sensitive_params(verbose: bool = True) -> None:
    """
    Override config.DIV_* with parameters tuned for 1 % heteroresistance.
    Original values are saved so they can be restored if needed.

    Pass verbose=False to suppress the announcement.
    """
    global _original_config
    for attr, val in SENSITIVE_PARAMS.items():
        _original_config[attr] = getattr(config, attr, None)
        setattr(config, attr, val)

    if verbose:
        print("Sensitive detection parameters applied:")
        for k, v in SENSITIVE_PARAMS.items():
            print(f"  config.{k} = {v}")


def restore_config() -> None:
    for attr, val in _original_config.items():
        if val is not None:
            setattr(config, attr, val)


def _pool_init() -> None:
    """multiprocessing.Pool initializer.
    """
    apply_sensitive_params(verbose=False)


# ════════════════════════════════════════════════════════════════════════════
# 2.  Data loading
# ════════════════════════════════════════════════════════════════════════════

def load_position(
    raw_dir: str,
    mask_dir: str,
    max_frames: int = MAX_FRAMES,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[int]]:
    """
    Load matched raw phase frames and Omnipose masks from one position.

    Supports two naming conventions used across the datasets:
      Sequential (0-based):  img_000000000.tiff  <->  MASK_img_000000000.tif
      Non-sequential:        img_000000046.tiff  <->  MASK_img_000000046.tif

    Frame pairs are matched by the integer embedded in their filenames.
    Empty mask files (0 bytes) and shape-mismatched pairs are skipped.

    Returns
    -------
    frames     : list of (H, W) float32 arrays
    masks      : list of (H, W) int32  arrays
    frame_nums : list of absolute frame numbers (for time-axis alignment)
    """
    raw_paths  = sorted(
        glob.glob(os.path.join(raw_dir,  "img_*.tiff")) +
        glob.glob(os.path.join(raw_dir,  "img_*.tif"))
    )
    mask_paths = sorted(
        glob.glob(os.path.join(mask_dir, "MASK_img_*.tif")) +
        glob.glob(os.path.join(mask_dir, "MASK_img_*.tiff"))
    )

    if not raw_paths:
        raise FileNotFoundError(f"No raw frames found in:\n  {raw_dir}")
    if not mask_paths:
        raise FileNotFoundError(f"No mask files found in:\n  {mask_dir}")

    _num = lambda p: int(re.search(r"(\d+)", os.path.basename(p)).group())
    raw_idx  = {_num(p): p for p in raw_paths}
    mask_idx = {_num(p): p for p in mask_paths}

    matched = sorted(raw_idx.keys() & mask_idx.keys())[:max_frames]
    if not matched:
        raise ValueError(
            f"No overlapping frame numbers between raw and mask:\n"
            f"  raw:  {raw_dir}\n  mask: {mask_dir}"
        )

    frames, masks, frame_nums = [], [], []
    for num in matched:
        # Skip empty mask files (bacteria fully gone — produces unreadable tiff)
        if os.path.getsize(mask_idx[num]) == 0:
            continue
        try:
            raw  = skimage.io.imread(raw_idx[num]).astype(np.float32)
            mask = skimage.io.imread(mask_idx[num])
        except Exception as exc:
            print(f"    Skipping frame {num}: {exc}")
            continue

        # Collapse channel dimension if present
        if raw.ndim  == 3: raw  = raw[...,  0]
        if mask.ndim == 3: mask = mask[..., 0]

        if raw.shape != mask.shape:
            dh = abs(raw.shape[0] - mask.shape[0])
            dw = abs(raw.shape[1] - mask.shape[1])
            if dh <= 2 and dw <= 2:
                # 1-2 pixel border discrepancy (common between raw and Omnipose masks)
                min_h = min(raw.shape[0], mask.shape[0])
                min_w = min(raw.shape[1], mask.shape[1])
                raw  = raw[:min_h,  :min_w]
                mask = mask[:min_h, :min_w]
            else:
                print(f"    Skipping frame {num}: shape mismatch "
                      f"raw={raw.shape} mask={mask.shape}")
                continue

        frames.append(raw)
        masks.append(mask.astype(np.int32))
        frame_nums.append(num)

    if not frames:
        raise ValueError(f"All frames skipped for {raw_dir}")

    return frames, masks, frame_nums


def drug_start_index(frame_nums: List[int],
                     drug_start_frame: int = DRUG_START_FRAME) -> int:
    """
    Return the index within the loaded sequence that corresponds to drug addition.
    If drug was added before the first loaded frame, returns 0.
    """
    for i, n in enumerate(frame_nums):
        if n >= drug_start_frame:
            return i
    return len(frame_nums)


# ════════════════════════════════════════════════════════════════════════════
# 3.  Paper's original detection method (area retention)
# ════════════════════════════════════════════════════════════════════════════

def paper_method_ratio(areas: np.ndarray, ds_idx: int) -> float:
    """
    Self-normalised area retention after drug addition.

    Reproduces the per-chamber criterion from Tran et al. (2025):
    chambers whose post-drug area resembles their own pre-drug baseline
    are flagged as heteroresistant candidates.

    Returns
    -------
    ratio : post_drug_mean / pre_drug_mean
      > 0.5  ->  area maintained  ->  heteroresistant candidate
      < 0.3  ->  area collapsed   ->  drug-susceptible
      nan    ->  insufficient pre- or post-drug data
    """
    pre  = areas[:ds_idx]
    post = areas[ds_idx:]
    if len(pre) < 3 or len(post) < 3:
        return math.nan
    pre_mean = float(np.mean(pre))
    if pre_mean < 1e-6:
        return math.nan
    return round(float(np.mean(post)) / pre_mean, 3)


# ════════════════════════════════════════════════════════════════════════════
# 4.  Verdict generation
# ════════════════════════════════════════════════════════════════════════════

def generate_verdict(
    results: HeterogeneityTimeSeries,
    ds_idx: int,
    post_mask_area_px: int,
) -> Dict:
    """
    Three-criterion verdict for heteroresistance in one position.

    C1  detection_flags_C fires at any post-drug frame
    C2  post-drug divergence CV > 1.3x pre-drug CV
    C4  late-post divergence CV  <  early-post divergence CV
        (heteroresistance: chaotic death resolves into a coherent
         resistant-cell expansion patch → spatial CV drops over time;
         pure susceptible kill never resolves → CV keeps growing)

    Rule:
      DETECTED      C1 AND C2 AND C4
      UNCERTAIN     C1 AND (C2 OR C4)
      NOT_DETECTED  otherwise

    `hotspot_pct` (fraction of mask occupied by persistent expansion hotspots) is
    reported as an informative value for inspection, but does not factor into
    the verdict — empirically it did not separate the experimental groups.

    Parameters
    ----------
    post_mask_area_px : peak segmented-pixel count over post-drug frames,
                        used to normalise the hotspot area fraction.
    """
    n_pairs  = len(results.divergence_cv)
    drug_tp  = max(0, ds_idx - 1)   # frame-pair index corresponding to drug start

    # ── C1: Strategy C binary flag ──────────────────────────────────────────
    post_flags = (results.detection_flags_C[drug_tp:]
                  if drug_tp < n_pairs else np.array([False]))
    c1 = bool(post_flags.any())

    # ── C2: divergence CV ratio (post vs pre drug) ──────────────────────────
    # Use median rather than mean: raw spatial CV values contain large
    # outliers (CV→∞ when mean divergence→0) that corrupt the mean.
    pre_cv   = results.divergence_cv[:drug_tp]
    post_cv  = results.divergence_cv[drug_tp: drug_tp + 30]
    pre_med  = float(np.median(pre_cv))  if len(pre_cv)  > 0 else 0.0
    post_med = float(np.median(post_cv)) if len(post_cv) > 0 else 0.0
    cv_ratio  = post_med / (pre_med + 1e-6)
    c2 = cv_ratio > 1.3

    # ── Informative: persistent hotspot area fraction (not in verdict) ──────
    hotspot_pct = 0.0
    if results.persistent_hotspot_map is not None and post_mask_area_px > 0:
        pmap = results.persistent_hotspot_map
        persistent_px = int((pmap >= config.DIV_PERSISTENCE_FRAMES).sum())
        hotspot_pct   = persistent_px / post_mask_area_px * 100.0

    #   B_RIF10 (100% susceptible, drugged):  cv_trend > 1
    #     Chaotic asynchronous death keeps producing heterogeneous flow →
    #     spatial CV of divergence keeps growing through the post-drug window.
    #
    #   C_HETERO (1% resistant mix):          cv_trend < 1
    #     The chaotic death phase resolves; surviving resistant cells produce
    #     a coherent positive-divergence patch → spatial CV drops as the
    #     signal becomes spatially concentrated.
    #
    # So heteroresistance fingerprint = "CV drops over time" (the chamber
    # settled into coherent expansion rather than continued chaotic death).
    early_window_frames = 30
    early_post = results.divergence_cv[drug_tp: drug_tp + early_window_frames]
    late_post  = results.divergence_cv[drug_tp + early_window_frames:]
    early_med  = float(np.median(early_post)) if len(early_post) > 3 else 0.0
    late_med   = float(np.median(late_post))  if len(late_post)  > 3 else 0.0
    cv_trend   = late_med / (early_med + 1e-6)
    c4 = cv_trend < 1.0   # heteroresistance: CV resolves downward

    # ── Overall verdict ──────────────────────────────────────────────────────
    if c1 and c2 and c4:
        verdict = "DETECTED"
    elif c1 and (c2 or c4):
        verdict = "UNCERTAIN"
    else:
        verdict = "NOT_DETECTED"

    # ── Time-to-first-detection (hours after drug addition) ─────────────────
    flagged = np.where(results.detection_flags_C[drug_tp:])[0]
    first_flag_h = (round(float(flagged[0]) * INTERVAL_MINUTES / 60.0, 2)
                    if len(flagged) > 0 else None)

    return {
        "verdict":      verdict,
        "c1":           c1,
        "c2":           c2,
        "c4":           c4,
        "cv_ratio":     round(cv_ratio,    3),
        "cv_trend":     round(cv_trend,    3),
        "hotspot_pct":  round(hotspot_pct, 2),
        "first_flag_h": first_flag_h,
    }


# ════════════════════════════════════════════════════════════════════════════
# 5.  Single-position pipeline
# ════════════════════════════════════════════════════════════════════════════

def run_position(pos: Dict) -> Optional[Dict]:
    """
    Full pipeline for one position:
      load  ->  hybrid segmentation  ->  detect  ->  verdict  ->  save dashboard

    Returns a result dict, or None if the position cannot be processed.
    """
    group = pos["group"]
    pid   = pos["id"]

    print(f"\n{'='*60}")
    print(f"  {STYLE[group]['label']}  |  {pid}")
    print(f"{'='*60}")

    # ── 1. Load data ─────────────────────────────────────────────────────────
    # Per-chamber max_frames override — C_HETERO chambers extend
    # to 240 frames so cv_trend has a ~6h late window; A_REF / B_RIF10 stay
    # at the global MAX_FRAMES=121 (~3h post-drug) since that's all they have.
    pos_max_frames = pos.get("max_frames", MAX_FRAMES)
    try:
        frames, omni_masks, frame_nums = load_position(
            pos["raw_dir"], pos["mask_dir"], max_frames=pos_max_frames
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"  SKIP: {exc}")
        return None

    n      = len(frames)
    ds_idx = drug_start_index(frame_nums)
    print(f"  Loaded {n} frames  "
          f"(abs. frame numbers {frame_nums[0]}–{frame_nums[-1]})")
    print(f"  Drug addition at frame {DRUG_START_FRAME} "
          f"-> local index {ds_idx}  "
          f"(pre: {ds_idx}, post: {n - ds_idx})")

    if n - ds_idx < 10:
        print("  WARNING: fewer than 10 post-drug frames — results unreliable")

    # ── 2. Hybrid segmentation (Gaussian + memory mask) ──────────────────────
    seg = HybridSegmentationPipeline(
        gaussian_sigma=config.GAUSSIAN_SIGMA,
    )
    refined_masks, _ = seg.process_sequence(frames, omni_masks, use_memory=True)

    # Post-drug peak mask area (used to normalise hotspot fraction)
    post_refined = refined_masks[ds_idx:] if ds_idx < n else refined_masks
    post_mask_area_px = int(
        np.mean([np.sum(m > 0) for m in post_refined]) if post_refined else 1
    )
    print(f"  Post-drug mean mask area: "
          f"{post_mask_area_px} px  "
          f"({post_mask_area_px * config.PIXEL_SIZE_UM**2:.1f} µm²)")

    # ── 3. Heteroresistance detection (Strategies B + C) ─────────────────────
    detector = HeteroresistanceDetector(
        cv_threshold_C=config.DIV_CV_THRESHOLD,
        persistence_frames=config.DIV_PERSISTENCE_FRAMES,
    )
    het_results = detector.run(refined_masks, frames, drug_start_frame=ds_idx)

    # ── 4. Verdict ────────────────────────────────────────────────────────────
    verdict = generate_verdict(het_results, ds_idx, post_mask_area_px)
    verdict.update({
        "position_id":           pid,
        "group":                 group,
        "n_frames":              n,
        "ds_idx":                ds_idx,
        "frame_nums":            frame_nums,
        "results":               het_results,
        # Tier 1D segmentation QC — surface the detector's clump warning
        "clump_warning":         bool(het_results.clump_warning),
        "baseline_cell_area_um2": float(het_results.baseline_cell_area_um2),
    })

    # ── 5. Paper method (area retention) ─────────────────────────────────────
    ga    = GrowthAnalyzer(interval_minutes=INTERVAL_MINUTES,
                           pixel_size_um=config.PIXEL_SIZE_UM)
    areas = ga.compute_area_growth(refined_masks)
    verdict["areas"]       = areas
    verdict["paper_ratio"] = paper_method_ratio(areas, ds_idx)

    # ── 6. Dashboard ─────────────────────────────────────────────────────────
    rep_idx   = min(ds_idx + 30, n - 1)
    dash_path = os.path.join(OUT_DIR, f"{group}_{pid}_dashboard.png")
    plot_heteroresistance_dashboard(
        het_results,
        raw_frame=frames[rep_idx],
        mask=refined_masks[rep_idx],
        drug_start_frame=ds_idx,
        interval_minutes=INTERVAL_MINUTES,
        rolling_window=config.HET_ROLLING_WINDOW,
        save_path=dash_path,
    )
    plt.close("all")

    # ── 7. Console summary ────────────────────────────────────────────────────
    marker     = ">>>" if verdict["verdict"] == "DETECTED" else "   "
    paper_str  = (f"{verdict['paper_ratio']:.3f}"
                  if not math.isnan(verdict["paper_ratio"]) else "n/a")
    first_str  = (f"+{verdict['first_flag_h']:.2f}h"
                  if verdict["first_flag_h"] is not None else "—")
    print(f"{marker} Verdict: {verdict['verdict']:<14}"
          f"C1={verdict['c1']}  C2={verdict['c2']}  C4={verdict['c4']}  "
          f"cv_ratio={verdict['cv_ratio']:.2f}  cv_trend={verdict['cv_trend']:.2f}  "
          f"hotspot={verdict['hotspot_pct']:.1f}%  "
          f"paper_ratio={paper_str}  "
          f"first_flag={first_str}")

    return verdict


def compute_pool_area_curve(pos: Dict) -> Optional[Dict]:
    """
    Cheap variant of run_position used only to feed the pop_score reference
    pool.  Loads frames + masks just long enough to extract the per-frame
    total-area curve; skips Strategy B/C, segmentation refinement, dashboard
    plotting.  Returns the minimum dict score_heteroresistance_population
    expects: {position_id, group, areas, ds_idx}.
    """
    pos_max_frames = pos.get("max_frames", MAX_FRAMES)
    try:
        frames, masks, frame_nums = load_position(
            pos["raw_dir"], pos["mask_dir"], max_frames=pos_max_frames
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"  [pool] SKIP {pos['group']}/{pos['id']}: {exc}")
        return None
    ds_idx = drug_start_index(frame_nums)
    ga = GrowthAnalyzer(
        interval_minutes=INTERVAL_MINUTES,
        pixel_size_um=config.PIXEL_SIZE_UM,
    )
    areas = ga.compute_area_growth(masks)
    return {
        "position_id": pos["id"],
        "group":       pos["group"],
        "areas":       areas,
        "ds_idx":      ds_idx,
        "n_frames":    len(frames),
    }


# ════════════════════════════════════════════════════════════════════════════
# 6.  Comparison figure
# ════════════════════════════════════════════════════════════════════════════

def plot_comparison_figure(all_verdicts: List[Dict]) -> None:
    """
    Two-panel figure comparing the paper's area method with Strategy C.

    Panel 1 — Normalised area growth (paper's original detection criterion):
      A heteroresistant chamber retains area close to 1.0 post-drug while
      susceptible chambers collapse toward 0.

    Panel 2 — Divergence spatial CV (Strategy C):
      Spatial CV of the flow-divergence field within the mask.  A rise after
      drug addition indicates spatial heterogeneity in growth/death, the
      signature of a surviving resistant subpopulation.
      Shaded bands mark frames where detection_flags_C is True.
    """
    fig = plt.figure(figsize=(14, 9), facecolor="#0d0d0d")
    gs  = plt.GridSpec(2, 1, figure=fig, hspace=0.38)

    def _ax(idx):
        ax = fig.add_subplot(gs[idx])
        ax.set_facecolor("#1a1a1a")
        for sp in ax.spines.values():
            sp.set_edgecolor("#444")
        ax.tick_params(colors="#aaa")
        ax.xaxis.label.set_color("#aaa")
        ax.yaxis.label.set_color("#aaa")
        ax.title.set_color("white")
        return ax

    ax_area = _ax(0)
    ax_div  = _ax(1)

    for v in all_verdicts:
        group  = v["group"]
        pid    = v["position_id"]
        ds_idx = v["ds_idx"]
        color  = STYLE[group]["color"]
        lw     = 2.5 if group == "C_HETERO" else 1.5
        alpha  = 1.0 if group == "C_HETERO" else 0.72
        lbl    = f"{STYLE[group]['label']} ({pid})"

        # ── Panel 1: normalised area ─────────────────────────────────────────
        areas    = v["areas"]
        n_a      = len(areas)
        t_a      = (np.arange(n_a) - ds_idx) * INTERVAL_MINUTES / 60.0
        pre_mean = float(np.mean(areas[:ds_idx])) if ds_idx > 0 else 1.0
        norm_a   = areas / max(pre_mean, 1e-6)
        ax_area.plot(t_a, norm_a, color=color, lw=lw, alpha=alpha, label=lbl)

        # ── Panel 2: divergence CV ───────────────────────────────────────────
        div_cv = v["results"].divergence_cv
        n_d    = len(div_cv)
        t_d    = (np.arange(n_d) - max(0, ds_idx - 1)) * INTERVAL_MINUTES / 60.0
        smooth = uniform_filter1d(div_cv.astype(float), size=5)
        ax_div.plot(t_d, smooth, color=color, lw=lw, alpha=alpha, label=lbl)

        # Shade Strategy C detection flags
        flags = v["results"].detection_flags_C
        for t_i, flagged in enumerate(flags):
            if flagged and t_i < len(t_d):
                ax_div.axvspan(
                    t_d[t_i],
                    t_d[min(t_i + 1, len(t_d) - 1)],
                    alpha=0.10, color=color,
                )

    # Shared decorations
    for ax in (ax_area, ax_div):
        ax.axvline(0, color="#FF1F5B", lw=1.8, linestyle="--", label="Drug addition (t=0)")
        ax.set_xlabel("Time relative to drug addition (h)")
        ax.grid(alpha=0.15)

    ax_area.axhline(1.0, color="#555", lw=1.0, linestyle=":")
    ax_area.axhline(0.5, color="#444", lw=0.8, linestyle=":", alpha=0.6)
    ax_area.set_ylabel("Normalised cell area")
    ax_area.set_title(
        "Panel 1 — Area growth (Tran et al. 2025 criterion)  |  "
        "heteroresistant chamber stays near 1.0 post-drug",
        fontsize=10,
    )
    ax_area.legend(fontsize=8, labelcolor="#aaa",
                   facecolor="#1a1a1a", edgecolor="#444", loc="upper right")

    ax_div.set_ylabel("Divergence spatial CV (smoothed, 5-frame window)")
    ax_div.set_title(
        "Panel 2 — Strategy C: divergence field variability  |  "
        "shading = detection_flags_C  (Group C should rise post-drug)",
        fontsize=10,
    )
    ax_div.legend(fontsize=8, labelcolor="#aaa",
                  facecolor="#1a1a1a", edgecolor="#444", loc="upper right")

    fig.suptitle(
        "Heteroresistance Detection: Paper Area Method vs Strategy C\n"
        "Protocol: M. smegmatis  |  RIF 10 mg/L  |  1 % resistant subpopulation",
        color="white", fontsize=12, fontweight="bold", y=0.99,
    )

    save_path = os.path.join(OUT_DIR, "comparison_area_vs_divergence_cv.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ════════════════════════════════════════════════════════════════════════════
# 7.  Summary report
# ════════════════════════════════════════════════════════════════════════════

def save_pop_candidate_strip(
    pos_meta: Dict,
    out_path: str,
    pop_score: float,
    n_strip: int = 6,
) -> bool:
    """Save a horizontal temporal strip of `n_strip` frames for human review.

    Picks pre-drug, drug-addition, +1h, +2h, +3h, last (or clamps to data
    length).  Each panel shows the raw phase image with the mask outline
    overlaid in red.  Useful for spot-checking whether a high pop_score is
    a real heteroresistance candidate or an artefact.

    Returns True if strip saved, False if data couldn't be loaded.
    """
    try:
        frames, masks, frame_nums = load_position(
            pos_meta["raw_dir"], pos_meta["mask_dir"]
        )
    except (FileNotFoundError, ValueError):
        return False

    ds_idx = drug_start_index(frame_nums)
    n = len(frames)
    if n < 4:
        return False

    # Six landmark timepoints — clamped to data range
    fpm = max(1, int(round(60.0 / INTERVAL_MINUTES)))  # frames per hour
    targets = [
        ("pre",  max(0, ds_idx - fpm)),
        ("drug", min(n - 1, ds_idx)),
        ("+1h",  min(n - 1, ds_idx + fpm)),
        ("+2h",  min(n - 1, ds_idx + 2 * fpm)),
        ("+3h",  min(n - 1, ds_idx + 3 * fpm)),
        ("last", n - 1),
    ]
    # Deduplicate while keeping order (some clamp to the same frame on short series)
    seen, picks = set(), []
    for label, i in targets:
        if i not in seen:
            picks.append((label, i))
            seen.add(i)

    fig, axes = plt.subplots(1, len(picks),
                             figsize=(3.0 * len(picks), 3.4),
                             facecolor="#0d0d0d")
    if len(picks) == 1:
        axes = [axes]
    for ax, (label, i) in zip(axes, picks):
        ax.imshow(frames[i], cmap="gray")
        # Overlay mask as a red-tinted alpha layer
        m = (masks[i] > 0)
        rgba = np.zeros((*m.shape, 4), dtype=np.float32)
        rgba[m] = [1.0, 0.2, 0.2, 0.35]
        ax.imshow(rgba, interpolation="nearest")
        # Hours relative to drug addition
        dt_h = (i - ds_idx) * INTERVAL_MINUTES / 60.0
        ax.set_title(f"{label}\nt={dt_h:+.2f}h",
                     color="white", fontsize=10)
        ax.axis("off")

    fig.suptitle(
        f"{pos_meta['group']} / {pos_meta['id']}  —  pop_score = {pop_score:+.3f}",
        color="white", fontsize=12, fontweight="bold",
    )
    fig.savefig(out_path, dpi=120, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    return True


def generate_pop_candidate_strips(
    all_chambers: List[Dict],
    pool_meta: List[Dict],
    test_meta: List[Dict],
    out_dir: str,
) -> int:
    """Generate mini-movie strips only for chambers showing **unexpected**
    pop_score behaviour — the cases that benefit from human inspection:

      * C_HETERO with pop_score > 0  → genuine heteroresistance candidate ★
      * A_REF    with pop_score < 0  → A_REF chamber that looks like kill ⚠
      * B_RIF10  with pop_score > 0  → B_RIF10 chamber that looks alive ⚠
      * (skip A_REF positive + B_RIF10 negative — those are the expected
        groupings, generating strips for them would just double the run
        time without adding information)
    """
    meta_by_id: Dict[str, Dict] = {p["id"]: p for p in pool_meta}
    for p in test_meta:
        meta_by_id.setdefault(p["id"], p)

    def is_interesting(ch: Dict) -> bool:
        s = ch.get("pop_score_loo")
        if not isinstance(s, float) or math.isnan(s):
            return False
        g = ch.get("group", "")
        return (
            (g == "C_HETERO" and s > 0) or       # heteroresistance candidate
            (g == "A_REF"    and s < 0) or       # unexpected: control looks dead
            (g == "B_RIF10"  and s > 0)          # unexpected: treated still alive
        )

    saved = 0
    for ch in all_chambers:
        if not is_interesting(ch):
            continue
        meta = meta_by_id.get(ch.get("position_id"))
        if meta is None:
            continue
        score = ch["pop_score_loo"]
        out_path = os.path.join(
            out_dir,
            f"strip_{ch['group']}_{ch['position_id']}_pop{score:+.3f}.png",
        )
        ok = save_pop_candidate_strip(meta, out_path, score)
        if ok:
            saved += 1
            print(f"  Strip → {os.path.basename(out_path)}")
    return saved


def print_loo_ranking(all_chambers: List[Dict]) -> None:
    """Print all chambers sorted by leave-one-out pop_score (high → low).

    Highlights:
      ★  C_HETERO chamber with positive score = expected heteroresistance candidate
      ⚠  B_RIF10 chamber with positive score = unexpected (spontaneous resistance?)
      ⚠  A_REF chamber with negative score   = unexpected (looks like a kill curve)
    """
    rows = [c for c in all_chambers
            if isinstance(c.get("pop_score_loo"), float)
            and not math.isnan(c["pop_score_loo"])]
    if not rows:
        print("  (no LOO scores computable — too few chambers in pool)")
        return
    rows.sort(key=lambda r: r["pop_score_loo"], reverse=True)

    row_fmt = "{flag:<3} {pos:<10} {grp:<10} {ctrl:>9}  {peer:>9}  {score:>+9.3f}"
    hdr_fmt = "{flag:<3} {pos:<10} {grp:<10} {ctrl:>9}  {peer:>9}  {score:>9}"
    print("\n" + "─" * 60)
    print(" Leave-one-out pop_score ranking (sorted high → low)")
    print("─" * 60)
    print(hdr_fmt.format(flag="", pos="Position", grp="Group",
                        ctrl="sim_ctrl", peer="sim_peer", score="pop_score"))
    print("─" * 60)
    for r in rows:
        g = r.get("group", "")
        s = r["pop_score_loo"]
        flag = ""
        if g == "C_HETERO" and s > 0:
            flag = "★"
        elif g == "B_RIF10" and s > 0:
            flag = "⚠"
        elif g == "A_REF" and s < 0:
            flag = "⚠"
        print(row_fmt.format(
            flag=flag,
            pos=r.get("position_id", ""),
            grp=g,
            ctrl=f"{r.get('sim_control_loo', 0):.3f}",
            peer=f"{r.get('sim_peer_loo', 0):.3f}",
            score=s,
        ))
    print("─" * 60)
    print(" ★ = heteroresistance candidate  |  ⚠ = unexpected group behaviour")


def print_and_save_summary(all_verdicts: List[Dict]) -> None:
    """Print a formatted summary table and write it to summary_report.txt."""
    col = ("{:<12} {:<10} {:<14} {:<5} {:<5} {:<5}"
           " {:<10} {:<10} {:<10} {:<12} {:<12} {:<10} {:<6} {:<6}")
    hdr = col.format(
        "Position", "Group", "Verdict",
        "C1", "C2", "C4",
        "CV_ratio", "CV_trend", "Hotspot%", "PaperRatio", "First_flag",
        "PopScore", "PopCand", "Clump",
    )
    sep = "─" * len(hdr)

    lines = [sep, hdr, sep]
    for v in all_verdicts:
        flag_str  = (f"+{v['first_flag_h']:.2f}h"
                     if v["first_flag_h"] is not None else "—")
        paper_str = (f"{v['paper_ratio']:.3f}"
                     if not math.isnan(v["paper_ratio"]) else "n/a")
        pop_score = v.get("pop_score", float("nan"))
        pop_score_str = (f"{pop_score:+.3f}"
                        if isinstance(pop_score, float) and not math.isnan(pop_score)
                        else "n/a")
        lines.append(col.format(
            v["position_id"],
            v["group"],
            v["verdict"],
            "yes" if v["c1"] else "no",
            "yes" if v["c2"] else "no",
            "yes" if v["c4"] else "no",
            f"{v['cv_ratio']:.3f}",
            f"{v['cv_trend']:.3f}",
            f"{v['hotspot_pct']:.2f}",
            paper_str,
            flag_str,
            pop_score_str,
            "yes" if v.get("pop_candidate") else "no",
            "YES" if v.get("clump_warning") else "no",
        ))
    lines.append(sep)

    report = "\n".join(lines)
    print("\n" + report)

    path = os.path.join(OUT_DIR, "summary_report.txt")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(
            "Heteroresistance Detection — Real Data Test\n"
            "Paper: Tran et al., Nat. Commun. 2025\n"
            "Protocol: M. smegmatis, 1+3h, RIF 10 mg/L, "
            f"drug_start_frame={DRUG_START_FRAME}\n"
            "Parameters: sensitive mode (see SENSITIVE_PARAMS in script)\n\n"
        )
        fh.write(report + "\n")
    print(f"\n  Report saved -> {path}")


# ════════════════════════════════════════════════════════════════════════════
# 8.  Automated validation
# ════════════════════════════════════════════════════════════════════════════

def validate_results(all_verdicts: List[Dict]) -> bool:
    """
    Check outcomes against known ground truth.

    Pass criteria
    -------------
    Group A  All positions NOT_DETECTED  (no false positives from normal growth)
    Group B  All positions NOT_DETECTED  (uniform drug suppression)
             Bonus check: cv_ratio < 1.2  (CV should not rise under uniform kill)
    Group C  >= 1 position DETECTED      (at least one heteroresistant chamber found)
    Cross    At least one DETECTED position also flagged by paper's area method
             (paper_ratio > 0.4 for the same position)
    """
    ga = [v for v in all_verdicts if v["group"] == "A_REF"]
    gb = [v for v in all_verdicts if v["group"] == "B_RIF10"]
    gc = [v for v in all_verdicts if v["group"] == "C_HETERO"]

    print("\n" + "=" * 50)
    print("AUTOMATED VALIDATION")
    print("=" * 50)

    # Group A
    a_nd   = [v for v in ga if v["verdict"] == "NOT_DETECTED"]
    a_pass = len(a_nd) == len(ga) if ga else False
    print(f"Group A (negative control) : {'PASS' if a_pass else 'FAIL'}"
          f"  {len(a_nd)}/{len(ga)} NOT_DETECTED")

    # Group B
    b_nd      = [v for v in gb if v["verdict"] == "NOT_DETECTED"]
    b_cv_ok   = [v for v in gb if v["cv_ratio"] < 1.2]
    b_pass    = len(b_nd) == len(gb) if gb else False
    print(f"Group B (100% susceptible) : {'PASS' if b_pass else 'FAIL'}"
          f"  {len(b_nd)}/{len(gb)} NOT_DETECTED"
          f"  |  {len(b_cv_ok)}/{len(gb)} cv_ratio < 1.2  (uniform suppression)")

    # Group C
    c_det  = [v for v in gc if v["verdict"] == "DETECTED"]
    c_pass = len(c_det) >= 1 if gc else False
    print(f"Group C (heteroresistant)  : {'PASS' if c_pass else 'FAIL'}"
          f"  {len(c_det)}/{len(gc)} DETECTED")

    # Cross-validation with paper method
    c_paper_hi = [v for v in gc
                  if not math.isnan(v["paper_ratio"]) and v["paper_ratio"] > 0.4]
    det_ids    = {v["position_id"] for v in c_det}
    paper_ids  = {v["position_id"] for v in c_paper_hi}
    agree_ids  = det_ids & paper_ids
    print(f"  Paper method cross-check : "
          f"{len(c_paper_hi)}/{len(gc)} Group C with paper_ratio > 0.4")
    if agree_ids:
        print(f"  Both methods agree on  : {sorted(agree_ids)}")
    elif c_det and not agree_ids:
        print("  NOTE: Strategy C detected heteroresistance but paper method "
              "did not flag same position(s)  — may need lower paper_ratio threshold")

    # Segmentation quality (Tier 1D): any chamber where Omnipose appears to have
    # merged cells into clumps?  Surface as a low-confidence advisory — does not
    # change the pass/fail verdict, but flags the run as needing manual review.
    clumped = [v for v in all_verdicts if v.get("clump_warning")]
    if clumped:
        ids = ", ".join(f"{v['group']}/{v['position_id']}" for v in clumped)
        print(f"\n  ⚠ Segmentation QC: {len(clumped)} chamber(s) flagged as "
              f"possibly clumped — verdicts here are low-confidence:")
        print(f"     {ids}")

    overall = a_pass and b_pass and c_pass
    status  = "ALL CHECKS PASSED" if overall else "SOME CHECKS FAILED"
    print(f"\n{status}")
    print("=" * 50)
    return overall


# ════════════════════════════════════════════════════════════════════════════
# 9.  Entry point
# ════════════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    """CLI options.

    --organism    pick a config.ORGANISM_PROFILES entry so frame interval +
                  rolling window lengths match the imaged organism.  Default
                  M_smegmatis is fast-growing (2 min/frame, ~30 min windows);
                  switch to M_bovis_BCG for slow-grower BCG datasets
                  (10 min/frame, 3 h windows per Tran et al. 2025).
    """
    p = argparse.ArgumentParser(description="Heteroresistance real-data test")
    p.add_argument(
        "--organism", default="M_smegmatis",
        choices=tuple(config.ORGANISM_PROFILES.keys()),
        help="Organism profile (default: M_smegmatis)",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    print("=" * 60)
    print("HETERORESISTANCE REAL-DATA TEST")
    print("Strategy C: Flow Divergence Analysis")
    print("=" * 60)

    # Apply organism profile BEFORE banner so INTERVAL_MINUTES / window sizes
    # reflect the selected organism.
    config.apply_organism_profile(args.organism)
    # Also update this module's INTERVAL_MINUTES copy (set at import time).
    global INTERVAL_MINUTES
    INTERVAL_MINUTES = config.INTERVAL_MINUTES

    print(f"\nExperimental protocol (Tran et al., Nat. Commun. 2025):")
    print(f"  Organism       : {args.organism}")
    print(f"  Drug           : Rifampicin {10} mg/L  (10x MIC)")
    print(f"  Protocol       : 1 + 3 h  =>  drug_start_frame = {DRUG_START_FRAME}")
    print(f"  Frame interval : {INTERVAL_MINUTES} min")
    print(f"  Rolling window : {config.HET_ROLLING_WINDOW} frames "
          f"(~{config.HET_ROLLING_WINDOW * INTERVAL_MINUTES:.0f} min)")
    # Show the global default + any per-chamber overrides in TEST_POSITIONS
    custom_caps = sorted({p.get("max_frames", MAX_FRAMES)
                         for p in TEST_POSITIONS} - {MAX_FRAMES})
    print(f"  Max frames     : {MAX_FRAMES}  (default — 1 h pre-drug + "
          f"{(MAX_FRAMES - DRUG_START_FRAME) * INTERVAL_MINUTES / 60.0:.1f} h post-drug)")
    if custom_caps:
        for cap in custom_caps:
            groups = sorted({p["group"] for p in TEST_POSITIONS
                            if p.get("max_frames", MAX_FRAMES) == cap})
            post_h = (cap - DRUG_START_FRAME) * INTERVAL_MINUTES / 60.0
            print(f"                   {cap:>4} for {','.join(groups)} "
                  f"(1 h pre-drug + {post_h:.1f} h post-drug)")
    print(f"  Heterores. mix : 99:1 susceptible:resistant  (1 % resistant)")

    os.makedirs(OUT_DIR, exist_ok=True)
    apply_sensitive_params()

    # ── Position-level parallelism ───────────────────────────────────────────
    # Each position is independent (own frames, own masks, own dashboard).
    # Default: up to 4 workers (RAM-bound: each position holds ~200-500 MB
    # of frame data).  Override via env var:
    #   HET_TEST_WORKERS=1        force sequential (useful for debugging)
    #   HET_TEST_WORKERS=N        use N workers
    env_workers = os.environ.get("HET_TEST_WORKERS", "").strip()
    try:
        n_workers = int(env_workers) if env_workers else 0
    except ValueError:
        n_workers = 0
    if n_workers <= 0:
        n_workers = min(cpu_count(), 4, len(TEST_POSITIONS))

    t0 = time.time()
    if n_workers >= 2:
        print(f"\nRunning {len(TEST_POSITIONS)} verdict positions "
              f"+ {len(REFERENCE_POOL_POSITIONS)} ref-pool positions "
              f"in parallel ({n_workers} workers) ...\n")
        with Pool(processes=n_workers, initializer=_pool_init) as pool:
            raw_results = pool.map(run_position, TEST_POSITIONS)
            # Cheap pass for the reference pool — just area curves, no Strategy B/C
            pool_curves = pool.map(compute_pool_area_curve, REFERENCE_POOL_POSITIONS)
    else:
        print(f"\nRunning {len(TEST_POSITIONS)} verdict positions "
              f"+ {len(REFERENCE_POOL_POSITIONS)} ref-pool positions "
              f"sequentially ...\n")
        raw_results = [run_position(pos) for pos in TEST_POSITIONS]
        pool_curves = [compute_pool_area_curve(pos) for pos in REFERENCE_POOL_POSITIONS]

    all_verdicts: List[Dict] = [r for r in raw_results if r is not None]
    pool_extra:   List[Dict] = [p for p in pool_curves if p is not None]

    if not all_verdicts:
        print("\nERROR: no positions were processed successfully.")
        restore_config()
        sys.exit(1)

    processed = len(all_verdicts)
    total     = len(TEST_POSITIONS)
    print(f"\nProcessed {processed}/{total} verdict positions + "
          f"{len(pool_extra)}/{len(REFERENCE_POOL_POSITIONS)} ref-pool "
          f"in {time.time() - t0:.1f}s.")

    # ── Population-level heteroresistance scoring ─────────
    # Reference (control) pool: original 3 A_REF verdict chambers
    #                         + 20 A_REF Pos111-130 ref-pool chambers
    # Peer pool: original 3 B_RIF10 verdict chambers + 7 Pos211-217 ref-pool
    # Test set: the original 9 verdict chambers (we still score B_RIF10 +
    # C_HETERO so the summary table shows their pop_score; A_REF gets nan).
    ctrl_verdict = [v for v in all_verdicts if v["group"] == "A_REF"]
    peer_verdict = [v for v in all_verdicts if v["group"] == "B_RIF10"]
    ctrl_pool    = ctrl_verdict + [p for p in pool_extra if p["group"] == "A_REF"]
    peer_pool    = peer_verdict + [p for p in pool_extra if p["group"] == "B_RIF10"]
    test         = [v for v in all_verdicts if v["group"] in ("B_RIF10", "C_HETERO")]

    if ctrl_pool and peer_pool and test:
        print(f"\nComputing population-level heteroresistance score "
              f"(|ctrl|={len(ctrl_pool)}, |peer|={len(peer_pool)}, "
              f"|test|={len(test)}) ...")
        pop = score_heteroresistance_population(test, ctrl_pool, peer_pool)
        for v, p in zip(test, pop):
            v["pop_score"]     = p["pop_score"]
            v["pop_candidate"] = p["pop_candidate"]
            v["sim_control"]   = p["sim_control"]
            v["sim_peer"]      = p["sim_peer"]
    for v in all_verdicts:
        v.setdefault("pop_score",     float("nan"))
        v.setdefault("pop_candidate", False)
        v.setdefault("sim_control",   float("nan"))
        v.setdefault("sim_peer",      float("nan"))

    # Pool everyone (verdict positions + ref-pool extras) into one list, then
    # compute LOO pop_score: each chamber is evaluated against control + peer
    # means that exclude itself.  This surfaces internal outliers within the
    # reference groups (e.g. an A_REF chamber that secretly behaves like
    # treated would get a negative score) and gives every ref-pool chamber a
    # score for inspection, not just the verdict 9.
    all_chambers = all_verdicts + pool_extra
    print(f"\nComputing leave-one-out pop_score over all "
          f"{len(all_chambers)} chambers ...")
    loo = loo_population_score(all_chambers)
    for ch, l in zip(all_chambers, loo):
        ch["pop_score_loo"]     = l["pop_score_loo"]
        ch["pop_candidate_loo"] = l["pop_candidate_loo"]
        ch["sim_control_loo"]   = l["sim_control_loo"]
        ch["sim_peer_loo"]      = l["sim_peer_loo"]

    print_loo_ranking(all_chambers)

    # ──  per-candidate mini-movie strips ────────────────────────────
    # Every chamber with positive LOO pop_score gets a 6-panel temporal
    # strip saved so the user can eyeball whether the heteroresistance call
    # is plausible (or an artefact).
    print("\nGenerating pop-candidate mini-movie strips ...")
    n_strips = generate_pop_candidate_strips(
        all_chambers,
        pool_meta=REFERENCE_POOL_POSITIONS,
        test_meta=TEST_POSITIONS,
        out_dir=OUT_DIR,
    )
    print(f"  Saved {n_strips} candidate strips (for unexpected-behaviour chambers).")

    print("\nGenerating comparison figure ...")
    plot_comparison_figure(all_verdicts)

    print_and_save_summary(all_verdicts)
    validate_results(all_verdicts)

    restore_config()
    print(f"\nAll outputs saved to: {OUT_DIR}/")
    print("  • <Group>_<Pos>_dashboard.png  — 3x3 panel per position")
    print("  • strip_<Group>_<Pos>_pop*.png  — 6-frame timeline for high-pop_score chambers")
    print("  • comparison_area_vs_divergence_cv.png  — group comparison")
    print("  • summary_report.txt  — verdict table + LOO ranking")


if __name__ == "__main__":
    main()
