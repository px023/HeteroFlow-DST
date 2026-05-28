"""

Heteroresistance detection for time-lapse phase-contrast microscopy of
mycobacteria in microfluidic chambers.

Two complementary strategies are implemented:

  Strategy B – Optical-flow field variability
    Compute a dense Farnebäck optical-flow field between consecutive frames
    and measure the *spatial variability* (coefficient of variation, local
    entropy, and high-magnitude hot-spots) of the magnitude map inside the
    segmentation mask.

  Strategy C – Flow divergence field variability
    Differentiate the dense optical flow to obtain div(v) = ∂u/∂x + ∂v/∂y
    and measure the spatial CV of divergence inside a fixed pre-drug union
    mask.  Heteroresistance produces sustained positive-divergence hotspots
    where the resistant minority continues to expand.

A global growth-rate estimator is used to gate detection on drug-induced
global suppression.  (An earlier "Strategy A" quadtree analyzer was shown
to be empirically redundant with Strategy C and has been removed.)

Each strategy produces:
  - a per-frame scalar heteroresistance score
  - a spatial heatmap (same pixel resolution as the raw image)
  - binary detection flags after persistence filtering

The module is designed to slot directly into the existing pipeline:

    from heteroresistance_detector import HeteroresistanceDetector
    detector = HeteroresistanceDetector()
    results = detector.run(masks, frames, drug_start_frame=30)

"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from scipy.ndimage import gaussian_filter, label as ndlabel
from skimage.measure import regionprops

import config


@dataclass
class HeterogeneityTimeSeries:
    """All scalar metrics produced by a single detector run."""

    # Global growth (used to gate Strategy C detection)
    global_growth_rates: np.ndarray = field(default_factory=lambda: np.array([]))

    # Strategy C — flow divergence field variability (the only detector;
    # Strategy B / FlowVariabilityAnalyzer was removed because
    # its spatial CV of |flow| could not discriminate between heterogeneous
    # death and heteroresistance on real data).
    divergence_cv: np.ndarray = field(default_factory=lambda: np.array([]))
    mean_divergence: np.ndarray = field(default_factory=lambda: np.array([]))
    divergence_hotspot_fraction: np.ndarray = field(default_factory=lambda: np.array([]))
    detection_flags_C: np.ndarray = field(default_factory=lambda: np.array([], dtype=bool))

    # Combined score (now identical to score_C_norm; kept for backward
    # compatibility with dashboard / downstream code).
    combined_score: np.ndarray = field(default_factory=lambda: np.array([]))
    detection_combined: np.ndarray = field(default_factory=lambda: np.array([], dtype=bool))

    # Spatial outputs
    flow_magnitude_map: Optional[np.ndarray] = None       # kept as a diagnostic overlay
    divergence_map: Optional[np.ndarray] = None           # representative divergence map
    persistent_hotspot_map: Optional[np.ndarray] = None  # per-pixel hotspot accumulation

    # Quality-control flags
    clump_warning: bool = False        # pre-drug median cell area >> single-cell
    baseline_cell_area_um2: float = 0.0  # for diagnostic printing


# ---------------------------------------------------------------------------
# Lightweight global growth estimator (replaces the legacy quadtree analyzer)
# ---------------------------------------------------------------------------

class GlobalGrowthEstimator:
    """Light-weight replacement for the legacy SpatialGrowthAnalyzer.

    Only computes the rolling global exponential growth rate (h^-1) used by
    Strategies B and C to gate detection on drug-induced global suppression.
    The full per-tile quadtree analysis was shown to be empirically redundant
    with Strategy C (flow divergence) and has been removed.
    """

    def __init__(self, rolling_window: Optional[int] = None):
        self.rolling_window = (rolling_window if rolling_window is not None
                               else config.HET_ROLLING_WINDOW)

    def _fit_log_linear(self, y: np.ndarray) -> float:
        y = np.clip(y, 1e-9, None)
        x = np.arange(len(y), dtype=float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            slope, *_ = stats.linregress(x, np.log(y))
        return float(slope / config.INTERVAL_MINUTES * 60.0)

    def compute(self, masks: List[np.ndarray]) -> np.ndarray:
        """Rolling log-linear fit on total segmented area. Returns shape (n_tp,)."""
        px_area = config.PIXEL_SIZE_UM ** 2
        total = np.array([(m > 0).sum() * px_area for m in masks], dtype=np.float32)
        rw = self.rolling_window
        n_tp = max(0, len(total) - rw)
        if n_tp == 0:
            return np.zeros(0, dtype=np.float32)
        rates = np.zeros(n_tp, dtype=np.float32)
        for t in range(n_tp):
            rates[t] = self._fit_log_linear(total[t: t + rw])
        return rates


# ---------------------------------------------------------------------------
# Shared detection helpers (used by Strategies B and C)
# ---------------------------------------------------------------------------

def _align_global_rates(g_in: np.ndarray, n: int) -> np.ndarray:
    """Resize a global-growth-rate series to length n by forward-filling.

    The global growth estimator produces fewer points than the per-frame-pair
    flow metrics (rolling window).  Forward-filling the tail with the last
    valid rate avoids treating "no data" as "rate = 0 = suppressed", which
    would otherwise spuriously trigger the suppression check at the end of
    the recording.
    """
    out = np.zeros(n, dtype=np.float32)
    n_avail = min(n, len(g_in))
    if n_avail == 0:
        return out
    out[:n_avail] = g_in[:n_avail]
    if n_avail < n:
        out[n_avail:] = g_in[n_avail - 1]
    return out


def _to_uint8(frame: np.ndarray) -> np.ndarray:
    """Normalise an arbitrary-dtype frame to uint8 for OpenCV consumption."""
    f = frame.astype(np.float32)
    lo, hi = f.min(), f.max()
    if hi - lo < 1e-6:
        return np.zeros_like(f, dtype=np.uint8)
    return ((f - lo) / (hi - lo) * 255).astype(np.uint8)


# Default Farneback parameters used by both Strategy B and Strategy C.
# Sharing this dict ensures the two strategies operate on identical flow
# fields, allowing the optical-flow computation to be performed once per
# frame pair instead of twice.
DEFAULT_FARNEBACK_PARAMS = dict(
    pyr_scale=0.5,
    levels=3,
    winsize=15,
    iterations=3,
    poly_n=5,
    poly_sigma=1.2,
    flags=0,
)


def _compute_farneback_uv(
    frame1: np.ndarray,
    frame2: np.ndarray,
    params: Optional[Dict] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run Farneback once and return the (u, v) flow components.

    Pulled out of the analyzer classes so a single call can be shared
    between Strategy B (flow magnitude variability) and Strategy C
    (flow divergence variability), which use the same Farneback params.
    """
    f1 = _to_uint8(frame1)
    f2 = _to_uint8(frame2)
    flow = cv2.calcOpticalFlowFarneback(
        f1, f2, None, **(params or DEFAULT_FARNEBACK_PARAMS)
    )
    return flow[..., 0].astype(np.float32), flow[..., 1].astype(np.float32)


def _persistence_filter(raw: np.ndarray, k: int) -> np.ndarray:
    """Return a bool array marking frames inside any run of >= k consecutive Trues.

    Mirrors the original inline implementation: each time a run of length
    >= k ends at index t, the k-frame window ending at t is set True.
    """
    flags = np.zeros_like(raw, dtype=bool)
    run = 0
    for t in range(len(raw)):
        run = run + 1 if raw[t] else 0
        if run >= k:
            flags[t - k + 1: t + 1] = True
    return flags


# ---------------------------------------------------------------------------
# Strategy C: Flow divergence field variability
# ---------------------------------------------------------------------------

class FlowDivergenceAnalyzer:
    """
    Strategy C: Flow Divergence Field Analysis.

    Computes the divergence of the Farneback dense optical-flow field:

        div(v) = du/dx + dv/dy

    where u = horizontal flow component, v = vertical flow component.

    Physical interpretation
    -----------------------
    div > 0  ->  local expansion  -> cells growing and pushing outward
    div < 0  ->  local contraction -> cells shrinking or dying
    div ~= 0  ->  pure translational drift (no local volume change)

    This is more specific than flow magnitude (Strategy B), which conflates
    translation with growth.  A bacterium pushed by its neighbours has a high
    magnitude but near-zero divergence.  A growing bacterium expands outward
    and produces positive divergence in its neighbourhood.

    Heteroresistance signature
    --------------------------
    After drug addition in a heteroresistant sample:
      - global mean divergence -> near zero  (susceptible majority suppressed)
      - spatial CV of divergence rises       (resistant minority still expanding)
      - persistent positive-divergence hotspots remain spatially clustered

    In a fully susceptible sample:
      - global mean divergence -> zero or negative
      - spatial CV also decreases (uniform suppression — all tiles behave alike)

    Parameters
    ----------
    farneback_params         : dict passed verbatim to cv2.calcOpticalFlowFarneback
    flow_smooth_sigma        : Gaussian sigma applied to u and v before differentiation
    div_smooth_sigma         : Gaussian sigma applied to the raw divergence map
    hotspot_sigma_thresh     : a pixel is a hotspot when div > mean + k * std
    hotspot_min_area_px      : minimum connected-component area (px) to keep a hotspot
    persistence_frames       : frames a hotspot must persist to count
    cv_threshold             : spatial CV threshold for the binary detection flag
    global_suppression_threshold : growth rate (h^-1) below which global growth is
                                   considered suppressed by the drug
    """

    def __init__(
        self,
        farneback_params: Optional[Dict] = None,
        flow_smooth_sigma: Optional[float] = None,
        div_smooth_sigma: Optional[float] = None,
        hotspot_sigma_thresh: Optional[float] = None,
        hotspot_min_area_px: Optional[int] = None,
        persistence_frames: Optional[int] = None,
        cv_threshold: Optional[float] = None,
        global_suppression_threshold: float = 0.03,
    ):
        self.farneback_params = farneback_params or DEFAULT_FARNEBACK_PARAMS
        self.flow_smooth_sigma = flow_smooth_sigma if flow_smooth_sigma is not None \
            else config.DIV_FLOW_SMOOTH_SIGMA
        self.div_smooth_sigma = div_smooth_sigma if div_smooth_sigma is not None \
            else config.DIV_MAP_SMOOTH_SIGMA
        self.hotspot_sigma_thresh = hotspot_sigma_thresh if hotspot_sigma_thresh is not None \
            else config.DIV_HOTSPOT_SIGMA_THRESH
        self.hotspot_min_area_px = hotspot_min_area_px if hotspot_min_area_px is not None \
            else config.DIV_HOTSPOT_MIN_AREA_PX
        self.persistence_frames = persistence_frames if persistence_frames is not None \
            else config.DIV_PERSISTENCE_FRAMES
        self.cv_threshold = cv_threshold if cv_threshold is not None \
            else config.DIV_CV_THRESHOLD
        self.global_suppression_threshold = global_suppression_threshold

    # ------------------------------------------------------------------
    # Core computation: divergence of a single frame pair
    # ------------------------------------------------------------------

    def divergence_from_flow(
        self,
        u: np.ndarray,
        v: np.ndarray,
        mask_bool: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        Compute the divergence map from precomputed (u, v) flow components.
        Lets the Farneback step be shared with Strategy B.

        Pipeline
        --------
        1. Gaussian-smooth u and v       ->  reduce high-frequency noise before
                                             numerical differentiation
        2. Central-difference divergence ->  du/dx + dv/dy  via np.gradient
           Note: np.gradient(arr) on a 2-D array returns [d/drow, d/dcol],
           i.e. [d/dy, d/dx] in image coordinates, so:
               du_dx = np.gradient(u_smooth)[1]
               dv_dy = np.gradient(v_smooth)[0]
        3. Gaussian-smooth the divergence ->  cleaner spatial map for visualisation
        4. Apply mask: background pixels set to NaN
        """
        # cv2.GaussianBlur is 2-5× faster than scipy.ndimage.gaussian_filter
        # for float32 inputs.  ksize=(0,0) lets OpenCV derive kernel size from
        # sigma (same default behaviour as gaussian_filter).
        u_s = cv2.GaussianBlur(u, ksize=(0, 0), sigmaX=self.flow_smooth_sigma)
        v_s = cv2.GaussianBlur(v, ksize=(0, 0), sigmaX=self.flow_smooth_sigma)

        du_dx = np.gradient(u_s)[1]   # d/d(col) = d/dx
        dv_dy = np.gradient(v_s)[0]   # d/d(row) = d/dy
        divergence = (du_dx + dv_dy).astype(np.float32)
        divergence = cv2.GaussianBlur(
            divergence, ksize=(0, 0), sigmaX=self.div_smooth_sigma
        )

        magnitude = np.sqrt(u_s ** 2 + v_s ** 2)
        magnitude[~mask_bool] = 0.0
        # In-place NaN write; no need to copy first since divergence is the
        # output of cv2.GaussianBlur (a fresh allocation).
        divergence[~mask_bool] = np.nan

        return {
            "divergence": divergence,
            "u": u_s,
            "v": v_s,
            "magnitude": magnitude,
        }

    # ------------------------------------------------------------------
    # Hotspot detection
    # ------------------------------------------------------------------

    def _detect_hotspots(
        self,
        divergence_map: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        """
        Identify pixels with divergence significantly above the chamber mean.

        A pixel qualifies as a hotspot when:
            divergence > nanmean(mask) + hotspot_sigma_thresh * nanstd(mask)

        Small isolated regions smaller than hotspot_min_area_px are removed via
        connected-component filtering to suppress segmentation noise.

        Returns
        -------
        hotspot_binary : (H, W) bool array — True where a hotspot exists
        """
        vals = divergence_map[mask > 0]
        valid = vals[np.isfinite(vals)]
        if len(valid) < 20:
            return np.zeros(divergence_map.shape, dtype=bool)

        mu = float(np.nanmean(valid))
        sigma_v = float(np.nanstd(valid))
        threshold = mu + self.hotspot_sigma_thresh * sigma_v

        raw_hotspot = (
            np.isfinite(divergence_map)
            & (divergence_map > threshold)
            & (mask > 0)
        )

        # Remove connected components smaller than the minimum area
        labeled, n_labels = ndlabel(raw_hotspot)
        cleaned = np.zeros_like(raw_hotspot, dtype=bool)
        for region_id in range(1, n_labels + 1):
            if (labeled == region_id).sum() >= self.hotspot_min_area_px:
                cleaned[labeled == region_id] = True

        return cleaned

    # ------------------------------------------------------------------
    # Per-pair metrics (from precomputed flow)
    # ------------------------------------------------------------------

    def compute_pair_metrics(
        self,
        u: np.ndarray,
        v: np.ndarray,
        mask_bool: np.ndarray,
        n_mask_px: Optional[int] = None,
    ) -> Dict:
        """
        Strategy C metrics for a single frame pair given precomputed (u, v).

        Mirrors `FlowVariabilityAnalyzer.metrics_from_flow` so the shared
        Farneback loop in `HeteroresistanceDetector.run` stays symmetric and
        free of inlined per-pair statistics.

        Parameters
        ----------
        u, v        : (H, W) float32 flow components from `_compute_farneback_uv`
        mask_bool   : (H, W) bool    region of interest for divergence stats
        n_mask_px   : optional pre-computed `mask_bool.sum()` (saves a recount)

        Returns
        -------
        dict with keys:
            'divergence_map'    (H, W) float32, NaN outside mask
            'mean_div'          float
            'spatial_cv_div'    float — sigma / |mu| of divergence in the mask
            'hotspot_fraction'  float — fraction of mask pixels above the
                                hotspot threshold
            'hotspot_binary'    (H, W) bool — cached for the persistent
                                accumulator
        """
        dr = self.divergence_from_flow(u, v, mask_bool)
        div_map = dr["divergence"]

        vals = div_map[mask_bool]
        valid = vals[np.isfinite(vals)]

        hotspot_bin = np.zeros(div_map.shape, dtype=bool)
        mean_div = 0.0
        spatial_cv = 0.0
        hotspot_frac = 0.0
        if len(valid) >= 10:
            mu = float(np.nanmean(valid))
            sigma_v = float(np.nanstd(valid))
            mean_div = mu
            spatial_cv = sigma_v / (abs(mu) + 1e-9)
            hotspot_bin = self._detect_hotspots(div_map, mask_bool)
            if n_mask_px is None:
                n_mask_px = int(mask_bool.sum())
            hotspot_frac = float(hotspot_bin.sum()) / max(n_mask_px, 1)

        return {
            "divergence_map":   div_map,
            "mean_div":         mean_div,
            "spatial_cv_div":   spatial_cv,
            "hotspot_fraction": hotspot_frac,
            "hotspot_binary":   hotspot_bin,
        }

    def build_persistent_hotspot_map(
        self,
        divergence_maps: List[np.ndarray],
        masks: List[np.ndarray],
        hotspot_binaries: Optional[List[np.ndarray]] = None,
    ) -> np.ndarray:
        """
        Accumulate hotspot occurrences across all frame pairs.

        The value at each pixel equals the number of frame pairs for which it
        qualified as a hotspot.  Pixels with values >= persistence_frames
        represent persistently expanding regions — candidate heteroresistant
        sub-colonies.

        If `hotspot_binaries` is provided (e.g. cached by the main detector
        run-loop), it is reused directly to avoid recomputing
        `_detect_hotspots` for every frame pair.

        Returns
        -------
        accumulator : (H, W) int32 array
        """
        if not divergence_maps:
            return np.zeros((1, 1), dtype=np.int32)

        accumulator = np.zeros(divergence_maps[0].shape, dtype=np.int32)
        if hotspot_binaries is not None and len(hotspot_binaries) == len(divergence_maps):
            for hotspot in hotspot_binaries:
                accumulator += hotspot.astype(np.int32)
        else:
            for div_map, mask in zip(divergence_maps, masks):
                hotspot = self._detect_hotspots(div_map, mask)
                accumulator += hotspot.astype(np.int32)
        return accumulator

    # ------------------------------------------------------------------
    # Heteroresistance detection
    # ------------------------------------------------------------------

    def detect_heteroresistance(
        self,
        mean_div: np.ndarray,
        spatial_cv_div: np.ndarray,
        global_growth_rates: np.ndarray,
        drug_start_frame: int = 0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Flag frame pairs where the divergence field shows a heteroresistance
        signature.

        Detection condition (both must hold after drug_start_frame):
          1. Global growth is suppressed:  global_growth_rates[t] < threshold
          2. Divergence spatial CV is high: spatial_cv_div[t] > cv_threshold

        A persistence filter requires both conditions to hold for at least
        persistence_frames consecutive frame pairs before a flag is raised.

        Returns
        -------
        flags : (n_pairs,) bool    per-frame-pair binary detection flag
        score : (n_pairs,) float32 normalised divergence CV score in [0, 1]
        """
        n = len(mean_div)
        g = _align_global_rates(global_growth_rates, n)
        drug_tp = max(0, drug_start_frame - 1)
        # Skip the first N frames after drug addition to dodge the transient
        # pressure-drop growth-rate spike documented by Tran et al. (2025).
        skip_until = drug_tp + max(0, config.DRUG_ARTIFACT_FRAMES)

        raw = np.zeros(n, dtype=bool)
        for t in range(n):
            if t < skip_until:
                continue
            global_suppressed = g[t] < self.global_suppression_threshold
            cv_high = spatial_cv_div[t] > self.cv_threshold
            raw[t] = global_suppressed and cv_high

        flags = _persistence_filter(raw, self.persistence_frames)

        # Normalised score: spatial_cv_div linearly rescaled to [0, 1]
        cv_min = float(spatial_cv_div.min())
        cv_max = float(spatial_cv_div.max())
        score = (spatial_cv_div - cv_min) / (cv_max - cv_min + 1e-9)

        return flags, score.astype(np.float32)


# ---------------------------------------------------------------------------
# Combined detector facade
# ---------------------------------------------------------------------------

class HeteroresistanceDetector:
    """
    Drop-in heteroresistance detector combining Strategies B and C.

    Usage
    -----
    detector = HeteroresistanceDetector()
    results  = detector.run(masks, frames, drug_start_frame=30)

    The `results` HeterogeneityTimeSeries object contains both scalar time
    series and spatial heatmaps ready for visualisation.
    """

    def __init__(
        self,
        rolling_window: Optional[int] = None,
        global_suppression_threshold: float = 0.03,
        persistence_frames: int = 4,
        # Strategy C
        cv_threshold_C: Optional[float] = None,
        div_flow_smooth_sigma: Optional[float] = None,
        div_map_smooth_sigma: Optional[float] = None,
    ):
        self.rolling_window = (rolling_window if rolling_window is not None
                               else config.HET_ROLLING_WINDOW)
        self.growth_estimator = GlobalGrowthEstimator(rolling_window=self.rolling_window)
        self.div_analyzer = FlowDivergenceAnalyzer(
            cv_threshold=cv_threshold_C,
            flow_smooth_sigma=div_flow_smooth_sigma,
            div_smooth_sigma=div_map_smooth_sigma,
            persistence_frames=persistence_frames,
            global_suppression_threshold=global_suppression_threshold,
        )
        # The shared-Farneback loop in run() uses DEFAULT_FARNEBACK_PARAMS.
        # If the analyzer was constructed with a custom params dict, the
        # outputs of run() would silently disagree with what the analyzer
        # thinks it uses.  Fail loudly here instead.
        if self.div_analyzer.farneback_params is not DEFAULT_FARNEBACK_PARAMS:
            raise ValueError(
                "HeteroresistanceDetector requires FlowDivergenceAnalyzer to use "
                "DEFAULT_FARNEBACK_PARAMS so the shared Farneback computation "
                "stays consistent.  Construct the analyzer without overriding "
                "farneback_params, or update DEFAULT_FARNEBACK_PARAMS itself."
            )

    def run(
        self,
        masks: List[np.ndarray],
        frames: List[np.ndarray],
        drug_start_frame: int = 0,
    ) -> HeterogeneityTimeSeries:
        """
        Run Strategies B + C and combine results.

        Parameters
        ----------
        masks            : list of segmentation masks (Omnipose output)
        frames           : list of raw phase-contrast frames (same length)
        drug_start_frame : index of the first post-drug frame (0 = no baseline)

        Returns
        -------
        HeterogeneityTimeSeries with all metrics and heatmaps filled in.
        """
        results = HeterogeneityTimeSeries()

        # ── Segmentation QC: detect possible cell clumping ──────────────────
        # Tran et al. (2025): "high cell density data (e.g., clumps or cords)
        # were problematic due to overlapping cells … entire blob of cells was
        # segmented as one."  When the median per-instance area in pre-drug
        # frames exceeds CLUMP_WARN_CELL_AREA_UM2 (a few × the expected single
        # cell), heteroresistance signals downstream may be misleading.
        px_area_um2 = config.PIXEL_SIZE_UM ** 2
        qc_end = drug_start_frame if drug_start_frame > 0 else len(masks)
        per_frame_medians: List[float] = []
        for m in masks[:qc_end]:
            props = regionprops(m.astype(np.int32))
            if props:
                per_frame_medians.append(
                    float(np.median([p.area for p in props])) * px_area_um2
                )
        if per_frame_medians:
            results.baseline_cell_area_um2 = float(np.median(per_frame_medians))
            if results.baseline_cell_area_um2 > config.CLUMP_WARN_CELL_AREA_UM2:
                results.clump_warning = True
                print(
                    f"  ⚠ Possible cell clumping: pre-drug median single-cell "
                    f"area {results.baseline_cell_area_um2:.1f} µm² "
                    f"(expected ~{config.SINGLE_CELL_AREA_UM2:.1f} µm²) — "
                    f"heteroresistance detection may be unreliable"
                )

        print("[HeteroresistanceDetector] Computing global growth rate …")
        global_rates = self.growth_estimator.compute(masks)
        results.global_growth_rates = global_rates

        # ── Build the Strategy-C analysis mask: union of all pre-drug masks ──
        # Fixed spatial ROI for post-drug frames keeps dying cells "in frame"
        # so their negative divergence contrasts with surviving resistant
        # cells — the signal that distinguishes heteroresistance from uniform
        # drug suppression (where the mask simply shrinks).
        if drug_start_frame > 0 and drug_start_frame < len(masks):
            pre_union = (masks[0] > 0).astype(np.int32)
            for m in masks[1:drug_start_frame]:
                pre_union = np.maximum(pre_union, (m > 0).astype(np.int32))
            analysis_masks = list(masks)
            for i in range(drug_start_frame, len(analysis_masks)):
                analysis_masks[i] = pre_union
        else:
            analysis_masks = list(masks)

        # Pre-binarise masks once per sequence (avoids repeated (m > 0) inside
        # the inner loop) and cache per-frame pixel counts.
        c_mask_bools = [(m > 0) for m in analysis_masks]
        c_mask_px    = [int(mb.sum()) for mb in c_mask_bools]

        # ── Single shared loop: Farneback runs once per pair, fed to Strategy C ──
        # An earlier flow-magnitude-CV strategy ("Strategy B") was removed
        # because its spatial CV of |flow| could not discriminate uniform
        # cell death from heteroresistance.  The flow magnitude map is still
        # kept as a diagnostic overlay for the dashboard (it's a free
        # byproduct of the Farneback we already need for divergence).
        print("[HeteroresistanceDetector] Strategy C (flow divergence) …")
        n_pairs = len(frames) - 1
        mag_maps:   List[np.ndarray] = []
        c_div_maps: List[np.ndarray] = []
        c_hot_bins: List[np.ndarray] = []
        c_mean_div = np.zeros(n_pairs, dtype=np.float32)
        c_cv       = np.zeros(n_pairs, dtype=np.float32)
        c_hotspot  = np.zeros(n_pairs, dtype=np.float32)

        for i in range(n_pairs):
            u, v = _compute_farneback_uv(
                frames[i], frames[i + 1], DEFAULT_FARNEBACK_PARAMS
            )

            # Diagnostic |flow| magnitude (free byproduct, used by dashboard)
            mag = np.sqrt(u * u + v * v).astype(np.float32)
            mag *= c_mask_bools[i].astype(np.float32)
            mag_maps.append(mag)

            # Strategy C: divergence variability inside the fixed pre-drug ROI
            cm = self.div_analyzer.compute_pair_metrics(
                u, v, c_mask_bools[i], n_mask_px=c_mask_px[i]
            )
            c_div_maps.append(cm["divergence_map"])
            c_hot_bins.append(cm["hotspot_binary"])
            c_mean_div[i] = cm["mean_div"]
            c_cv[i]       = cm["spatial_cv_div"]
            c_hotspot[i]  = cm["hotspot_fraction"]

        # Strategy C detection
        flags_C, score_C = self.div_analyzer.detect_heteroresistance(
            c_mean_div, c_cv, global_rates,
            drug_start_frame=drug_start_frame,
        )
        results.divergence_cv = c_cv
        results.mean_divergence = c_mean_div
        results.divergence_hotspot_fraction = c_hotspot
        results.detection_flags_C = flags_C

        # Combined score and combined detection are now identical to
        # Strategy C's outputs (no Strategy B to average in).  Kept here
        # for backward compatibility with dashboard + downstream code.
        results.combined_score     = score_C.astype(np.float32)
        results.detection_combined = flags_C.copy()

        # Spatial outputs: use last flagged frame, or last frame if none flagged
        flagged_C = np.where(flags_C)[0]
        chosen_frame = int(flagged_C[-1]) if len(flagged_C) > 0 \
            else n_pairs - 1
        results.flow_magnitude_map = mag_maps[chosen_frame]
        results.divergence_map     = c_div_maps[chosen_frame]
        results.persistent_hotspot_map = self.div_analyzer.build_persistent_hotspot_map(
            c_div_maps, analysis_masks, hotspot_binaries=c_hot_bins,
        )

        return results


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_heteroresistance_dashboard(
    results: HeterogeneityTimeSeries,
    raw_frame: np.ndarray,
    mask: np.ndarray,
    drug_start_frame: int = 0,
    interval_minutes: float = 2.0,
    rolling_window: Optional[int] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Seven-panel dashboard (3 rows × 3 cols, row 2 spans all cols) for a
    single microchamber position:

      Row 0 — spatial overlays
        [0,0] Raw phase image overlaid with flow magnitude  (Strategy B)
        [0,1] Raw phase image overlaid with flow divergence (Strategy C)
        [0,2] Persistent expansion hotspot accumulation map (Strategy C)

      Row 1 — scalar time series
        [1,0] Global growth rate vs time (with drug-addition line)
        [1,1] Flow spatial CV + hotspot fraction            (Strategy B)
        [1,2] Divergence spatial CV + mean divergence       (Strategy C)

      Row 2 — full-width combined score with detection flags

    Parameters
    ----------
    results          : output of HeteroresistanceDetector.run()
    raw_frame        : a representative raw phase image (numpy array)
    mask             : corresponding segmentation mask
    drug_start_frame : index in the original frame sequence
    """
    if rolling_window is None:
        rolling_window = config.HET_ROLLING_WINDOW
    fig = plt.figure(figsize=(18, 13), facecolor="#0d0d0d")
    fig.suptitle(
        "Heteroresistance Detection Dashboard",
        fontsize=16,
        fontweight="bold",
        color="white",
        y=0.98,
    )
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.3)

    def _ax(r, c):
        ax = fig.add_subplot(gs[r, c])
        ax.set_facecolor("#1a1a1a")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")
        ax.tick_params(colors="#aaa")
        ax.xaxis.label.set_color("#aaa")
        ax.yaxis.label.set_color("#aaa")
        ax.title.set_color("white")
        return ax

    # --- time axis ---
    def _time(n):
        return (np.arange(n) - (drug_start_frame - rolling_window)) * interval_minutes / 60.0

    # Panel 0,0: flow magnitude overlay
    ax00 = _ax(0, 0)
    if raw_frame is not None:
        ax00.imshow(raw_frame, cmap="gray", interpolation="nearest")
    if results.flow_magnitude_map is not None:
        ax00.imshow(results.flow_magnitude_map, cmap="inferno", alpha=0.65,
                    vmin=0, vmax=float(np.percentile(results.flow_magnitude_map, 99)))
    ax00.set_title("Dense Flow Magnitude  (Strategy B)", fontsize=11)
    ax00.axis("off")

    # Panel 0,1: divergence map overlay on raw phase image
    ax01 = _ax(0, 1)
    if raw_frame is not None:
        ax01.imshow(raw_frame, cmap="gray", interpolation="nearest")
    if results.divergence_map is not None:
        # Diverging colormap: blue = contraction/death, white = zero, red = expansion/growth
        div_abs = float(np.nanpercentile(np.abs(results.divergence_map[np.isfinite(results.divergence_map)]), 95)) \
            if np.any(np.isfinite(results.divergence_map)) else 1.0
        ax01.imshow(
            results.divergence_map,
            cmap="RdBu_r",
            alpha=0.65,
            vmin=-div_abs,
            vmax=div_abs,
        )
    ax01.set_title("Flow Divergence  (red = growth, blue = death)", fontsize=10)
    ax01.axis("off")

    # Panel 0,2: persistent hotspot accumulation map
    ax02 = _ax(0, 2)
    if raw_frame is not None:
        ax02.imshow(raw_frame, cmap="gray", interpolation="nearest")
    if results.persistent_hotspot_map is not None:
        pmap = results.persistent_hotspot_map.astype(np.float32)
        if pmap.max() > 0:
            ax02.imshow(
                np.ma.masked_where(pmap == 0, pmap),
                cmap="hot",
                alpha=0.7,
                vmin=0,
                vmax=pmap.max(),
            )
    ax02.set_title("Persistent Expansion Hotspots  (brighter = longer)", fontsize=10)
    ax02.axis("off")

    # Panel 1,0: global growth rate
    ax10 = _ax(1, 0)
    if len(results.global_growth_rates) > 0:
        t_g = _time(len(results.global_growth_rates))
        ax10.plot(t_g, results.global_growth_rates, color="#009ADE", lw=2)
        ax10.axhline(0, color="#555", lw=1, linestyle="--")
        ax10.axvline(0, color="#FF1F5B", lw=1.5, linestyle="--", label="Drug")
        ax10.set_xlabel("Time (h)")
        ax10.set_ylabel("Growth rate (h⁻¹)")
        ax10.set_title("Global Growth Rate", fontsize=11)
        ax10.legend(fontsize=9, labelcolor="#aaa", facecolor="#1a1a1a", edgecolor="#444")
        ax10.grid(alpha=0.2)

    # Panel 1,1: Mean divergence over time (signed — positive = expansion,
    # negative = contraction).  An earlier flow-magnitude-CV strategy was
    # removed because it couldn't distinguish heteroresistance from
    # heterogeneous cell death; mean divergence is a sharper signal: net
    # positive after drug indicates growth survived somewhere.
    ax11 = _ax(1, 1)
    if len(results.mean_divergence) > 0:
        t_md = _time(len(results.mean_divergence))
        ax11.plot(t_md, results.mean_divergence, color="#E9C46A", lw=2,
                  label="Mean divergence")
        ax11.axhline(0, color="#555", lw=1, linestyle="--")
        ax11.fill_between(t_md, results.mean_divergence, 0,
                          where=(results.mean_divergence >= 0),
                          color="#E9C46A", alpha=0.25, interpolate=True)
        ax11.fill_between(t_md, results.mean_divergence, 0,
                          where=(results.mean_divergence < 0),
                          color="#4A90E2", alpha=0.25, interpolate=True)
        ax11.axvline(0, color="#FF1F5B", lw=1.5, linestyle="--", label="Drug")
        ax11.set_xlabel("Time (h)")
        ax11.set_ylabel("Mean divergence  (+ growth / − death)")
        ax11.set_title("Net Expansion vs Contraction", fontsize=11)
        ax11.legend(fontsize=9, labelcolor="#aaa", facecolor="#1a1a1a", edgecolor="#444")
        ax11.grid(alpha=0.2)

    # Panel 1,2: Strategy C scalar time series — divergence spatial CV
    ax12 = _ax(1, 2)
    if len(results.divergence_cv) > 0:
        t_c_ts = _time(len(results.divergence_cv))
        ax12.plot(t_c_ts, results.divergence_cv, color="#A8DADC", lw=2,
                  label="Divergence spatial CV")
        ax12b = ax12.twinx()
        ax12b.set_facecolor("#1a1a1a")
        ax12b.plot(t_c_ts, results.mean_divergence, color="#E9C46A", lw=1.5,
                   linestyle=":", label="Mean divergence")
        ax12b.set_ylabel("Mean divergence", color="#E9C46A", fontsize=9)
        ax12b.tick_params(colors="#E9C46A")
        for t, flag in enumerate(results.detection_flags_C):
            if flag and t < len(t_c_ts):
                ax12.axvspan(t_c_ts[t], t_c_ts[min(t + 1, len(t_c_ts) - 1)],
                             alpha=0.15, color="#A8DADC")
        ax12.axvline(0, color="#FF1F5B", lw=1.5, linestyle="--")
        ax12.set_xlabel("Time (h)")
        ax12.set_ylabel("Divergence CV", color="#A8DADC")
        ax12.tick_params(axis="y", colors="#A8DADC")
        ax12.set_title("Strategy C: Divergence Variability", fontsize=11)
        lines1, labels1 = ax12.get_legend_handles_labels()
        lines2, labels2 = ax12b.get_legend_handles_labels()
        ax12.legend(lines1 + lines2, labels1 + labels2,
                    fontsize=9, labelcolor="#aaa", facecolor="#1a1a1a", edgecolor="#444")
        ax12.grid(alpha=0.2)

    # ------------------------------------------------------------------
    # Row 2: combined score — full width
    # ------------------------------------------------------------------

    ax2 = fig.add_subplot(gs[2, :])
    ax2.set_facecolor("#1a1a1a")
    for spine in ax2.spines.values():
        spine.set_edgecolor("#444")
    ax2.tick_params(colors="#aaa")
    ax2.xaxis.label.set_color("#aaa")
    ax2.yaxis.label.set_color("#aaa")
    ax2.title.set_color("white")

    if len(results.combined_score) > 0:
        t_c = _time(len(results.combined_score))
        ax2.plot(t_c, results.combined_score, color="#FFD700", lw=2.5, label="Combined score")
        ax2.fill_between(t_c, results.combined_score, alpha=0.15, color="#FFD700")
        for t, flag in enumerate(results.detection_combined):
            if flag and t < len(t_c):
                ax2.axvspan(t_c[t], t_c[min(t + 1, len(t_c) - 1)],
                             alpha=0.2, color="#FF1F5B")
        # first detection
        first = np.where(results.detection_combined)[0]
        if len(first) > 0:
            ft = t_c[first[0]]
            ax2.axvline(ft, color="#FF1F5B", lw=2, linestyle="-.",
                        label=f"First detection: {ft:.2f}h")
        ax2.axvline(0, color="#FF1F5B", lw=1.5, linestyle="--", label="Drug addition")
        ax2.set_xlabel("Time (h)")
        ax2.set_ylabel("Heteroresistance Score (0–1)")
        ax2.set_title("Combined Heteroresistance Score  |  red shading = detected", fontsize=12)
        ax2.set_ylim(-0.05, 1.05)
        ax2.legend(fontsize=10, labelcolor="#aaa", facecolor="#1a1a1a", edgecolor="#444")
        ax2.grid(alpha=0.2)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"  Dashboard saved → {save_path}")

    return fig


# ---------------------------------------------------------------------------
# Population-level heteroresistance score
# ---------------------------------------------------------------------------

def _post_drug_curve(areas: np.ndarray, drug_idx: int) -> Optional[np.ndarray]:
    """Self-normalise an area trace to its pre-drug mean and return the
    post-drug portion (log-ratio).  Returns None when there is too little
    pre- or post-drug data to be meaningful.
    """
    if drug_idx < 3 or drug_idx >= len(areas) - 3:
        return None
    pre = areas[:drug_idx]
    post = areas[drug_idx:]
    pre_mean = float(np.mean(pre))
    if pre_mean <= 0 or not np.isfinite(pre_mean):
        return None
    # log-ratio is more linear under exponential growth/decay and avoids
    # blow-up when the post mean approaches zero
    return np.log(np.clip(post / pre_mean, 1e-6, None)).astype(np.float32)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity on equal-length 1-D arrays.  0 if either has zero norm."""
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def score_heteroresistance_population(
    test_chambers: List[Dict],
    control_chambers: List[Dict],
    peer_chambers: List[Dict],
    similarity_threshold: float = 0.0,
) -> List[Dict]:
    """Population-level heteroresistance scoring.

    Automates the manual visual step described in Tran et al. (2025) for
    heteroresistance assays:

        "Microchambers with growth patterns matching untreated reference
         (rather than drug-treated controls) flagged as candidates for
         resistant cells; movies manually reviewed to confirm."

    For each chamber in `test_chambers`, computes how much its post-drug
    self-normalised area curve resembles the control-group mean curve
    versus the peer-treatment-group mean curve.  Positive score → looks
    like control → heteroresistance candidate.

    Parameters
    ----------
    test_chambers     : list of dicts, each with keys 'areas' (np.ndarray)
                        and 'ds_idx' (int, drug-start index in that chamber).
                        These are the chambers being evaluated for heteroresistance.
    control_chambers  : same shape; the untreated reference group.
    peer_chambers     : same shape; treated chambers expected to be fully
                        susceptible (the "what susceptible kill looks like"
                        anchor).
    similarity_threshold : score above this flags the chamber as a candidate.
                        0.0 means "more like control than like peer-treated".

    Returns
    -------
    list of dicts (one per test chamber, same order), each with keys:
        'sim_control'     : float in [-1, 1]
        'sim_peer'        : float in [-1, 1]
        'pop_score'       : sim_control - sim_peer
        'pop_candidate'   : bool — pop_score > similarity_threshold
        'curve'           : post-drug log-ratio curve used for the score
                            (None if the chamber had too little data)
    """
    def _curve(chamber):
        return _post_drug_curve(np.asarray(chamber["areas"]),
                                int(chamber["ds_idx"]))

    ctrl_curves = [c for c in (_curve(ch) for ch in control_chambers) if c is not None]
    peer_curves = [c for c in (_curve(ch) for ch in peer_chambers)    if c is not None]
    if not ctrl_curves or not peer_curves:
        return [{
            "sim_control": float("nan"),
            "sim_peer":    float("nan"),
            "pop_score":   float("nan"),
            "pop_candidate": False,
            "curve":       None,
        } for _ in test_chambers]

    # Truncate every curve to the shortest length so means align
    min_len = min(
        min(len(c) for c in ctrl_curves),
        min(len(c) for c in peer_curves),
    )

    ctrl_stack = np.vstack([c[:min_len] for c in ctrl_curves])
    peer_stack = np.vstack([c[:min_len] for c in peer_curves])
    ctrl_mean = ctrl_stack.mean(axis=0)
    peer_mean = peer_stack.mean(axis=0)

    out: List[Dict] = []
    for ch in test_chambers:
        curve = _curve(ch)
        if curve is None or len(curve) < min_len:
            out.append({
                "sim_control": float("nan"),
                "sim_peer":    float("nan"),
                "pop_score":   float("nan"),
                "pop_candidate": False,
                "curve":       curve,
            })
            continue
        c = curve[:min_len]
        s_ctrl = _cosine_similarity(c, ctrl_mean)
        s_peer = _cosine_similarity(c, peer_mean)
        score = s_ctrl - s_peer
        out.append({
            "sim_control":   round(s_ctrl, 3),
            "sim_peer":      round(s_peer, 3),
            "pop_score":     round(score,  3),
            "pop_candidate": bool(score > similarity_threshold),
            "curve":         c,
        })
    return out


def loo_population_score(
    chambers: List[Dict],
    ctrl_group: str = "A_REF",
    peer_group: str = "B_RIF10",
    similarity_threshold: float = 0.0,
) -> List[Dict]:
    """Leave-one-out variant of :func:`score_heteroresistance_population`.

    For each chamber, builds the control/peer reference pools from *every
    other* chamber in the matching group and computes pop_score against
    those leave-one-out means.  Strictly more rigorous than the fixed-pool
    variant because:

    * An A_REF chamber's own curve never contributes to the ctrl mean
      against which it is scored — so an outlier A_REF can't artificially
      drag the mean toward itself and hide.
    * Same protection for B_RIF10 chambers vs the peer mean.
    * C_HETERO chambers (or any group not equal to ctrl/peer) use the full
      pools, equivalent to the non-LOO version.

    Returns one dict per chamber (same order), each with the extra keys
    ``pop_score_loo``, ``pop_candidate_loo``, ``sim_control_loo``,
    ``sim_peer_loo``.  Chambers with insufficient pre/post-drug data, or
    in groups where the LOO pool becomes empty, return NaN.
    """
    # Pre-compute each chamber's post-drug log-ratio curve once.
    curves: List[Tuple[Optional[np.ndarray], str]] = []
    for ch in chambers:
        c = _post_drug_curve(np.asarray(ch["areas"]), int(ch["ds_idx"]))
        curves.append((c, ch.get("group", "")))

    valid_lengths = [len(c) for c, _ in curves if c is not None]
    if not valid_lengths:
        return [{
            "position_id":       ch.get("position_id", ""),
            "group":             ch.get("group", ""),
            "sim_control_loo":   float("nan"),
            "sim_peer_loo":      float("nan"),
            "pop_score_loo":     float("nan"),
            "pop_candidate_loo": False,
        } for ch in chambers]
    min_len = min(valid_lengths)

    # Precompute truncated curves keyed by index so the LOO inner loop is cheap.
    trimmed = [(c[:min_len] if c is not None else None, g) for c, g in curves]

    out: List[Dict] = []
    for i, ch in enumerate(chambers):
        c_i, g_i = trimmed[i]
        if c_i is None:
            out.append({
                "position_id":       ch.get("position_id", ""),
                "group":             g_i,
                "sim_control_loo":   float("nan"),
                "sim_peer_loo":      float("nan"),
                "pop_score_loo":     float("nan"),
                "pop_candidate_loo": False,
            })
            continue

        # Build leave-one-out pools: every other chamber in matching group
        ctrl_pool = [c for j, (c, g) in enumerate(trimmed)
                     if c is not None and g == ctrl_group and j != i]
        peer_pool = [c for j, (c, g) in enumerate(trimmed)
                     if c is not None and g == peer_group and j != i]

        if not ctrl_pool or not peer_pool:
            out.append({
                "position_id":       ch.get("position_id", ""),
                "group":             g_i,
                "sim_control_loo":   float("nan"),
                "sim_peer_loo":      float("nan"),
                "pop_score_loo":     float("nan"),
                "pop_candidate_loo": False,
            })
            continue

        ctrl_mean = np.mean(np.vstack(ctrl_pool), axis=0)
        peer_mean = np.mean(np.vstack(peer_pool), axis=0)
        sim_ctrl = _cosine_similarity(c_i, ctrl_mean)
        sim_peer = _cosine_similarity(c_i, peer_mean)
        score = sim_ctrl - sim_peer
        out.append({
            "position_id":       ch.get("position_id", ""),
            "group":             g_i,
            "sim_control_loo":   round(sim_ctrl, 3),
            "sim_peer_loo":      round(sim_peer, 3),
            "pop_score_loo":     round(score, 3),
            "pop_candidate_loo": bool(score > similarity_threshold),
        })
    return out

