"""
Configuration file for Hybrid CV Pipeline
All universal parameters and paths in one place
"""

import os

# ============================================================================
# DATA PATHS
# ============================================================================

# Base directory (update this to your data location)
BASE_DIR = "./data"

# Raw data directories
REF_RAW_DIR = os.path.join(BASE_DIR, "REF_raw_data101_110")
REF_MASK_DIR = os.path.join(BASE_DIR, "REF_masks101_110")
RIF10_RAW_DIR = os.path.join(BASE_DIR, "RIF10_raw_data201_210")
RIF10_MASK_DIR = os.path.join(BASE_DIR, "RIF10_masks201_210")

# Output directories
OUTPUT_DIR = os.path.join("results", "full_hybrid_output")
SAMPLE_OUTPUT_DIR = os.path.join("results", "sample_analysis_output")


def ensure_output_dirs() -> None:
    """Create the output directories on first use.

    Call this from script entry points (e.g. analysis.main_analysis,
    sample_position.single_position) so that simply importing config does
    not create directories as a side effect.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(SAMPLE_OUTPUT_DIR, exist_ok=True)

# ============================================================================
# EXPERIMENT PARAMETERS
# ============================================================================

# Time-lapse parameters
INTERVAL_MINUTES = 2.0  # Time between frames (minutes)
PIXEL_SIZE_UM = 0.0733  # Pixel size at 150x magnification (μm/pixel)

# Outlier replacement threshold for GrowthAnalyzer.smooth_areas.
# Per Tran et al. (2025): "measurements that deviated more than 5% from the
# curve fitted in the sliding window with the fitted value".
AREA_OUTLIER_THRESHOLD = 0.05

# Analysis parameters
#
# Two rolling windows are used in this project for distinct purposes:
#
#   ROLLING_WINDOW (area-based growth analysis, analysis.py / GrowthAnalyzer)
#     Smoother window for the Tran et al. (2025) log-linear area fit.
#     24 frames * 2 min = 48 min — chosen for stability of the population
#     growth-rate baseline used in time-to-detection statistics.
#
#   HET_ROLLING_WINDOW (heteroresistance detector, heteroresistance_detector.py)
#     Tighter window for the global-suppression gate in Strategies B and C.
#     16 frames * 2 min = 32 min — responsive enough that the detector
#     reacts to drug action within ~30 min after addition.
ROLLING_WINDOW     = 24
HET_ROLLING_WINDOW = 16


# ----------------------------------------------------------------------------
# Organism profiles
# ----------------------------------------------------------------------------
# Tran et al. (2025) image M. bovis BCG every 10 min (slow grower) and
# M. smegmatis every 2 min (fast grower).  Each organism needs its own
# rolling-window length so that ~30-180 min of biological signal is fitted
# regardless of frame interval.
#
# Use `apply_organism_profile("M_bovis_BCG")` at the entry point of any
# script that processes BCG data; the function rewrites INTERVAL_MINUTES,
# ROLLING_WINDOW and HET_ROLLING_WINDOW at module level so downstream code
# picks up the new values transparently.

ORGANISM_PROFILES = {
    "M_smegmatis": {
        "INTERVAL_MINUTES":   2.0,
        "ROLLING_WINDOW":     24,   # 48 min — slightly above paper's 30 min
        "HET_ROLLING_WINDOW": 16,   # 32 min — detector responsiveness
    },
    "M_bovis_BCG": {
        "INTERVAL_MINUTES":   10.0,
        "ROLLING_WINDOW":     18,   # 180 min  = paper's 3 h sliding window
        "HET_ROLLING_WINDOW": 12,   # 120 min  — proportionally tighter
    },
}


def apply_organism_profile(name: str) -> None:
    """Rewrite the time-resolution constants for a known organism.

    Raises KeyError if the profile name is unknown.
    """
    global INTERVAL_MINUTES, ROLLING_WINDOW, HET_ROLLING_WINDOW
    p = ORGANISM_PROFILES[name]
    INTERVAL_MINUTES   = p["INTERVAL_MINUTES"]
    ROLLING_WINDOW     = p["ROLLING_WINDOW"]
    HET_ROLLING_WINDOW = p["HET_ROLLING_WINDOW"]
    print(f"[config] Applied profile '{name}': "
          f"INTERVAL={INTERVAL_MINUTES}min, "
          f"ROLLING_WINDOW={ROLLING_WINDOW}, "
          f"HET_ROLLING_WINDOW={HET_ROLLING_WINDOW}")


# Position ranges
REF_POSITIONS = list(range(101, 111))  # Reference positions (101-110)
RIF10_POSITIONS = list(range(201, 221))  # Treatment positions (201-220)


# ============================================================================
# PIPELINE PARAMETERS
# ============================================================================

# Segmentation pipeline
GAUSSIAN_SIGMA = 1.0  # Standard deviation for Gaussian blur
SOBEL_KSIZE = 3  # Kernel size for Sobel edge detection (must be odd)
USE_MEMORY = True  # Whether to use continuous memory mask

# Flow divergence analysis parameters (Strategy C)
# Pre-differentiation Gaussian smoothing applied to the u and v flow components
# before computing numerical gradients.  Larger values reduce high-frequency
# noise at the cost of spatial resolution.
DIV_FLOW_SMOOTH_SIGMA = 2.0
# Post-computation Gaussian smoothing applied to the raw divergence map.
DIV_MAP_SMOOTH_SIGMA = 1.5
# Hotspot threshold multiplier k: a pixel is a hotspot when
#   divergence > mean + k * std   (computed within the segmentation mask).
DIV_HOTSPOT_SIGMA_THRESH = 1.5
# Minimum connected-component area (pixels) for a hotspot to be retained.
# Smaller regions are treated as segmentation noise and discarded.
DIV_HOTSPOT_MIN_AREA_PX = 50
# A hotspot must persist for at least this many consecutive frame-pairs to be
# counted in the persistent hotspot accumulation map.
DIV_PERSISTENCE_FRAMES = 4
# Spatial CV threshold for the binary heteroresistance detection flag.
# Frames where CV(divergence) exceeds this value are flagged as suspicious.
DIV_CV_THRESHOLD = 0.8

# Frames to skip immediately after drug addition.
# Tran et al. (2025) report a transient growth-rate spike right after media
# change, caused by a pressure drop in the perfusion line.  The Strategy B/C
# raw-detection loops skip the first N frame-pairs post drug_start_frame so
# this artifact is not misread as biological signal.  Set to 0 to reproduce
# the pre-2026-05 behaviour exactly.
DRUG_ARTIFACT_FRAMES = 3

# ----------------------------------------------------------------------------
# Segmentation quality-control thresholds
# ----------------------------------------------------------------------------
# Typical single-cell area for M. smegmatis / M. bovis BCG bacilli in
# microchamber phase-contrast at our pixel scale (~3-6 µm²).  When Omnipose
# fails on dense colonies / cords, multiple cells get merged into a single
# "cell" — driving the median single-cell area well above this.  The
# heteroresistance detector raises `results.clump_warning = True` and prints
# a warning when the pre-drug median exceeds CLUMP_WARN_CELL_AREA_UM2 so
# downstream consumers can flag the run as low-confidence.
SINGLE_CELL_AREA_UM2     = 5.0
CLUMP_WARN_CELL_AREA_UM2 = 25.0

# Statistical parameters
DIVERGENCE_ALPHA = 0.05  # Significance level for t-test
DIVERGENCE_MIN_CONSECUTIVE = 3  # Minimum consecutive significant frames for TTD


# ============================================================================
# FILE NAMING PATTERNS
# ============================================================================

RAW_IMAGE_EXTENSIONS = ['.tiff', '.tif']
MASK_PREFIX = 'MASK_'
MASK_EXTENSIONS = ['.tiff', '.tif']


# ============================================================================
# VISUALIZATION PARAMETERS
# ============================================================================

# Color schemes
COLOR_REF = '#009ADE'  # Blue for reference
COLOR_TREATMENT = '#FF1F5B'  # Red/pink for treatment
COLOR_TTD = 'orange'  # Orange for time-to-detection markers

# Figure sizes
FIGSIZE_SINGLE = (15, 4)
FIGSIZE_COMPARISON = (12, 5)
FIGSIZE_COMPREHENSIVE = (14, 10)
FIGSIZE_VIEWER = (24, 14)

# DPI for saved figures
DPI = 150


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_position_paths(position_id, is_treatment=False):
    """
    Get raw and mask directory paths for a given position
    
    Args:
        position_id: Position number (e.g., 101, 201)
        is_treatment: Whether this is a treatment position
        
    Returns:
        tuple: (raw_dir, mask_dir)
    """
    if is_treatment:
        raw_base = RIF10_RAW_DIR
        mask_base = RIF10_MASK_DIR
    else:
        raw_base = REF_RAW_DIR
        mask_base = REF_MASK_DIR
    
    raw_dir = os.path.join(raw_base, f"Pos{position_id}", "aphase")
    mask_dir = os.path.join(mask_base, f"Pos{position_id}", "PreprocessedPhaseMasks")
    
    return raw_dir, mask_dir


def get_time_array(n_frames, start_at_zero=False):
    """
    Generate time array in hours
    
    Args:
        n_frames: Number of frames
        start_at_zero: If True, time starts at 0; if False, starts at -1
        
    Returns:
        numpy array: Time in hours
    """
    import numpy as np
    time_hours = np.arange(n_frames) * INTERVAL_MINUTES / 60
    if not start_at_zero:
        time_hours -= 1.0  # Pre-drug baseline starts at -1h
    return time_hours


def print_config():
    """Print current configuration"""
    print("\n" + "="*70)
    print("HYBRID CV PIPELINE CONFIGURATION")
    print("="*70)
    print(f"\nData Paths:")
    print(f"  Base directory: {BASE_DIR}")
    print(f"  REF raw:        {REF_RAW_DIR}")
    print(f"  REF masks:      {REF_MASK_DIR}")
    print(f"  RIF10 raw:      {RIF10_RAW_DIR}")
    print(f"  RIF10 masks:    {RIF10_MASK_DIR}")
    print(f"  Output:         {OUTPUT_DIR}")
    
    print(f"\nExperiment Parameters:")
    print(f"  Frame interval:  {INTERVAL_MINUTES} minutes")
    print(f"  Pixel size:      {PIXEL_SIZE_UM} μm/pixel")
    print(f"  Rolling window:  {ROLLING_WINDOW} frames (~{ROLLING_WINDOW * INTERVAL_MINUTES} min)")
    
    print(f"\nPipeline Parameters:")
    print(f"  Gaussian σ:      {GAUSSIAN_SIGMA}")
    print(f"  Sobel kernel:    {SOBEL_KSIZE}")
    print(f"  Use memory:      {USE_MEMORY}")

    print(f"\nStatistical:")
    print(f"  Alpha:           {DIVERGENCE_ALPHA}")
    print(f"  Min consecutive: {DIVERGENCE_MIN_CONSECUTIVE}")
    
    print("="*70)


if __name__ == "__main__":
    print_config()
