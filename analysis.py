"""
Analysis Notebook: Hybrid CV Pipeline for M. smegmatis NCTC 8159
Applies the hybrid pipeline to REF (untreated) and RIF10 (treated) datasets
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import time
from scipy import stats
from typing import Dict, List, Optional, Tuple

# Import the hybrid pipeline (assuming it's saved as hybrid_pipeline.py)
from hybrid_pipeline import (
    HybridSegmentationPipeline,
    GrowthAnalyzer,
    plot_growth_comparison,
)

from heteroresistance_detector import HeteroresistanceDetector, plot_heteroresistance_dashboard

import config

import skimage.io
from sklearn.metrics import jaccard_score

# Create output directory on script start
config.ensure_output_dirs()


# HELPER FUNCTIONS

def process_single_position(raw_pos_dir: str,
                           mask_pos_dir: str,
                           output_subdir: str,
                           use_hybrid: bool = True) -> Dict:
    """
    Process a single microchamber position.
    
    Args:
        raw_pos_dir: Directory containing aphase folder with raw images
        mask_pos_dir: Directory containing PreprocessedPhaseMasks with masks
        output_subdir: Subdirectory for outputs
        use_hybrid: Use hybrid pipeline (True) or baseline Omnipose (False)
        
    Returns:
        results: Dictionary with processed data
    """
    # Actual folder structure: Pos101/aphase/ and Pos101/PreprocessedPhaseMasks/
    raw_dir = os.path.join(raw_pos_dir, "aphase")
    mask_dir = os.path.join(mask_pos_dir, "PreprocessedPhaseMasks")
    
    output_dir = os.path.join(output_subdir, os.path.basename(raw_pos_dir))
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if directories exist
    if not os.path.exists(raw_dir):
        print(f"  ⚠️  Raw directory not found: {raw_dir}")
        return None
    if not os.path.exists(mask_dir):
        print(f"  ⚠️  Mask directory not found: {mask_dir}")
        return None
    
    # Load frames and masks
    raw_files = sorted([os.path.join(raw_dir, f) for f in os.listdir(raw_dir)
                       if f.endswith('.tiff') or f.endswith('.tif')])
    mask_files = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir)
                        if f.startswith('MASK_') and (f.endswith('.tiff') or f.endswith('.tif'))])
    
    if len(raw_files) == 0:
        print(f"  ⚠️  No raw images found in {raw_dir}")
        return None
    if len(mask_files) == 0:
        print(f"  ⚠️  No mask files found in {mask_dir}")
        return None
    
    frames = [skimage.io.imread(f) for f in raw_files]
    omnipose_masks = [skimage.io.imread(f) for f in mask_files]
    
    # Ensure same number of frames and masks
    min_len = min(len(frames), len(omnipose_masks))
    frames = frames[:min_len]
    omnipose_masks = omnipose_masks[:min_len]
    
    # Initialize components
    seg_pipeline = HybridSegmentationPipeline(gaussian_sigma=config.GAUSSIAN_SIGMA)
    growth_analyzer = GrowthAnalyzer(rolling_window=config.ROLLING_WINDOW,
                                    interval_minutes=config.INTERVAL_MINUTES,
                                    pixel_size_um=config.PIXEL_SIZE_UM)

    # Process based on method
    if use_hybrid:
        refined_masks, edges = seg_pipeline.process_sequence(
            frames, omnipose_masks, use_memory=config.USE_MEMORY)
        masks_to_use = refined_masks
    else:
        # Baseline: Omnipose only
        masks_to_use = omnipose_masks

    # Compute area-based metrics
    areas = growth_analyzer.compute_area_growth(masks_to_use)

    # Smooth areas to reduce segmentation noise (Tran et al. 2025 method)
    areas_smoothed = growth_analyzer.smooth_areas(areas, window=8)

    # Compute growth rates from smoothed areas
    growth_rates = growth_analyzer.compute_growth_rate_rolling(areas_smoothed)

    results = {
        'areas': areas,
        'growth_rates': growth_rates,
        'masks': masks_to_use
    }
    
    # Save
    with open(os.path.join(output_dir, 'results.pickle'), 'wb') as f:
        pickle.dump(results, f)
    
    return results


def aggregate_positions(position_results: List[Dict],
                       metric: str = 'areas') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Aggregate results across multiple positions.
    
    Args:
        position_results: List of result dictionaries
        metric: Which metric to aggregate ('areas', 'growth_rates')
        
    Returns:
        mean, std, sem: Aggregated statistics
    """
    data = [r[metric] for r in position_results if len(r[metric]) > 0]
    
    # Handle different lengths by truncating to minimum
    min_len = min(len(d) for d in data)
    data_truncated = [d[:min_len] for d in data]
    
    stacked = np.vstack(data_truncated)
    mean = np.mean(stacked, axis=0)
    std = np.std(stacked, axis=0, ddof=1)
    sem = stats.sem(stacked, axis=0, ddof=1)
    
    return mean, std, sem


def plot_normalized_growth(time_hours: np.ndarray,
                          normalized_rates: np.ndarray,
                          save_path: str = None):
    """
    Plot normalized growth rates (treatment/reference ratio).
    Shows relative change over time.
    Based on the reference implementation (Tran et al. 2025).
    
    Args:
        time_hours: Time array in hours
        normalized_rates: Treatment/reference ratio (1.0 = no change)
        save_path: Optional path to save figure
    """
    plt.figure(figsize=(10, 6), facecolor='white')
    
    # Smooth normalized rates to reduce volatility
    window = 5
    rw = config.ROLLING_WINDOW
    if len(normalized_rates) >= window:
        smoothed = np.convolve(normalized_rates, np.ones(window)/window, mode='valid')
        # Adjust time to match smoothed data (growth_rates start at frame rw)
        time_plot = time_hours[rw:rw + len(smoothed)]
    else:
        smoothed = normalized_rates
        time_plot = time_hours[rw:rw + len(normalized_rates)]
    
    # Add drug addition line
    plt.axvline(x=0, color='#FF1F5B', linestyle='--', lw=2, label='Drug addition')
    
    # Add reference baseline (ratio = 1.0)
    plt.axhline(y=1.0, color='gray', linestyle=':', lw=2, alpha=0.7, label='Reference baseline')
    
    # Plot smoothed normalized growth rate
    plt.plot(time_plot, smoothed, lw=3, color='#AF58BA', label='Normalized (Treatment/Reference, smoothed)')
    
    # Auto-scale y-axis based on data percentiles (ignore outliers)
    p05, p95 = np.percentile(smoothed, [5, 95])
    y_margin = (p95 - p05) * 0.2
    y_min = max(0, p05 - y_margin)
    y_max = min(2.0, p95 + y_margin)
    
    plt.xlabel('Time (hours)', fontsize=12)
    plt.ylabel('Normalized Growth Rate (Treatment / Reference)', fontsize=12)
    plt.title('Normalized Growth Rate: RIF10 / REF (5-frame moving average)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.ylim([y_min, y_max])
    
    if save_path:
        plt.savefig(save_path, dpi=config.DPI, bbox_inches='tight')
    plt.close()


def bootstrap_ci(data: np.ndarray, n_bootstrap: int = 1000, ci_level: float = 0.95) -> Tuple[float, float]:
    """
    Calculate bootstrap confidence interval.
    
    Args:
        data: 1D array of values
        n_bootstrap: Number of bootstrap samples
        ci_level: Confidence level (default 95%)
        
    Returns:
        ci_lower, ci_upper: Confidence interval bounds
    """
    bootstrap_means = []
    n = len(data)
    
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        bootstrap_means.append(np.median(sample))
    
    alpha = (1 - ci_level) / 2
    ci_lower = np.percentile(bootstrap_means, alpha * 100)
    ci_upper = np.percentile(bootstrap_means, (1 - alpha) * 100)
    
    return ci_lower, ci_upper


def detect_separation_sem_overlap(
    ref_data_list: List[np.ndarray],
    treat_data_list: List[np.ndarray],
    time_hours: np.ndarray,
    min_consecutive: int = 3,
) -> Optional[float]:
    """
    Paper-faithful TTD: first time the normalized ±1·SEM bands of the two
    populations stop overlapping.

    Reproduces the criterion described in Tran et al. (2025):
        "the separation time of the treatment population from the reference
         population was based on the time at the separation of the normalized
         SEM values between the two populations".

    Provides a comparison baseline alongside `compute_ttd_statistics` (which
    uses bootstrap CIs + per-timepoint t-tests, i.e. a stricter criterion).

    Parameters
    ----------
    ref_data_list   : list of (T,) arrays — one signal per reference position
    treat_data_list : list of (T,) arrays — one signal per treatment position
    time_hours      : (T,) time axis in hours
    min_consecutive : require the non-overlap to hold for >= this many
                      consecutive timepoints before declaring separation

    Returns
    -------
    t_sep_hours : float or None
        The first hour where the SEM bands separate (paper's TTD), or None
        if the bands never separate over the requested window.
    """
    min_len = min(
        min(len(d) for d in ref_data_list),
        min(len(d) for d in treat_data_list),
        len(time_hours),
    )
    ref_stack   = np.vstack([d[:min_len] for d in ref_data_list])
    treat_stack = np.vstack([d[:min_len] for d in treat_data_list])

    ref_mean   = np.mean(ref_stack,   axis=0)
    treat_mean = np.mean(treat_stack, axis=0)
    ref_sem    = stats.sem(ref_stack,   axis=0)
    treat_sem  = stats.sem(treat_stack, axis=0)

    # Separated when treat band sits entirely below ref band
    # (susceptible: treat < ref).  No-overlap also if treat > ref + sems.
    sep_below = (ref_mean - ref_sem) > (treat_mean + treat_sem)
    sep_above = (treat_mean - treat_sem) > (ref_mean + ref_sem)
    separated = sep_below | sep_above

    # Persistence: require run of >= min_consecutive
    run = 0
    for t in range(len(separated)):
        run = run + 1 if separated[t] else 0
        if run >= min_consecutive:
            return float(time_hours[t - min_consecutive + 1])
    return None


def compute_ttd_statistics(ref_data_list: List[np.ndarray],
                          treat_data_list: List[np.ndarray],
                          time_hours: np.ndarray,
                          alpha: float = 0.05,
                          min_consecutive: int = 3) -> Dict:
    """
    Compute time-to-detection with statistical significance and confidence intervals.
    Uses independent t-tests at each timepoint to find first significant divergence.
    
    Args:
        ref_data_list: List of reference time series (one per position)
        treat_data_list: List of treatment time series (one per position)
        time_hours: Time array in hours
        alpha: Significance level for t-test (default 0.05)
        min_consecutive: Minimum consecutive significant timepoints (default 3)
        
    Returns:
        statistics: Dictionary with TTD, p-values, and confidence intervals
    """
    # Truncate to minimum length
    min_len = min(
        min(len(d) for d in ref_data_list),
        min(len(d) for d in treat_data_list)
    )
    
    ref_data_list = [d[:min_len] for d in ref_data_list]
    treat_data_list = [d[:min_len] for d in treat_data_list]
    time_hours = time_hours[:min_len]
    
    # Stack data for statistics
    ref_stacked = np.vstack(ref_data_list)  # shape: (n_positions, n_timepoints)
    treat_stacked = np.vstack(treat_data_list)
    
    # Compute p-values at each timepoint
    p_values = []
    for t in range(min_len):
        ref_t = ref_stacked[:, t]
        treat_t = treat_stacked[:, t]
        
        # Independent samples t-test
        _, p = stats.ttest_ind(ref_t, treat_t)
        p_values.append(p)
    
    p_values = np.array(p_values)
    
    # Find first sustained significant divergence
    ttd_idx = None
    for t in range(len(p_values) - min_consecutive + 1):
        if np.all(p_values[t:t+min_consecutive] < alpha):
            ttd_idx = t
            break
    
    # Compute confidence interval via bootstrap
    if ttd_idx is not None:
        # Bootstrap TTD across positions
        ttd_times = []
        n_positions = min(len(ref_data_list), len(treat_data_list))
        
        for _ in range(1000):  # 1000 bootstrap samples
            # Resample positions with replacement
            sample_indices = np.random.choice(n_positions, size=n_positions, replace=True)
            
            ref_sample = np.vstack([ref_data_list[i] for i in sample_indices])
            treat_sample = np.vstack([treat_data_list[i] for i in sample_indices])
            
            # Find TTD for this bootstrap sample
            for t in range(len(p_values) - min_consecutive + 1):
                p_vals_boot = []
                for tp in range(t, t + min_consecutive):
                    _, p = stats.ttest_ind(ref_sample[:, tp], treat_sample[:, tp])
                    p_vals_boot.append(p)
                
                if np.all(np.array(p_vals_boot) < alpha):
                    ttd_times.append(time_hours[t])
                    break
        
        # Compute 95% CI
        if len(ttd_times) > 0:
            ttd_median = np.median(ttd_times)
            ci_lower, ci_upper = bootstrap_ci(np.array(ttd_times))
        else:
            ttd_median = time_hours[ttd_idx]
            ci_lower = ci_upper = ttd_median
        
        ttd_hours = time_hours[ttd_idx]
    else:
        ttd_hours = None
        ttd_median = None
        ci_lower = None
        ci_upper = None
    
    return {
        'ttd_idx': ttd_idx,
        'ttd_hours': ttd_hours,
        'ttd_median': ttd_median,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'p_values': p_values,
        'min_p_value': np.min(p_values) if len(p_values) > 0 else None
    }


def compute_segmentation_metrics(pred_mask: np.ndarray,
                                gt_mask: np.ndarray) -> Dict[str, float]:
    """
    Compute segmentation quality metrics.
    
    Args:
        pred_mask: Predicted segmentation
        gt_mask: Ground truth segmentation
        
    Returns:
        metrics: Dictionary of metric values
    """
    # Binarize
    pred_binary = (pred_mask > 0).astype(int).flatten()
    gt_binary = (gt_mask > 0).astype(int).flatten()
    
    # Pixel-level IoU (Jaccard)
    iou = jaccard_score(gt_binary, pred_binary, average='binary')
    
    # Pixel accuracy
    accuracy = np.mean(pred_binary == gt_binary)
    
    return {
        'iou': iou,
        'pixel_accuracy': accuracy
    }


# MAIN ANALYSIS WORKFLOW

def main_analysis():
    """
    Run complete analysis comparing baseline vs hybrid methods.
    """
    start_time = time.time()
    
    print("="*70)
    print("HYBRID CV PIPELINE ANALYSIS")
    print("="*70)
    print(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    

    # STEP 1: Process Reference (Untreated) Data

    print("\n[1] Processing REFERENCE (untreated) data...")
    
    ref_baseline_results = []
    ref_hybrid_results = []
    
    for pos in config.REF_POSITIONS:
        raw_pos_dir = os.path.join(config.REF_RAW_DIR, f"Pos{pos}")
        mask_pos_dir = os.path.join(config.REF_MASK_DIR, f"Pos{pos}")
        
        if not os.path.exists(raw_pos_dir) or not os.path.exists(mask_pos_dir):
            continue
        
        print(f"  Processing Pos{pos}...")
        
        # Baseline
        baseline_res = process_single_position(
            raw_pos_dir,
            mask_pos_dir,
            os.path.join(config.OUTPUT_DIR, "REF_baseline"),
            use_hybrid=False
        )
        if baseline_res is not None:
            ref_baseline_results.append(baseline_res)
        
        # Hybrid
        hybrid_res = process_single_position(
            raw_pos_dir,
            mask_pos_dir,
            os.path.join(config.OUTPUT_DIR, "REF_hybrid"),
            use_hybrid=True
        )
        if hybrid_res is not None:
            ref_hybrid_results.append(hybrid_res)
    
    # Aggregate reference results
    ref_base_area_mean, ref_base_area_std, ref_base_area_sem = \
        aggregate_positions(ref_baseline_results, 'areas')
    ref_hyb_area_mean, ref_hyb_area_std, ref_hyb_area_sem = \
        aggregate_positions(ref_hybrid_results, 'areas')
    
    ref_base_gr_mean, ref_base_gr_std, ref_base_gr_sem = \
        aggregate_positions(ref_baseline_results, 'growth_rates')
    ref_hyb_gr_mean, ref_hyb_gr_std, ref_hyb_gr_sem = \
        aggregate_positions(ref_hybrid_results, 'growth_rates')

    ref_processing_time = time.time() - start_time
    print(f"  ✓ Processed {len(ref_baseline_results)} reference positions in {ref_processing_time:.1f}s")
    

    # STEP 2: Process Treatment (RIF10) Data

    print("\n[2] Processing TREATMENT (RIF10) data...")
    
    treat_baseline_results = []
    treat_hybrid_results = []
    
    for pos in config.RIF10_POSITIONS:
        raw_pos_dir = os.path.join(config.RIF10_RAW_DIR, f"Pos{pos}")
        mask_pos_dir = os.path.join(config.RIF10_MASK_DIR, f"Pos{pos}")
        
        if not os.path.exists(raw_pos_dir) or not os.path.exists(mask_pos_dir):
            continue
        
        print(f"  Processing Pos{pos}...")
        
        # Baseline
        baseline_res = process_single_position(
            raw_pos_dir,
            mask_pos_dir,
            os.path.join(config.OUTPUT_DIR, "RIF10_baseline"),
            use_hybrid=False
        )
        if baseline_res is not None:
            treat_baseline_results.append(baseline_res)
        
        # Hybrid
        hybrid_res = process_single_position(
            raw_pos_dir,
            mask_pos_dir,
            os.path.join(config.OUTPUT_DIR, "RIF10_hybrid"),
            use_hybrid=True
        )
        if hybrid_res is not None:
            treat_hybrid_results.append(hybrid_res)
    
    # Aggregate treatment results
    treat_base_area_mean, treat_base_area_std, treat_base_area_sem = \
        aggregate_positions(treat_baseline_results, 'areas')
    treat_hyb_area_mean, treat_hyb_area_std, treat_hyb_area_sem = \
        aggregate_positions(treat_hybrid_results, 'areas')
    
    treat_base_gr_mean, treat_base_gr_std, treat_base_gr_sem = \
        aggregate_positions(treat_baseline_results, 'growth_rates')
    treat_hyb_gr_mean, treat_hyb_gr_std, treat_hyb_gr_sem = \
        aggregate_positions(treat_hybrid_results, 'growth_rates')

    treat_processing_time = time.time() - start_time - ref_processing_time
    print(f"  ✓ Processed {len(treat_baseline_results)} treatment positions in {treat_processing_time:.1f}s")
    

    # STEP 3: Calculate Time-to-Detection with Statistical Tests

    print("\n[3] Calculating time-to-detection with statistical validation...")
    
    growth_analyzer = GrowthAnalyzer(rolling_window=config.ROLLING_WINDOW,
                                    interval_minutes=config.INTERVAL_MINUTES)
    
    # Convert to hours
    time_array = np.arange(len(ref_base_area_mean)) * config.INTERVAL_MINUTES / 60
    time_gr = time_array[config.ROLLING_WINDOW:]

    # Extract individual position data for statistical tests
    ref_base_gr_list = [r['growth_rates'] for r in ref_baseline_results]
    treat_base_gr_list = [r['growth_rates'] for r in treat_baseline_results]

    ref_hyb_gr_list = [r['growth_rates'] for r in ref_hybrid_results]
    treat_hyb_gr_list = [r['growth_rates'] for r in treat_hybrid_results]

    # Statistical TTD calculations with confidence intervals
    print("\n  Computing statistical significance tests...")

    # Baseline method (area-based growth rate)
    ttd_stats_baseline = compute_ttd_statistics(
        ref_base_gr_list, treat_base_gr_list, time_gr, alpha=0.05, min_consecutive=3)

    # Hybrid method (area-based growth rate)
    ttd_stats_hybrid = compute_ttd_statistics(
        ref_hyb_gr_list, treat_hyb_gr_list, time_gr, alpha=0.05, min_consecutive=3)

    # Paper-faithful SEM-overlap baseline (Tran et al. 2025 — no statistical
    # significance test, just the time the SEM bands stop overlapping).
    # Reported alongside so we can directly compare against paper numbers.
    ttd_paper_baseline = detect_separation_sem_overlap(
        ref_base_gr_list, treat_base_gr_list, time_gr, min_consecutive=3)
    ttd_paper_hybrid = detect_separation_sem_overlap(
        ref_hyb_gr_list, treat_hyb_gr_list, time_gr, min_consecutive=3)

    # Extract TTD values
    ttd_base_hours = ttd_stats_baseline['ttd_hours']
    ttd_hyb_area_hours = ttd_stats_hybrid['ttd_hours']

    # Use indices for plotting (from original simple method for backward compatibility)
    ttd_baseline = growth_analyzer.detect_divergence_time(
        ref_base_gr_mean, treat_base_gr_mean, alpha=0.05, min_consecutive=3)
    ttd_hybrid_area = growth_analyzer.detect_divergence_time(
        ref_hyb_gr_mean, treat_hyb_gr_mean, alpha=0.05, min_consecutive=3)

    # Print results with confidence intervals
    print(f"\n  TIME-TO-DETECTION RESULTS (with 95% Confidence Intervals):")
    print(f"  {'Method':<30} {'TTD (hours)':<25} {'Min p-value':<15}")
    print(f"  {'-'*70}")

    if ttd_base_hours is not None:
        ci_str = f"[{ttd_stats_baseline['ci_lower']:.2f}, {ttd_stats_baseline['ci_upper']:.2f}]"
        print(f"  {'Baseline (Omnipose + Area)':<30} {ttd_base_hours:>6.2f}h  95% CI: {ci_str:<10}  {ttd_stats_baseline['min_p_value']:.2e}")
    else:
        print(f"  {'Baseline (Omnipose + Area)':<30} {'Not detected':>25}")

    if ttd_hyb_area_hours is not None:
        ci_str = f"[{ttd_stats_hybrid['ci_lower']:.2f}, {ttd_stats_hybrid['ci_upper']:.2f}]"
        print(f"  {'Hybrid (Area-based)':<30} {ttd_hyb_area_hours:>6.2f}h  95% CI: {ci_str:<10}  {ttd_stats_hybrid['min_p_value']:.2e}")
    else:
        print(f"  {'Hybrid (Area-based)':<30} {'Not detected':>25}")

    # Paper-faithful baseline (no statistical test, just SEM-band separation)
    print(f"\n  PAPER-STYLE BASELINE (Tran et al. 2025 — SEM-overlap criterion):")
    if ttd_paper_baseline is not None:
        print(f"  {'Baseline (paper SEM)':<30} {ttd_paper_baseline:>6.2f}h")
    else:
        print(f"  {'Baseline (paper SEM)':<30} {'Not detected':>25}")
    if ttd_paper_hybrid is not None:
        print(f"  {'Hybrid   (paper SEM)':<30} {ttd_paper_hybrid:>6.2f}h")
    else:
        print(f"  {'Hybrid   (paper SEM)':<30} {'Not detected':>25}")

    ttd_time = time.time() - start_time - ref_processing_time - treat_processing_time
    print(f"\n  Completed statistical analysis in {ttd_time:.1f}s")
    

    # STEP 4: Visualize Results

    print("\n[4] Generating visualizations...")
    
    # Plot 1: Area growth comparison (baseline)
    plot_growth_comparison(
        time_array, ref_base_area_mean, treat_base_area_mean,
        ref_base_area_sem, treat_base_area_sem,
        ttd_area=(ttd_baseline + config.ROLLING_WINDOW
                  if ttd_baseline is not None else None),
        save_path=os.path.join(config.OUTPUT_DIR, "baseline_area_growth.png")
    )

    # Plot 2: Area growth comparison (hybrid)
    plot_growth_comparison(
        time_array, ref_hyb_area_mean, treat_hyb_area_mean,
        ref_hyb_area_sem, treat_hyb_area_sem,
        ttd_area=(ttd_hybrid_area + config.ROLLING_WINDOW
                  if ttd_hybrid_area is not None else None),
        save_path=os.path.join(config.OUTPUT_DIR, "hybrid_area_growth.png")
    )
    
    # Plot 3: Growth rate comparison (baseline vs hybrid)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor='white')
    
    # Growth rates - baseline
    time_gr = time_array[config.ROLLING_WINDOW:len(ref_base_gr_mean)+config.ROLLING_WINDOW]
    axes[0, 0].axvline(x=0, color='#FF1F5B', linestyle='--', lw=1, alpha=0.5)
    axes[0, 0].plot(time_gr, ref_base_gr_mean, lw=2, color='#009ADE', label='REF')
    axes[0, 0].fill_between(time_gr, ref_base_gr_mean - ref_base_gr_sem,
                           ref_base_gr_mean + ref_base_gr_sem, alpha=0.3, color='#009ADE')
    axes[0, 0].plot(time_gr, treat_base_gr_mean, lw=2, color='#FF1F5B', label='RIF10')
    axes[0, 0].fill_between(time_gr, treat_base_gr_mean - treat_base_gr_sem,
                           treat_base_gr_mean + treat_base_gr_sem, alpha=0.3, color='#FF1F5B')
    axes[0, 0].set_title('Baseline: Growth Rate (h⁻¹)', fontweight='bold')
    axes[0, 0].set_ylabel('Growth Rate (h⁻¹)')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # Growth rates - hybrid
    time_gr_hyb = time_array[config.ROLLING_WINDOW:len(ref_hyb_gr_mean)+config.ROLLING_WINDOW]
    axes[0, 1].axvline(x=0, color='#FF1F5B', linestyle='--', lw=1, alpha=0.5)
    axes[0, 1].plot(time_gr_hyb, ref_hyb_gr_mean, lw=2, color='#009ADE', label='REF')
    axes[0, 1].fill_between(time_gr_hyb, ref_hyb_gr_mean - ref_hyb_gr_sem,
                           ref_hyb_gr_mean + ref_hyb_gr_sem, alpha=0.3, color='#009ADE')
    axes[0, 1].plot(time_gr_hyb, treat_hyb_gr_mean, lw=2, color='#FF1F5B', label='RIF10')
    axes[0, 1].fill_between(time_gr_hyb, treat_hyb_gr_mean - treat_hyb_gr_sem,
                           treat_hyb_gr_mean + treat_hyb_gr_sem, alpha=0.3, color='#FF1F5B')
    axes[0, 1].set_title('Hybrid: Growth Rate (h⁻¹)', fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # Variance comparison
    axes[1, 0].plot(time_gr, ref_base_gr_std, lw=2, color='#009ADE', 
                   label='Baseline REF', linestyle='--')
    axes[1, 0].plot(time_gr_hyb, ref_hyb_gr_std, lw=2, color='#009ADE',
                   label='Hybrid REF')
    axes[1, 0].set_title('Reference Variability (Std Dev)', fontweight='bold')
    axes[1, 0].set_ylabel('Standard Deviation')
    axes[1, 0].set_xlabel('Time (hours)')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    # Treatment variability (hybrid vs baseline)
    axes[1, 1].plot(time_gr, treat_base_gr_std, lw=2, color='#FF1F5B',
                    label='Baseline RIF10', linestyle='--')
    axes[1, 1].plot(time_gr_hyb, treat_hyb_gr_std, lw=2, color='#FF1F5B',
                    label='Hybrid RIF10')
    axes[1, 1].set_title('Treatment Variability (Std Dev)', fontweight='bold')
    axes[1, 1].set_xlabel('Time (hours)')
    axes[1, 1].set_ylabel('Standard Deviation')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.OUTPUT_DIR, "comprehensive_comparison.png"), dpi=config.DPI)
    plt.close()
    
    viz_time = time.time() - start_time - ref_processing_time - treat_processing_time - ttd_time
    print(f"  ✓ Generated visualizations in {viz_time:.1f}s")
    
    # Plot 5: Normalized growth rate (treatment/reference ratio)
    if len(treat_hyb_gr_mean) > 0 and len(ref_hyb_gr_mean) > 0:
        normalized_rates = growth_analyzer.normalize_growth_rates(
            treat_hyb_gr_mean, ref_hyb_gr_mean)
        plot_normalized_growth(
            time_array,
            normalized_rates,
            save_path=os.path.join(config.OUTPUT_DIR, "normalized_growth_rate.png")
        )
    

    # STEP 5: Generate Summary Report

    print("\n[5] Generating summary report...")
    
    report = {
        'n_ref_positions': len(ref_baseline_results),
        'n_treat_positions': len(treat_baseline_results),
        'ttd_baseline_hours': ttd_base_hours,
        'ttd_hybrid_area_hours': ttd_hyb_area_hours,
        'ttd_paper_baseline_hours': ttd_paper_baseline,
        'ttd_paper_hybrid_hours':   ttd_paper_hybrid,
        'ttd_stats_baseline': ttd_stats_baseline,
        'ttd_stats_hybrid': ttd_stats_hybrid,
        'ref_baseline_area_mean': ref_base_area_mean,
        'ref_hybrid_area_mean': ref_hyb_area_mean,
        'treat_baseline_area_mean': treat_base_area_mean,
        'treat_hybrid_area_mean': treat_hyb_area_mean,
        'ref_baseline_gr_std': np.mean(ref_base_gr_std),
        'ref_hybrid_gr_std': np.mean(ref_hyb_gr_std),
    }
    
    # Calculate variance reduction
    if report['ref_baseline_gr_std'] > 0:
        variance_reduction = (1 - report['ref_hybrid_gr_std'] / report['ref_baseline_gr_std']) * 100
        report['variance_reduction_pct'] = variance_reduction
    
    with open(os.path.join(config.OUTPUT_DIR, 'analysis_summary.pickle'), 'wb') as f:
        pickle.dump(report, f)
    
    # Print summary
    print("\n" + "="*70)
    print("ANALYSIS SUMMARY")
    print("="*70)
    print(f"Positions analyzed: {report['n_ref_positions']} REF, {report['n_treat_positions']} RIF10")
    print(f"\nVariance Reduction (REF growth rate):")
    if 'variance_reduction_pct' in report:
        print(f"  Hybrid vs Baseline: {report['variance_reduction_pct']:.1f}% reduction")
    print(f"\nTime-to-Detection (with 95% Confidence Intervals):")
    
    if report['ttd_baseline_hours']:
        ci_base = f"[{report['ttd_stats_baseline']['ci_lower']:.2f}, {report['ttd_stats_baseline']['ci_upper']:.2f}]"
        print(f"  Baseline:        {report['ttd_baseline_hours']:.2f}h  (95% CI: {ci_base})")
    else:
        print(f"  Baseline:        Not detected")
    
    if report['ttd_hybrid_area_hours']:
        ci_hyb = f"[{report['ttd_stats_hybrid']['ci_lower']:.2f}, {report['ttd_stats_hybrid']['ci_upper']:.2f}]"
        print(f"  Hybrid (area):   {report['ttd_hybrid_area_hours']:.2f}h  (95% CI: {ci_hyb})")
    else:
        print(f"  Hybrid (area):   Not detected")

    if report['ttd_baseline_hours'] and report['ttd_hybrid_area_hours']:
        improvement = report['ttd_baseline_hours'] - report['ttd_hybrid_area_hours']
        p_val = report['ttd_stats_hybrid']['min_p_value']
        print(f"\n✨ Improvement: {improvement:.2f}h faster detection with hybrid (memory mask)")
        print(f"   Statistical significance: p < {p_val:.2e}")
    
    total_time = time.time() - start_time
    
    print("\n" + "="*70)
    print("EXECUTION TIMING")
    print("="*70)
    print(f"REF processing:      {ref_processing_time:>8.1f}s")
    print(f"RIF10 processing:    {treat_processing_time:>8.1f}s")
    print(f"Statistical tests:   {ttd_time:>8.1f}s")
    print(f"Visualizations:      {viz_time:>8.1f}s")
    print(f"{'='*70}")
    print(f"TOTAL EXECUTION:     {total_time:>8.1f}s ({total_time/60:.1f} minutes)")
    print("\n" + "="*70)
    print(f"Results saved to: {config.OUTPUT_DIR}")
    print(f"Completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    return report


if __name__ == "__main__":
    results = main_analysis()
