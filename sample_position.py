import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import os
import skimage.io
from skimage.segmentation import mark_boundaries
from hybrid_pipeline import (
    HybridSegmentationPipeline,
    GrowthAnalyzer,
)

import config

# INTERACTIVE FRAME VIEWER

def view_frames_interactive(frames, omnipose_masks, refined_masks, edges=None):
    """
    Interactive viewer to navigate through all frames with slider and keyboard
    
    Args:
        frames: List of raw images
        omnipose_masks: List of Omnipose segmentation masks
        refined_masks: List of hybrid refined masks
        edges: Optional list of edge maps
    """
    class FrameViewer:
        def __init__(self, frames, omni_masks, refine_masks, edges):
            self.frames = frames
            self.omni_masks = omni_masks
            self.refine_masks = refine_masks
            self.edges = edges
            self.n_frames = min(30, len(frames)) # view 30 frames
            self.current_idx = 0
            
            # Create figure
            self.fig = plt.figure(figsize=(24, 14))
            self.fig.canvas.manager.set_window_title('Segmentation Viewer - Use Slider/Arrow Keys')
            
            # Create grid for subplots (2x3)
            gs = self.fig.add_gridspec(2, 3, hspace=0.12, wspace=0.12, 
                                       left=0.03, right=0.97, top=0.92, bottom=0.08)
            
            # Create axes
            self.ax1 = self.fig.add_subplot(gs[0, 0])  # Raw
            self.ax2 = self.fig.add_subplot(gs[0, 1])  # Omnipose mask
            self.ax3 = self.fig.add_subplot(gs[0, 2])  # Refined mask
            self.ax4 = self.fig.add_subplot(gs[1, 0])  # Omnipose overlay
            self.ax5 = self.fig.add_subplot(gs[1, 1])  # Refined overlay
            self.ax6 = self.fig.add_subplot(gs[1, 2])  # Difference
            
            # Create slider
            self.slider_ax = plt.axes([0.15, 0.05, 0.65, 0.02])
            self.slider = Slider(
                self.slider_ax, 'Frame', 0, self.n_frames - 1,
                valinit=0, valstep=1, color='#009ADE'
            )
            self.slider.on_changed(self.update_from_slider)
            
            # Create buttons
            self.btn_prev_ax = plt.axes([0.15, 0.01, 0.08, 0.03])
            self.btn_next_ax = plt.axes([0.72, 0.01, 0.08, 0.03])
            self.btn_prev = Button(self.btn_prev_ax, '◀ Previous')
            self.btn_next = Button(self.btn_next_ax, 'Next ▶')
            self.btn_prev.on_clicked(self.prev_frame)
            self.btn_next.on_clicked(self.next_frame)
            
            # Connect keyboard events
            self.fig.canvas.mpl_connect('key_press_event', self.on_key)
            
            # Initial display
            self.update_display()
            
        def update_display(self):
            """Update all subplots with current frame"""
            idx = self.current_idx
            
            # Clear all axes
            for ax in [self.ax1, self.ax2, self.ax3, self.ax4, self.ax5, self.ax6]:
                ax.clear()
            
            # Get current data
            frame = self.frames[idx]
            omni_mask = self.omni_masks[idx]
            refine_mask = self.refine_masks[idx]
            
            # Ensure all have same shape (crop to smallest dimensions)
            min_h = min(frame.shape[0], omni_mask.shape[0], refine_mask.shape[0])
            min_w = min(frame.shape[1], omni_mask.shape[1], refine_mask.shape[1])
            frame = frame[:min_h, :min_w]
            omni_mask = omni_mask[:min_h, :min_w]
            refine_mask = refine_mask[:min_h, :min_w]
            
            # Calculate time and difference stats
            time_hours = idx * config.INTERVAL_MINUTES / 60
            omni_cells = np.max(omni_mask)
            refine_cells = np.max(refine_mask)
            
            # Calculate difference
            omni_binary = (omni_mask > 0).astype(float)
            refine_binary = (refine_mask > 0).astype(float)
            diff = refine_binary - omni_binary
            added_pct = (diff > 0).sum() / (omni_binary.sum() + 1e-6) * 100
            removed_pct = (diff < 0).sum() / (omni_binary.sum() + 1e-6) * 100
            
            # Main title with all stats
            self.fig.suptitle(f'Frame {idx}/{self.n_frames-1} | Time: {time_hours:.2f}h | '
                            f'Omnipose: {omni_cells} cells | Hybrid: {refine_cells} cells | '
                            f'Difference: +{added_pct:.1f}% / -{removed_pct:.1f}%', 
                            fontsize=14, fontweight='bold')
            
            # 1. Raw image
            self.ax1.imshow(frame, cmap='gray')
            self.ax1.set_title('Raw Image', fontsize=12)
            self.ax1.axis('off')
            
            # 2. Omnipose mask
            self.ax2.imshow(omni_mask, cmap='nipy_spectral')
            self.ax2.set_title('Omnipose Segmentation', fontsize=12)
            self.ax2.axis('off')
            
            # 3. Refined mask
            self.ax3.imshow(refine_mask, cmap='nipy_spectral')
            self.ax3.set_title('Hybrid Refined', fontsize=12)
            self.ax3.axis('off')
            
            # 4. Omnipose overlay
            omni_overlay = mark_boundaries(frame, omni_mask, color=(0, 1, 0), mode='thick')
            self.ax4.imshow(omni_overlay)
            self.ax4.set_title('Omnipose Overlay', fontsize=12)
            self.ax4.axis('off')
            
            # 5. Refined overlay
            refine_overlay = mark_boundaries(frame, refine_mask, color=(1, 0.5, 0), mode='thick')
            self.ax5.imshow(refine_overlay)
            self.ax5.set_title('Hybrid Overlay', fontsize=12)
            self.ax5.axis('off')
            
            # 6. Difference visualization
            # Create RGB difference: red = removed, green = added, gray = same
            diff_rgb = np.zeros((*frame.shape, 3))
            diff_rgb[diff > 0] = [0, 1, 0]  # Added in hybrid (green)
            diff_rgb[diff < 0] = [1, 0, 0]  # Removed in hybrid (red)
            diff_rgb[omni_binary * refine_binary > 0] = [0.7, 0.7, 0.7]  # Same (gray)
            
            self.ax6.imshow(frame, cmap='gray', alpha=0.5)
            self.ax6.imshow(diff_rgb, alpha=0.5)
            self.ax6.set_title('Difference (Green=Added, Red=Removed)', fontsize=12)
            self.ax6.axis('off')
            
            self.fig.canvas.draw_idle()
        
        def update_from_slider(self, val):
            """Update from slider movement"""
            self.current_idx = int(self.slider.val)
            self.update_display()
        
        def next_frame(self, event):
            """Go to next frame"""
            if self.current_idx < self.n_frames - 1:
                self.current_idx += 1
                self.slider.set_val(self.current_idx)
        
        def prev_frame(self, event):
            """Go to previous frame"""
            if self.current_idx > 0:
                self.current_idx -= 1
                self.slider.set_val(self.current_idx)
        
        def on_key(self, event):
            """Handle keyboard input"""
            if event.key == 'right' or event.key == 'down':
                self.next_frame(None)
            elif event.key == 'left' or event.key == 'up':
                self.prev_frame(None)
            elif event.key == 'home':
                self.current_idx = 0
                self.slider.set_val(0)
            elif event.key == 'end':
                self.current_idx = self.n_frames - 1
                self.slider.set_val(self.n_frames - 1)
    
    # Create and show viewer
    viewer = FrameViewer(frames, omnipose_masks, refined_masks, edges)
    plt.show()


# Process a single position (e.g., Pos101)

def single_position(position=None, is_treatment=False):
    """
    Process a single position to see how the pipeline works
    
    Args:
        position: Position number (e.g., 101, 102, 201, etc.). If None, user will be prompted.
        is_treatment: True for RIF10 treatment positions, False for REF reference positions
    """
    
    # Get position from user if not provided
    if position is None:
        print("\nAvailable positions:")
        print("  REF (Reference/Untreated):  101-110")
        print("  RIF10 (Treatment):          201-220")
        
        while True:
            try:
                user_input = input("\nEnter position number (e.g., 101, 201): ").strip()
                position = int(user_input)
                
                # Determine if treatment based on position number
                if 101 <= position <= 110:
                    is_treatment = False
                    break
                elif 201 <= position <= 220:
                    is_treatment = True
                    break
                else:
                    print("❌ Invalid position. Must be 101-110 (REF) or 201-220 (RIF10)")
            except ValueError:
                print("❌ Please enter a valid number")
            except KeyboardInterrupt:
                print("\n\nOperation cancelled by user")
                return
    
    group_name = "RIF10 (Treatment)" if is_treatment else "REF (Reference)"
    print(f"\n✓ Selected: Pos{position} ({group_name})")
    
    # Paths to raw images and masks
    raw_dir, mask_dir = config.get_position_paths(position, is_treatment=is_treatment)
    
    # Check if directories exist
    if not os.path.exists(raw_dir):
        print(f"❌ Raw directory not found: {raw_dir}")
        print("Please update BASE_DIR in this script!")
        return
    
    if not os.path.exists(mask_dir):
        print(f"❌ Mask directory not found: {mask_dir}")
        return
    
    print(f"✓ Raw images: {raw_dir}")
    print(f"✓ Masks: {mask_dir}")
    
    # Load images
    raw_files = sorted([os.path.join(raw_dir, f) for f in os.listdir(raw_dir)
                       if f.endswith('.tiff') or f.endswith('.tif')])
    mask_files = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir)
                        if f.startswith('MASK_') and (f.endswith('.tiff') or f.endswith('.tif'))])
    
    print(f"\nFound {len(raw_files)} raw images")
    print(f"Found {len(mask_files)} mask files")
    
    if len(raw_files) == 0 or len(mask_files) == 0:
        print("❌ No files found! Check your paths.")
        return
    
    # Load frames and masks (let's use first 30 frames for quick demo)
    n_frames = min(len(raw_files), len(mask_files))
    print(f"\nLoading first {n_frames} frames...")
    
    frames = [skimage.io.imread(f) for f in raw_files[:n_frames]]
    omnipose_masks = [skimage.io.imread(f) for f in mask_files[:n_frames]]
    
    print(f"Frame shape: {frames[0].shape}")
    print(f"Mask shape: {omnipose_masks[0].shape}")
    
    # Initialize pipeline components
    print("\n" + "-"*70)
    print("Running Hybrid Pipeline...")
    print("-"*70)
    
    seg_pipeline = HybridSegmentationPipeline(
        gaussian_sigma=config.GAUSSIAN_SIGMA,
        sobel_ksize=config.SOBEL_KSIZE,
    )

    growth_analyzer = GrowthAnalyzer(
        rolling_window=config.ROLLING_WINDOW,
        interval_minutes=config.INTERVAL_MINUTES,
        pixel_size_um=config.PIXEL_SIZE_UM
    )

    # Process with hybrid pipeline
    print("1. Applying Gaussian blur + Sobel edges...")
    print("2. Applying continuous memory mask...")

    refined_masks, edges = seg_pipeline.process_sequence(
        frames, omnipose_masks, use_memory=True)

    print(f"✓ Segmentation complete! Refined {len(refined_masks)} masks")

    # Compute growth metrics
    print("\n3. Computing growth metrics...")
    areas = growth_analyzer.compute_area_growth(refined_masks)
    growth_rates = growth_analyzer.compute_growth_rate_rolling(areas)

    print(f"✓ Areas computed: {len(areas)} time points")
    print(f"✓ Growth rates: {len(growth_rates)} time points")

    # Launch interactive frame viewer
    print("\n" + "-"*70)
    print("Launching Interactive Frame Viewer...")
    print("-"*70)
    print("Use slider or arrow keys to navigate frames")
    print("Close the window when done to continue...")

    view_frames_interactive(frames, omnipose_masks, refined_masks, edges)

    # Plot growth curves
    time_hours = config.get_time_array(len(areas), start_at_zero=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), facecolor='white')

    # Area growth
    axes[0].plot(time_hours, areas, lw=2, color='#009ADE')
    axes[0].axvline(x=0, color='red', linestyle='--', alpha=0.5, label='Drug addition')
    axes[0].set_xlabel('Time (hours)')
    axes[0].set_ylabel('Area (μm²)')
    axes[0].set_title('Area Growth')
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    # Growth rate
    time_gr = time_hours[config.ROLLING_WINDOW:]
    if len(growth_rates) > 0:
        axes[1].plot(time_gr[:len(growth_rates)], growth_rates, lw=2, color='#009ADE')
        axes[1].axvline(x=0, color='red', linestyle='--', alpha=0.5)
        axes[1].set_xlabel('Time (hours)')
        axes[1].set_ylabel('Growth Rate (h⁻¹)')
        axes[1].set_title('Growth Rate')
        axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(config.SAMPLE_OUTPUT_DIR, f"growth_curves_Pos{position}.png"), dpi=config.DPI)
    plt.show()

    print(
        "\nFor heteroresistance detection (Strategies B + C) run "
        "test_heteroresistance_real_data.py — sample_position.py is a "
        "single-position viewer only."
    )

    print(f"\n✓ Results saved to: {config.SAMPLE_OUTPUT_DIR}")
    print("\n" + "="*70)
    print("Example 1 Complete!")
    print("="*70)
    print(f"\nKey Results for Pos{position} ({group_name}):")
    print(f"  • Mean area: {np.mean(areas):.1f} μm²")
    print(f"  • Mean growth rate: {np.mean(growth_rates):.3f} h⁻¹" if len(growth_rates) > 0 else "")
    

if __name__ == "__main__":
    print("HYBRID CV PIPELINE - SAMPLE USAGE")
    print("="*70)
    print("\nThis script demonstrates processing a single position with the hybrid pipeline")
    print("\n⚠️  Configuration loaded from config.py")
    print("="*70)

    config.ensure_output_dirs()
    single_position()
    
    print(f"\nOutputs saved to: {config.SAMPLE_OUTPUT_DIR}")
    print("1. Review the segmentation comparison image")
    print("2. Check the growth curves")
    print("3. Run parameter_tuner.py to optimize settings")
    print("4. Run analysis_notebook.py for full analysis")
    print("="*70)
