"""
Interactive Parameter Tuning Tool for Hybrid Pipeline

This script helps you find optimal parameters by visualizing results
with different gaussian_sigma and sobel_ksize values.

PS: We don't really see much of a change with sigma and sobel sizes at the moment
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import os
import skimage.io
from skimage.segmentation import mark_boundaries
from hybrid_pipeline import HybridSegmentationPipeline
import config


class ParameterTuner:
    """Interactive parameter tuning interface"""
    
    def __init__(self, frames, omnipose_masks):
        self.frames = frames
        self.omnipose_masks = omnipose_masks
        self.n_frames = len(frames)
        self.current_frame = 0
        
        # Initial parameters
        self.gaussian_sigma = 1.0
        self.sobel_ksize = 3
        self.use_memory = False
        
        # Create figure with subplots and sliders
        self.fig = plt.figure(figsize=(20, 12))
        self.fig.suptitle('Parameter Tuning Tool - Adjust sliders to see effects', 
                         fontsize=16, fontweight='bold')
        
        # Create grid
        gs = self.fig.add_gridspec(3, 3, hspace=0.25, wspace=0.2,
                                   left=0.05, right=0.95, top=0.88, bottom=0.25)
        
        # Image axes
        self.ax_raw = self.fig.add_subplot(gs[0, 0])
        self.ax_omni = self.fig.add_subplot(gs[0, 1])
        self.ax_refined = self.fig.add_subplot(gs[0, 2])
        self.ax_edges = self.fig.add_subplot(gs[1, 0])
        self.ax_omni_overlay = self.fig.add_subplot(gs[1, 1])
        self.ax_refined_overlay = self.fig.add_subplot(gs[1, 2])
        self.ax_diff = self.fig.add_subplot(gs[2, :])
        
        # Create sliders
        slider_left = 0.15
        slider_width = 0.3
        
        # Frame slider
        ax_frame = plt.axes([slider_left, 0.18, slider_width, 0.02])
        self.slider_frame = Slider(ax_frame, 'Frame', 0, self.n_frames - 1,
                                   valinit=0, valstep=1, color='#009ADE')
        self.slider_frame.on_changed(self.update_frame)
        
        # Gaussian sigma slider
        ax_sigma = plt.axes([slider_left, 0.14, slider_width, 0.02])
        self.slider_sigma = Slider(ax_sigma, 'Gaussian σ', 0.1, 5.0,
                                   valinit=1.0, color='#FF6B35')
        self.slider_sigma.on_changed(self.update_sigma)
        
        # Sobel ksize slider (must be odd)
        ax_sobel = plt.axes([slider_left, 0.10, slider_width, 0.02])
        self.slider_sobel = Slider(ax_sobel, 'Sobel Kernel', 1, 15,
                                   valinit=3, valstep=2, color='#4ECDC4')
        self.slider_sobel.on_changed(self.update_sobel)

        # Buttons
        ax_btn_memory = plt.axes([0.55, 0.10, 0.15, 0.03])
        self.btn_memory = Button(ax_btn_memory, 'Memory: OFF')
        self.btn_memory.on_clicked(self.toggle_memory)

        ax_btn_reset = plt.axes([0.55, 0.06, 0.15, 0.03])
        self.btn_reset = Button(ax_btn_reset, 'Reset to Defaults')
        self.btn_reset.on_clicked(self.reset_params)
        
        # Info text
        self.info_text = self.fig.text(0.75, 0.15, '', fontsize=10,
                                       verticalalignment='top',
                                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Initial display
        self.update_display()
    
    def update_frame(self, val):
        """Update frame index"""
        self.current_frame = int(self.slider_frame.val)
        self.update_display()
    
    def update_sigma(self, val):
        """Update gaussian sigma"""
        self.gaussian_sigma = self.slider_sigma.val
        self.update_display()
    
    def update_sobel(self, val):
        """Update sobel kernel size"""
        self.sobel_ksize = int(self.slider_sobel.val)
        self.update_display()

    def toggle_memory(self, event):
        """Toggle memory mask on/off"""
        self.use_memory = not self.use_memory
        self.btn_memory.label.set_text(f'Memory: {"ON" if self.use_memory else "OFF"}')
        self.update_display()

    def reset_params(self, event):
        """Reset to default parameters"""
        self.slider_sigma.set_val(1.0)
        self.slider_sobel.set_val(3)
        self.use_memory = False
        self.btn_memory.label.set_text('Memory: OFF')
        self.update_display()
    
    def update_display(self):
        """Process and display with current parameters"""
        # Clear axes
        for ax in [self.ax_raw, self.ax_omni, self.ax_refined, self.ax_edges,
                   self.ax_omni_overlay, self.ax_refined_overlay, self.ax_diff]:
            ax.clear()
        
        # Get current frame
        idx = self.current_frame
        frame = self.frames[idx]
        omni_mask = self.omnipose_masks[idx]
        
        # Ensure same shape
        min_h = min(frame.shape[0], omni_mask.shape[0])
        min_w = min(frame.shape[1], omni_mask.shape[1])
        frame = frame[:min_h, :min_w]
        omni_mask = omni_mask[:min_h, :min_w]
        
        # Create pipeline with current parameters
        pipeline = HybridSegmentationPipeline(
            gaussian_sigma=self.gaussian_sigma,
            sobel_ksize=self.sobel_ksize,
        )

        # Process current frame
        blurred, edges = pipeline.preprocess_frame(frame)
        refined_mask = omni_mask.copy()

        # Apply memory if enabled (need to process from start)
        if self.use_memory and idx > 0:
            refined_masks, _ = pipeline.process_sequence(
                self.frames[:idx+1], 
                self.omnipose_masks[:idx+1], 
                use_memory=True
            )
            refined_mask = refined_masks[-1]
        
        # Crop refined mask to match frame
        refined_mask = refined_mask[:min_h, :min_w]
        edges = edges[:min_h, :min_w] if edges.shape[0] > min_h or edges.shape[1] > min_w else edges
        
        # Calculate metrics
        omni_cells = np.max(omni_mask)
        refined_cells = np.max(refined_mask)
        omni_binary = (omni_mask > 0).astype(float)
        refined_binary = (refined_mask > 0).astype(float)
        diff = refined_binary - omni_binary
        added_pct = (diff > 0).sum() / (omni_binary.sum() + 1e-6) * 100
        removed_pct = (diff < 0).sum() / (omni_binary.sum() + 1e-6) * 100
        
        # 1. Raw image
        self.ax_raw.imshow(frame, cmap='gray')
        self.ax_raw.set_title('Raw Image', fontsize=12, fontweight='bold')
        self.ax_raw.axis('off')
        
        # 2. Omnipose mask
        self.ax_omni.imshow(omni_mask, cmap='nipy_spectral')
        self.ax_omni.set_title(f'Omnipose ({omni_cells} cells)', fontsize=12, fontweight='bold')
        self.ax_omni.axis('off')
        
        # 3. Refined mask
        self.ax_refined.imshow(refined_mask, cmap='nipy_spectral')
        self.ax_refined.set_title(f'Hybrid ({refined_cells} cells)', fontsize=12, fontweight='bold')
        self.ax_refined.axis('off')
        
        # 4. Edges
        self.ax_edges.imshow(edges, cmap='hot')
        self.ax_edges.set_title(f'Sobel Edges (σ={self.gaussian_sigma:.1f}, k={self.sobel_ksize})', 
                               fontsize=12, fontweight='bold')
        self.ax_edges.axis('off')
        
        # 5. Omnipose overlay
        omni_overlay = mark_boundaries(frame, omni_mask, color=(0, 1, 0), mode='thick')
        self.ax_omni_overlay.imshow(omni_overlay)
        self.ax_omni_overlay.set_title('Omnipose Overlay', fontsize=12, fontweight='bold')
        self.ax_omni_overlay.axis('off')
        
        # 6. Refined overlay
        refined_overlay = mark_boundaries(frame, refined_mask, color=(1, 0.5, 0), mode='thick')
        self.ax_refined_overlay.imshow(refined_overlay)
        self.ax_refined_overlay.set_title('Hybrid Overlay', fontsize=12, fontweight='bold')
        self.ax_refined_overlay.axis('off')
        
        # 7. Difference
        diff_rgb = np.zeros((*frame.shape, 3))
        diff_rgb[diff > 0] = [0, 1, 0]  # Added
        diff_rgb[diff < 0] = [1, 0, 0]  # Removed
        diff_rgb[omni_binary * refined_binary > 0] = [0.7, 0.7, 0.7]  # Same
        
        self.ax_diff.imshow(frame, cmap='gray', alpha=0.5)
        self.ax_diff.imshow(diff_rgb, alpha=0.5)
        self.ax_diff.set_title(f'Difference: +{added_pct:.1f}% / -{removed_pct:.1f}%', 
                              fontsize=12, fontweight='bold')
        self.ax_diff.axis('off')
        
        # Update info text
        info = f"Current Parameters:\n"
        info += f"Frame: {idx}/{self.n_frames-1}\n"
        info += f"Gaussian σ: {self.gaussian_sigma:.2f}\n"
        info += f"Sobel kernel: {self.sobel_ksize}\n"
        info += f"Memory: {'ON' if self.use_memory else 'OFF'}\n\n"
        info += f"Results:\n"
        info += f"Cell count change: {refined_cells - omni_cells:+d}\n"
        info += f"Pixels added: {added_pct:.1f}%\n"
        info += f"Pixels removed: {removed_pct:.1f}%"
        
        self.info_text.set_text(info)
        
        self.fig.canvas.draw_idle()


def main():
    """Run the parameter tuner"""
    print("="*70)
    print("PARAMETER TUNING TOOL")
    print("="*70)
    
    # Load data
    raw_dir, mask_dir = config.get_position_paths(101, is_treatment=False)
    
    if not os.path.exists(raw_dir) or not os.path.exists(mask_dir):
        print(f"❌ Data not found. Please update BASE_DIR in config.py")
        return
    
    print(f"Loading frames from Pos101...")
    
    raw_files = sorted([os.path.join(raw_dir, f) for f in os.listdir(raw_dir)
                       if f.endswith('.tiff') or f.endswith('.tif')])[:30]
    mask_files = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir)
                        if f.startswith('MASK_')])[:30]
    
    frames = [skimage.io.imread(f) for f in raw_files]
    omnipose_masks = [skimage.io.imread(f) for f in mask_files]
    
    tuner = ParameterTuner(frames, omnipose_masks)
    plt.show()


if __name__ == "__main__":
    main()
