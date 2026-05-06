"""
Experimental Image Processing & Theoretical Overlay for Complex Caustics

This script processes experimental CCD images of complex 3D caustics (e.g., cubic surfaces),
aligns them using a robust autoconvolution algorithm, and overlays the theoretical 
parametric curves for comparison (Corresponding to Figure 4 in the paper).

Key Features:
- Global Coherent Alignment: Uses geometric autoconvolution of diffraction rings 
  to locate the optical axis with extreme precision, immune to local overexposure.
- Automated Cropping & Padding: Centers the beam accurately within the physical coordinate system.
- Publication-Quality Overlay: Renders clean, axis-free images with theoretical curve overlays.
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, sobel
from scipy.signal import fftconvolve

# ==========================================
# Physical & System Configurations
# ==========================================
INPUT_DIR = "cubic"  # Directory containing the raw experimental images
OUTPUT_DIR = "cubic_overlay"  # Directory to save the processed overlay images
PIXEL_SIZE = 2.4e-3  # CCD pixel pitch (mm/pixel)
CROP_SIZE = 3000  # Enlarged crop size in pixels
HALF_CROP = CROP_SIZE // 2
PHYSICAL_EXTENT = CROP_SIZE * PIXEL_SIZE  # 3000 * 2.4um = 7.2 mm

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# ==========================================
# Core Functions
# ==========================================


def find_optical_axis(image_path):
    """
    [Core Algorithm] Robust optical axis localization using geometric autoconvolution.
    
    Instead of relying on local peak detection, this approach treats all concentric 
    rings as a unified coherent structure. By calculating the global autoconvolution 
    of the extracted geometric edges, it perfectly resolves alignment shifts caused 
    by local overexposure or asymmetric intensity profiles.
    """
    img = plt.imread(image_path)
    if img.ndim == 3:
        intensity = np.mean(img, axis=2)
    else:
        intensity = img.copy()

    # 1. Coarse localization: Large-scale blur to find the general beam area
    rough_blur = gaussian_filter(intensity, sigma=50)
    rough_cy, rough_cx = np.unravel_index(np.argmax(rough_blur),
                                          rough_blur.shape)

    # 2. Geometric edge extraction: Apply light Gaussian smoothing and Sobel operators
    smoothed = gaussian_filter(intensity, sigma=2)
    gx = sobel(smoothed, axis=1)
    gy = sobel(smoothed, axis=0)
    grad_mag = np.sqrt(gx**2 + gy**2)

    # 3. Spatial weighting: Suppress stray edges far from the center
    Y, X = np.ogrid[:intensity.shape[0], :intensity.shape[1]]
    dist_sq = (X - rough_cx)**2 + (Y - rough_cy)**2
    # Focus on the core concentric rings within a ~500-pixel radius
    window = np.exp(-dist_sq / (2 * 500**2))
    grad_mag = grad_mag * window

    # 4. Binarization: Extract the top 10% strong edges to equalize weights
    # This automatically discards the overexposed center (which lacks internal gradients)
    threshold = np.percentile(grad_mag, 90)
    edges = (grad_mag > threshold).astype(float)

    # 5. Geometric Autoconvolution:
    # For any centrally symmetric 2D pattern f(x, y), the absolute maximum of
    # its autoconvolution f*f lies exactly at (2*cx, 2*cy).
    # mode='full' ensures boundary responses are completely preserved.
    conv = fftconvolve(edges, edges, mode='full')
    py, px = np.unravel_index(np.argmax(conv), conv.shape)

    # Divide peak coordinates by 2 to obtain the absolute center with extreme precision
    cx = px / 2.0
    cy = py / 2.0

    return int(np.round(cx)), int(np.round(cy))


def crop_and_pad(img, cx, cy, size=CROP_SIZE):
    """
    Crop a square region centered on the optical axis.
    Applies zero-padding if the optical axis is too close to the image boundaries.
    """
    half = size // 2
    H, W = img.shape[:2]

    # Calculate required padding
    pad_left = max(0, half - cx)
    pad_right = max(0, cx + half - W)
    pad_top = max(0, half - cy)
    pad_bottom = max(0, cy + half - H)

    # Apply padding if out of bounds
    if pad_left > 0 or pad_right > 0 or pad_top > 0 or pad_bottom > 0:
        if img.ndim == 3:
            img = np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right),
                               (0, 0)),
                         mode='constant')
        else:
            img = np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right)),
                         mode='constant')
        # Update center coordinates after padding
        cx += pad_left
        cy += pad_top

    # Execute cropping
    cropped = img[cy - half:cy + half, cx - half:cx + half]
    return cropped


def get_theoretical_curve(z_m):
    """
    Obtain the theoretical spatial coordinates for the Cubic fold caustic.
    """
    v = np.linspace(-1.0, 1.0, 2000)
    scale = 2.0  # Equivalent to 2e-3 m = 2 mm

    xc = scale * v
    yc = 2 * scale * (v**3 / 3 - z_m * v)
    return xc, yc


# ==========================================
# Main Workflow
# ==========================================


def process_experimental_images():
    print("★ Starting experimental image processing for complex caustics...")
    z_distances_cm = [20, 40, 60]

    for z_cm in z_distances_cm:
        z_m = z_cm / 100.0  # Convert to meters for the theoretical equation

        # 1. Locate the reference optical axis for the current propagation plane
        ref_file = os.path.join(INPUT_DIR, f"refer{z_cm}.bmp")
        if not os.path.exists(ref_file):
            print(
                f"  [Warning] Reference file not found: {ref_file}. Skipping z={z_cm}cm."
            )
            continue

        cx, cy = find_optical_axis(ref_file)
        print(
            f"  [Z = {z_cm:2d} cm] Optical axis located at: (x={cx}, y={cy})")

        # 2. Retrieve all experimental images for this plane (including the reference)
        target_files = glob.glob(os.path.join(INPUT_DIR, f"*{z_cm}.bmp"))

        for file_path in target_files:
            filename = os.path.basename(file_path)
            is_refer = "refer" in filename

            # Read and crop the image
            img = plt.imread(file_path)
            cropped_img = crop_and_pad(img, cx, cy)

            # Physical coordinate mapping:
            # Flip vertically to align with mathematical Cartesian coordinates (origin='lower')
            cropped_img = cropped_img[::-1, ...]

            # 3. Create a minimalist canvas for publication overlay
            fig, ax = plt.subplots(figsize=(6, 6), dpi=300)

            # Physical boundaries (mm)
            limit = PHYSICAL_EXTENT / 2.0
            extent = [-limit, limit, -limit, limit]

            ax.imshow(cropped_img,
                      extent=extent,
                      origin='lower',
                      cmap='gray' if img.ndim == 2 else None)

            if is_refer:
                # For reference images, plot a red cross at the physical center (0,0)
                ax.plot(0,
                        0,
                        marker='+',
                        color='red',
                        markersize=20,
                        markeredgewidth=2.5,
                        zorder=10)
            else:
                # For target images, plot the theoretical curve overlay (semi-transparent)
                xt, yt = get_theoretical_curve(z_m)
                ax.plot(xt,
                        yt,
                        color='white',
                        linewidth=3.5,
                        linestyle='--',
                        alpha=0.6,
                        zorder=10)

            # Remove axes and borders for a clean look
            ax.set_axis_off()
            ax.set_xlim([-limit, limit])
            ax.set_ylim([-limit, limit])

            # 5. Export high-resolution overlay
            out_name = filename.replace('.bmp', '_overlay.png')
            out_path = os.path.join(OUTPUT_DIR, out_name)

            plt.savefig(out_path,
                        bbox_inches='tight',
                        pad_inches=0,
                        transparent=True)
            plt.close(fig)

            print(f"    ✓ Exported successfully: {out_name}")


if __name__ == "__main__":
    process_experimental_images()
    print(
        "★ All experimental images processed! Please check the output directory."
    )
