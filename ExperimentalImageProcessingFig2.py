"""
Experimental Image Processing & Beam Radius Extraction

This script processes experimental CCD images of the reconstructed 3D caustics,
extracts the beam radius with asymmetric error bounds, and plots the propagation 
trajectory against theoretical expectations (Corresponding to Figure 2f in the paper).

Key Features:
- Dual-strategy beam center finding (ray-tracing + least squares fitting).
- Asymmetric error evaluation based on radial intensity profiles.
- Automated caching mechanism to accelerate batch processing.
- Publication-quality plotting.
"""

import os
import glob
import json
import numpy as np
import matplotlib.pyplot as plt
import imageio.v3 as iio
from scipy.ndimage import center_of_mass, gaussian_filter1d
from matplotlib.patches import Circle
from matplotlib.ticker import MaxNLocator

# ==========================================
# 1. Global Parameters & Plotting Settings
# ==========================================
PIXEL_PITCH = 2.4  # Pixel pitch of the CCD camera (μm)
IMAGE_DIR = './images_inverse'  # Directory containing experimental images
PEAK_THRESHOLD_RATIO = 0.9  # Threshold ratio for error bounds (90% of peak intensity)
DIAGNOSTIC_DIR = './diagnostics'  # Directory to save diagnostic verification plots
CACHE_DIR = './cache'  # Directory to save computational caches

# Ensure output directories exist
os.makedirs(DIAGNOSTIC_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# Academic plotting style configuration
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial"],
    "font.weight": "normal",
    "axes.labelweight": "normal",
})


# ==========================================
# 2. Core Algorithm: Extract Beam Radius & Error
# ==========================================
def extract_radius_and_error(filepath,
                             save_diagnostic=False,
                             basename="",
                             z_cm=0.0):
    """
    Process a single experimental image to extract the physical beam radius 
    and its asymmetric error bounds (in μm).
    """
    # 1. Read image and extract the green channel (optimized for green laser excitation)
    img = iio.imread(filepath)
    if img.ndim >= 3:
        img_data = img[:, :,
                       1].astype(np.float32)  # Extract G channel from RGB
    else:
        img_data = img.astype(np.float32)

    # 2. Preprocessing: Subtract CCD dark current background noise
    bg_noise = np.median(img_data[:50, :50])
    img_clean = np.clip(img_data - bg_noise, 0, None)

    # ==========================================
    # Cache Detection and Loading
    # ==========================================
    cache_path = os.path.join(CACHE_DIR, f"cache_{basename}.json")
    need_compute = True

    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r') as f:
                cache_data = json.load(f)
            # Ensure cache version and parameters match
            if cache_data.get('version') == 4 and cache_data.get(
                    'threshold_ratio') == PEAK_THRESHOLD_RATIO:
                xc = cache_data['xc']
                yc = cache_data['yc']
                r_inner_bound = cache_data['r_inner_bound']
                r_outer_bound = cache_data['r_outer_bound']
                real_r_px = cache_data['real_r_px']
                err_lower_px = cache_data['err_lower_px']
                err_upper_px = cache_data['err_upper_px']
                need_compute = False
                print(
                    f"  -> [Cache Loaded] {basename}: Skipping extraction, proceeding to plot."
                )
        except Exception as e:
            print(f"  -> Failed to load cache, recomputing: {e}")

    if need_compute:
        # ==========================================
        # 3. Beam Center Localization (Dual-Strategy based on Z distance)
        # ==========================================
        peak_intensity = np.percentile(img_clean, 99.9)
        pts_list = []
        num_rays = 72
        ray_edges = np.linspace(-np.pi, np.pi, num_rays + 1)

        # Strategy A: For near-field (z < 30 cm), features may be less sharp
        if z_cm < 30:
            mask_rough = img_clean > (peak_intensity * 0.2)
            yc_rough, xc_rough = center_of_mass(mask_rough)

            y_idx, x_idx = np.indices(img_clean.shape, dtype=np.float32)
            r_rough = np.sqrt((x_idx - xc_rough)**2 + (y_idx - yc_rough)**2)
            theta_rough = np.arctan2(y_idx - yc_rough, x_idx - xc_rough)

            threshold_boundary = peak_intensity * 0.3

            for i in range(num_rays):
                mask_ray = (theta_rough >= ray_edges[i]) & (theta_rough
                                                            < ray_edges[i + 1])
                r_ray = r_rough[mask_ray]
                i_ray = img_clean[mask_ray]
                if len(r_ray) == 0: continue

                sort_idx = np.argsort(r_ray)
                r_ray_sorted = r_ray[sort_idx]
                i_ray_sorted = i_ray[sort_idx]

                cross_indices = np.where((i_ray_sorted > threshold_boundary)
                                         & (r_ray_sorted > 5))[0]
                if len(cross_indices) > 0:
                    r_pt = r_ray_sorted[cross_indices[0]]
                    theta_mid = (ray_edges[i] + ray_edges[i + 1]) / 2.0
                    pts_list.append([
                        xc_rough + r_pt * np.cos(theta_mid),
                        yc_rough + r_pt * np.sin(theta_mid)
                    ])

        # Strategy B: For far-field (z >= 30 cm), robust peak detection
        else:
            mask_rough = img_clean > np.percentile(img_clean, 95)
            yc_rough, xc_rough = center_of_mass(mask_rough)

            y_idx, x_idx = np.indices(img_clean.shape, dtype=np.float32)
            r_rough = np.sqrt((x_idx - xc_rough)**2 + (y_idx - yc_rough)**2)
            theta_rough = np.arctan2(y_idx - yc_rough, x_idx - xc_rough)

            for i in range(num_rays):
                mask_ray = (theta_rough >= ray_edges[i]) & (theta_rough
                                                            < ray_edges[i + 1])
                r_ray = r_rough[mask_ray]
                i_ray = img_clean[mask_ray]
                if len(r_ray) == 0: continue

                r_int = np.round(r_ray).astype(int)
                bc = np.bincount(r_int)
                mc = np.bincount(r_int, weights=i_ray)
                valid = bc > 0
                if np.sum(valid) < 3: continue

                profile_ray = np.zeros(len(bc))
                profile_ray[valid] = mc[valid] / bc[valid]

                smoothed_ray = gaussian_filter1d(profile_ray, sigma=2.0)
                peak_idx_local = np.argmax(smoothed_ray)

                if smoothed_ray[peak_idx_local] < np.percentile(img_clean, 50):
                    continue

                r_peak = peak_idx_local
                theta_mid = (ray_edges[i] + ray_edges[i + 1]) / 2.0
                pts_list.append([
                    xc_rough + r_peak * np.cos(theta_mid),
                    yc_rough + r_peak * np.sin(theta_mid)
                ])

        # Least squares fitting of the points to a circle to find the precise center
        pts_array = np.array(pts_list)
        if len(pts_array) > 10:
            x_b = pts_array[:, 0]
            y_b = pts_array[:, 1]
            w = x_b**2 + y_b**2
            A_mat = np.c_[x_b, y_b, np.ones_like(x_b)]
            C_fit, _, _, _ = np.linalg.lstsq(A_mat, w, rcond=None)
            xc = C_fit[0] / 2.0
            yc = C_fit[1] / 2.0
        else:
            xc, yc = xc_rough, yc_rough

        # 4. Coordinate Transformation (Cartesian to Polar)
        y, x = np.indices(img_clean.shape, dtype=np.float32)
        r_grid = np.sqrt((x - xc)**2 + (y - yc)**2)

        # 5. Global Radial Intensity Distribution
        r_flat = r_grid.ravel()
        i_flat = img_clean.ravel()

        r_int_global = np.round(r_flat).astype(int)
        bincounts = np.bincount(r_int_global)
        intensity_sums = np.bincount(r_int_global, weights=i_flat)

        radial_profile = np.zeros_like(bincounts, dtype=float)
        valid_bins = bincounts > 0
        radial_profile[
            valid_bins] = intensity_sums[valid_bins] / bincounts[valid_bins]

        smoothed_profile = gaussian_filter1d(radial_profile, sigma=2.0)

        # 6. Calculate Peak Radius and Asymmetric Error Bounds
        peak_idx = np.argmax(smoothed_profile)
        peak_val = smoothed_profile[peak_idx]
        threshold_val = peak_val * PEAK_THRESHOLD_RATIO

        inner_indices = np.where(
            smoothed_profile[:peak_idx] <= threshold_val)[0]
        r_inner_bound = inner_indices[-1] if len(inner_indices) > 0 else 0

        outer_indices = np.where(
            smoothed_profile[peak_idx:] <= threshold_val)[0]
        r_outer_bound = outer_indices[0] + peak_idx if len(
            outer_indices) > 0 else len(smoothed_profile) - 1

        # Use maximum intensity position as the actual radius
        real_r_px = float(peak_idx)

        # Calculate asymmetric errors
        err_lower_px = real_r_px - r_inner_bound
        err_upper_px = r_outer_bound - real_r_px

        cache_data = {
            'version': 4,
            'threshold_ratio': PEAK_THRESHOLD_RATIO,
            'xc': float(xc),
            'yc': float(yc),
            'r_inner_bound': float(r_inner_bound),
            'r_outer_bound': float(r_outer_bound),
            'real_r_px': float(real_r_px),
            'err_lower_px': float(err_lower_px),
            'err_upper_px': float(err_upper_px)
        }
        with open(cache_path, 'w') as f:
            json.dump(cache_data, f, indent=4)

    # Convert from pixels to physical dimensions
    radius_um = real_r_px * PIXEL_PITCH
    err_lower_um = err_lower_px * PIXEL_PITCH
    err_upper_um = err_upper_px * PIXEL_PITCH

    # ==========================================
    # 7. Generate Diagnostic Verification Plots
    # ==========================================
    if save_diagnostic and basename:
        fig_diag, ax_diag = plt.subplots(figsize=(6, 6))
        im = ax_diag.imshow(img_clean, cmap='viridis', origin='upper')
        plt.colorbar(im,
                     ax=ax_diag,
                     fraction=0.046,
                     pad=0.04,
                     label='Intensity (A.U.)')

        ax_diag.plot(xc,
                     yc,
                     'r+',
                     markersize=10,
                     markeredgewidth=1.0,
                     alpha=0.6,
                     label='Center')

        # Indicate Peak Radius
        circle_mean = Circle((xc, yc),
                             real_r_px,
                             color='red',
                             fill=False,
                             linestyle='-',
                             linewidth=1.2,
                             alpha=0.6,
                             label='Peak Radius')
        ax_diag.add_patch(circle_mean)

        # Indicate Error Bounds
        circle_in = Circle((xc, yc),
                           r_inner_bound,
                           color='orange',
                           fill=False,
                           linestyle='--',
                           linewidth=1.5,
                           alpha=0.8,
                           label=f'{int(PEAK_THRESHOLD_RATIO*100)}% Bounds')
        circle_out = Circle((xc, yc),
                            r_outer_bound,
                            color='orange',
                            fill=False,
                            linestyle='--',
                            linewidth=1.5,
                            alpha=0.8)

        ax_diag.add_patch(circle_in)
        ax_diag.add_patch(circle_out)

        ax_diag.set_title(
            f'Diagnostic: {basename}\nr = {radius_um:.1f} (-{err_lower_um:.1f}/+{err_upper_um:.1f}) $\\mu$m'
        )
        ax_diag.legend(loc='upper right', framealpha=0.8)

        diag_save_path = os.path.join(DIAGNOSTIC_DIR, f'diag_{basename}.png')
        fig_diag.savefig(diag_save_path, dpi=150, bbox_inches='tight')
        plt.close(fig_diag)

    return radius_um, err_lower_um, err_upper_um


# ==========================================
# 3. Main Workflow: Batch Processing & Final Plot
# ==========================================
def main():
    file_pattern = os.path.join(IMAGE_DIR, '*.bmp')
    files = glob.glob(file_pattern)

    if not files:
        print(
            f"No .bmp files found in '{IMAGE_DIR}'. Please check the directory path."
        )
        return

    results = []
    print("Starting image processing and diagnostic plot generation...")
    for f in files:
        basename = os.path.basename(f)
        try:
            z_str = basename.replace('.bmp', '')
            z_cm = float(z_str)
        except ValueError:
            print(
                f"Skipping file {basename}: Cannot parse numeric Z position.")
            continue

        r_um, err_lower, err_upper = extract_radius_and_error(
            f, save_diagnostic=True, basename=z_str, z_cm=z_cm)
        results.append((z_cm, r_um, err_lower, err_upper))
        print(
            f"Processed {basename:^10} | z = {z_cm:4.1f} cm | r = {r_um:6.2f} (-{err_lower:5.2f} / +{err_upper:5.2f}) μm"
        )

    results = np.array(sorted(results, key=lambda x: x[0]))
    z_vals = results[:, 0]

    # Convert units to mm
    r_vals = results[:, 1] / 1000.0
    r_errs_lower = results[:, 2] / 1000.0
    r_errs_upper = results[:, 3] / 1000.0

    # Stack asymmetric errors into shape (2, N) for matplotlib errorbar
    r_errs_asym = np.vstack([r_errs_lower, r_errs_upper])

    # --- Calculate Theoretical Curve ---
    z_theory = np.linspace(np.min(z_vals), np.max(z_vals), 200)

    # [NOTE TO USERS]: Change this theoretical equation according to your target.
    # The current equation represents the 'inverse' expansion case:
    r_theory = (1600 / (1 + z_theory / 80.0)) / 1000.0

    # Example for 'linear' case:
    # r_theory = ((1600 - 800) * (z_theory / 80.0) + 800) / 1000.0

    # --- Draw Publication-Quality Plot ---
    fig, ax = plt.subplots(figsize=(6.5, 5.5))

    # 1. Plot experimental data (Scatter + Error bars)
    ax.errorbar(z_vals,
                r_vals,
                yerr=r_errs_asym,
                fmt='o',
                color='#1f77b4',
                ecolor='black',
                elinewidth=2.5,
                capsize=4,
                capthick=2.5,
                markersize=8,
                markerfacecolor='white',
                markeredgewidth=2,
                label='Experiment',
                zorder=2)

    # 2. Plot theoretical curve (Smooth solid line)
    ax.plot(z_theory,
            r_theory,
            color='#d62728',
            linewidth=3,
            linestyle='-',
            label='Theory',
            zorder=1)

    LABEL_SIZE = 34
    TICK_SIZE = 30
    LEGEND_SIZE = 26

    ax.set_xlabel('$z$ (cm)', fontsize=LABEL_SIZE)
    ax.set_ylabel('Radius (mm)', fontsize=LABEL_SIZE)

    ax.tick_params(axis='both',
                   which='major',
                   labelsize=TICK_SIZE,
                   length=8,
                   width=2)

    ax.set_xticks([20, 40, 60])
    ax.set_yticks([0.8, 1.2, 1.6])

    ax.spines['top'].set_linewidth(3.0)
    ax.spines['right'].set_linewidth(3.0)
    ax.spines['bottom'].set_linewidth(3.0)
    ax.spines['left'].set_linewidth(3.0)

    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(frameon=False, fontsize=LEGEND_SIZE)

    plt.tight_layout()
    plt.savefig('beam_radius_propagation.pdf', format='pdf', transparent=True)
    plt.show()

    print("Processing complete! Plot saved as 'beam_radius_propagation.pdf'")
    print(
        f"Diagnostic plots are available in: {os.path.abspath(DIAGNOSTIC_DIR)}"
    )


if __name__ == '__main__':
    main()
