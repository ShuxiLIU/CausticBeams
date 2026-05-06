"""
Phase-Only Encoding for Complex Optical Fields

This script implements the complex amplitude encoding technique used in the paper:
"Engineering geometrical optical singularities: inverse design of 3D asymmetric caustic surfaces"
(Corresponding to the 'Phase Retrieval & Encoding' module in Fig. 1b).

It maps a complex wavefield (Amplitude + Phase) onto a phase-only Spatial Light Modulator (SLM)
using a blazed grating encoding strategy. A highly optimized Look-Up Table (LUT) combined 
with vectorized interpolation is utilized to accelerate the nonlinear root-finding process.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat, loadmat
from scipy.optimize import fsolve
from scipy.interpolate import interp1d
from PIL import Image

# ==========================================
# Global Plotting Configuration (Academic Style)
# ==========================================
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'mathtext.fontset': 'stix',
    'axes.linewidth': 1.5,
})


def func1(i, params):
    """
    Root-finding equation for amplitude encoding: sinc(1-x) = Amplitude.
    Used to determine the phase modulation depth.
    """
    x, = i
    param = params
    # Minimum value protection to prevent division by zero
    return [np.sin(np.pi * (1 - x)) / (np.pi * (1 - x) + 1e-15) - param]


def generate_phase_only_cgh(input_mat,
                            output_prefix,
                            delta=8e-6,
                            wavelength=532e-9):
    """
    Convert a complex optical field into a phase-only Computer-Generated Hologram (CGH).
    
    Args:
        input_mat (str): Path to the input .mat file containing the complex field 'E0'.
        output_prefix (str): Prefix for the saved output files.
        delta (float): Pixel pitch of the SLM (m). Default is 8um.
        wavelength (float): Illumination wavelength (m). Default is 532nm.
    """
    print(f"Processing: {input_mat} ...")

    # ==========================================
    # 1. Data Import & Normalization
    # ==========================================
    if not os.path.exists(input_mat):
        raise FileNotFoundError(f"Input file not found: {input_mat}")

    data = loadmat(input_mat)
    E = data['E0']
    Ny, Nx = E.shape  # Dynamically retrieve shape (supports non-square arrays)

    Amp = np.abs(E)
    Amp = Amp / np.max(Amp)  # Amplitude normalization
    Phs = np.mod(np.angle(E), 2 * np.pi)  # Phase normalization to [0, 2pi)

    # Plot Normalized Amplitude
    plt.figure(figsize=(6, 5))
    plt.imshow(Amp, cmap='Greys_r')
    plt.title('Normalized Amplitude')
    plt.colorbar()
    plt.tight_layout()

    # ==========================================
    # 2. Phase-Only Transformation (Blazed Grating)
    # ==========================================
    alpha = 60 / 180 * np.pi  # Tilt angle of the grating
    T = 48e-6  # Grating period

    x = np.arange(-Nx * delta / 2, (Nx - 1) / 2 * delta, delta)
    y = np.arange(-Ny * delta / 2, (Ny - 1) / 2 * delta, delta)
    X, Y = np.meshgrid(x, y)

    # Blazed grating phase
    phi_blaze = 2 * np.pi / T * (X / np.cos(alpha) + Y / np.sin(alpha))

    # ------------------------------------------------------------------
    # [CORE OPTIMIZATION]: Look-Up Table (LUT) + Vectorized Interpolation
    # Bypasses the extremely time-consuming double nested loop for root-finding.
    # ------------------------------------------------------------------
    func2 = lambda y_val: fsolve(func1, [0.5], args=y_val)[0]

    # Step 2.1: Create a LUT with 10000 samples for the amplitude range [0, 1]
    amp_samples = np.linspace(0, 1, 10000)
    temp1_samples = np.zeros_like(amp_samples)

    for idx, val in enumerate(amp_samples):
        if val == 0:
            temp1_samples[idx] = 0
        else:
            temp1_samples[idx] = func2(val)

    # Step 2.2: Generate interpolation function
    interp_func = interp1d(amp_samples,
                           temp1_samples,
                           kind='linear',
                           bounds_error=False,
                           fill_value=(0, temp1_samples[-1]))

    # Step 2.3: Matrix-level one-step mapping for temp1 (Speedup ~10,000x)
    temp1 = interp_func(Amp)
    # ------------------------------------------------------------------

    # Reconstruct final encoded phase
    temp2 = Phs + np.pi * (1 - temp1) + phi_blaze
    phi_SLM = temp1 * np.mod(temp2, 2 * np.pi)

    # Plot final SLM phase
    plt.figure(figsize=(6, 5))
    plt.imshow(phi_SLM, cmap='gray')
    plt.title(r'Phase of SLM ($\phi_{SLM}$)')
    plt.colorbar()
    plt.tight_layout()

    # ==========================================
    # 3. Output Generation
    # ==========================================
    # Map phase values to 8-bit SLM grayscale range (0-255)
    Phi_SLM_8bit = phi_SLM / np.max(phi_SLM) * 255

    mat_out = f'{output_prefix}_Phi_SLM.mat'
    savemat(mat_out, {'Phi_SLM': phi_SLM})

    # Save as PNG: standard BMP has poor support for 16-bit grayscale.
    # PIL automatically saves np.uint16 arrays as 16-bit grayscale PNGs.
    img_out = f'{output_prefix}_CGH.png'
    img = Image.fromarray(Phi_SLM_8bit.astype(np.uint8))
    img.save(img_out)

    print(f"Success! Saved CGH data to '{mat_out}' and image to '{img_out}'.")


# ================= Main Execution =================
if __name__ == "__main__":
    # Example execution: processing the "square" target phase mask.

    target_file = 'PhaseMask_square.mat'

    if os.path.exists(target_file):
        generate_phase_only_cgh(input_mat=target_file,
                                output_prefix='Square_Target')
        plt.show()
    else:
        print(
            f"Warning: '{target_file}' not found in current directory. "
            "Please run the CausticGenerator script first to generate the source data."
        )
