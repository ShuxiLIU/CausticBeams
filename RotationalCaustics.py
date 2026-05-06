"""
Inverse Design of Rotationally Symmetric Caustics

This script implements the analytical inverse design and free-space propagation 
simulation for rotationally symmetric caustic surfaces, corresponding to Figure 2 
in the paper: 
"Engineering geometrical optical singularities: inverse design of 3D asymmetric caustic surfaces"

It reproduces the generalized mapping relation featuring a transverse rotation matrix
and calculates the effective topological charge (OAM spectrum) along the propagation.
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy import exp, sqrt, cos, sin, pi
from matplotlib.colors import hsv_to_rgb
from matplotlib.ticker import MaxNLocator, MultipleLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.fft import fftshift, ifft2, fft2
from scipy.interpolate import interp1d
from scipy.ndimage import map_coordinates
from scipy.io import savemat


class CausticGenerator:
    """
    A class to generate, simulate, and visualize rotationally symmetric caustic beams.
    """

    def __init__(self, N=1001, width=4e-3, wavelength=532e-9):
        """
        Initialize the computational grid and physical parameters.
        
        Args:
            N (int): Number of grid points along one dimension.
            width (float): Half-width of the source plane (m).
            wavelength (float): Illumination wavelength (m).
        """
        self.N = N
        self.width = width
        self.wl = wavelength
        self.k = 2 * np.pi / wavelength

        x = np.linspace(-width, width, N)
        y = np.linspace(-width, width, N)
        self.X, self.Y = np.meshgrid(x, y)
        self.r_grid = np.sqrt(self.X**2 + self.Y**2)
        self.theta_grid = np.angle(self.X + 1j * self.Y)

        self.phi = None
        self.E0 = None
        self.data = {}

    def solve_and_generate(self, case_type):
        """
        Analytically solve the mapping functions and generate the source phase mask.
        
        Args:
            case_type (str): The expansion law of the caustic radius 
                             ('linear', 'square', or 'inverse').
        """
        # 1. Initialize longitudinal propagation axis (Z-axis)
        Z = np.linspace(0.001, 0.8, 100000)
        deltaZ = Z[1] - Z[0]

        # 2. Define expansion profiles and target parameters
        if case_type == 'linear':
            W0 = 3.81e-3
            Z0 = Z[-1]
            f_init = 0.8e-3
            f_end = 1.6e-3
            cal_range = 4e-3
            f = (f_end - f_init) * (Z / Z0) + f_init
        elif case_type == 'square':
            W0 = 3.936e-3
            Z0 = Z[-1]
            f_init = 0.8e-3
            f_end = 1.6e-3
            cal_range = 4e-3
            f = (f_end - f_init) * (Z / Z0)**2 + f_init
        elif case_type == 'inverse':
            W0 = 3.7254e-3
            Z0 = Z[-1]
            f_init = 1.6e-3
            f_end = 0.8e-3
            delta_Z = f_end * Z0 / (f_init - f_end)
            cal_range = 4e-3
            f = f_init * delta_Z / (Z + delta_Z)
        else:
            raise ValueError(
                "Unknown target trajectory type. Choose 'linear', 'square', or 'inverse'."
            )

        # 3. Solve auxiliary functions F, W, and mapping radius R
        F = f - Z * np.gradient(f) / deltaZ
        integrand = F / Z / f
        tempW = np.cumsum(integrand) * deltaZ
        idx_z0 = np.argmin(np.abs(Z - Z0))
        W = W0 * exp(tempW - tempW[idx_z0])
        R = sqrt(F**2 + W**2)

        # Truncate invalid mapping regions (before the caustic forms or out of bounds)
        cut1 = np.argmin(R)
        cut2 = len(R) - 1

        Z_v = Z[cut1:cut2]
        R_v = R[cut1:cut2]
        f_v = f[cut1:cut2]
        F_v = F[cut1:cut2]
        W_v = W[cut1:cut2]

        # 5. Calculate phase gradients (Radial and Azimuthal components)
        grad_R = np.gradient(R_v)
        func_r = self.k * (f_v * F_v / R_v - R_v) / Z_v * grad_R
        sum_r = np.cumsum(func_r)
        sum_r = sum_r - sum_r[0]

        func_l = self.k * W_v * f_v / Z_v

        # 6. Map the 1D analytical solutions onto the 2D source plane
        sort_idx = np.argsort(R_v)
        R_sorted = R_v[sort_idx]
        sum_r_sorted = sum_r[sort_idx]
        func_l_sorted = func_l[sort_idx]

        interp_radial = interp1d(R_sorted,
                                 sum_r_sorted,
                                 kind='linear',
                                 bounds_error=False,
                                 fill_value=0)
        interp_l = interp1d(R_sorted,
                            func_l_sorted,
                            kind='linear',
                            bounds_error=False,
                            fill_value=0)

        phi_radial = interp_radial(self.r_grid)
        l_eff = interp_l(self.r_grid)
        phi_azimuthal = l_eff * self.theta_grid

        self.phi = phi_radial + phi_azimuthal

        # Apply aperture mask
        R_min, R_max = np.min(R_v), np.max(R_v)
        mask = (self.r_grid >= R_min) & (self.r_grid <= R_max)

        self.phi[~mask] = 0
        self.E0 = np.exp(1j * self.phi)
        self.E0[~mask] = 0

        # Save Phase Mask for experimental encoding (SLM)
        savemat(f"PhaseMask_{case_type}.mat", {'E0': self.E0})

        self.data = {
            'Z_v': Z_v,
            'R_v': R_v,
            'f_v': f_v,
            'Z_full': Z,
            'f_full': f,
            'F_full': F,
            'W_full': W,
            'R_full': R,
            'cal_range': cal_range,
            'case': case_type,
            'l_theory_array': func_l
        }

    def propagate_angular_spectrum(self, z):
        """
        Simulate free-space propagation using the Angular Spectrum Method (ASM).
        """
        fx = np.fft.fftfreq(self.N, 2 * self.width / self.N)
        fy = np.fft.fftfreq(self.N, 2 * self.width / self.N)
        FX, FY = np.meshgrid(fx, fy)
        H = np.exp(1j * self.k * z * np.sqrt(1 - (self.wl * FX)**2 -
                                             (self.wl * FY)**2))
        return ifft2(fft2(self.E0) * H)

    def locate_main_lobe(self, Ez):
        """
        Helper function: Locate the main lobe radius of the caustic profile.
        """
        I = np.abs(Ez)**2
        center = self.N // 2
        y, x = np.indices(I.shape)
        r_grid = np.sqrt((x - center)**2 + (y - center)**2)
        r_int = r_grid.astype(int)

        tbin = np.bincount(r_int.ravel(), I.ravel())
        nr = np.bincount(r_int.ravel())
        radial_profile = np.zeros_like(tbin)
        radial_profile[nr > 0] = tbin[nr > 0] / nr[nr > 0]

        min_search = int(0.02 * self.N)
        if len(radial_profile) <= min_search: return 0.0, 0, (0, 0)

        r_peak_pixel = np.argmax(radial_profile[min_search:]) + min_search

        # Generate sampling coordinates for visualization (dashed white ring)
        theta = np.linspace(0, 2 * np.pi, 2048, endpoint=False)
        x_sample = center + r_peak_pixel * np.cos(theta)
        y_sample = center + r_peak_pixel * np.sin(theta)

        return r_peak_pixel, (x_sample, y_sample)

    def measure_global_oam_spectrum(self, Ez):
        """
        Measure the Orbital Angular Momentum (OAM) spectrum of the optical field
        via polar coordinate transformation and Fourier analysis.
        """
        center = self.N // 2
        num_r = self.N // 2
        num_theta = 2048

        r_space = np.linspace(0, self.width, num_r)
        theta_space = np.linspace(0, 2 * np.pi, num_theta, endpoint=False)
        R_pol, T_pol = np.meshgrid(r_space, theta_space)
        X_pol = R_pol * np.cos(T_pol)
        Y_pol = R_pol * np.sin(T_pol)

        dx = 2 * self.width / self.N
        x_pix = center + X_pol / dx
        y_pix = center + Y_pol / dx

        real_part = map_coordinates(np.real(Ez), [y_pix, x_pix],
                                    order=1,
                                    mode='constant',
                                    cval=0)
        imag_part = map_coordinates(np.imag(Ez), [y_pix, x_pix],
                                    order=1,
                                    mode='constant',
                                    cval=0)
        E_polar = real_part + 1j * imag_part

        fft_vals = np.fft.fft(E_polar, axis=0) / num_theta
        power_l_r = np.abs(fft_vals)**2
        total_power = np.sum(power_l_r * r_space[None, :], axis=1)

        freqs = np.fft.fftfreq(num_theta) * num_theta
        freqs_shifted = np.fft.fftshift(freqs)
        power_shifted = np.fft.fftshift(total_power)

        max_idx = np.argmax(power_shifted)
        peak_l = freqs_shifted[max_idx]

        if peak_l < 0:
            freqs_shifted = -freqs_shifted
            sort_order = np.argsort(freqs_shifted)
            freqs_shifted = freqs_shifted[sort_order]
            power_shifted = power_shifted[sort_order]
            peak_l = -peak_l

        new_max_idx = np.argmax(power_shifted)

        # Calculate expected OAM value within a narrow window
        windowleft = 30
        windowright = 30
        s = max(0, new_max_idx - windowleft)
        e = min(len(power_shifted), new_max_idx + windowright + 1)

        win_freqs = freqs_shifted[s:e]
        win_power = power_shifted[s:e]
        l_expectation = np.sum(win_freqs * win_power) / np.sum(win_power)

        # Display window limits
        windowleft = 2
        windowright = 3
        s = max(0, new_max_idx - windowleft)
        e = min(len(power_shifted), new_max_idx + windowright + 1)

        win_freqs = freqs_shifted[s:e]
        win_power = power_shifted[s:e]

        return l_expectation, (win_freqs, win_power)

    def export_plots_for_illustrator(self, prefix):
        """
        Export high-quality figures suitable for publication.
        """
        d = self.data
        case_name = d['case']

        # ==========================================
        # Global Plotting Configuration (Publication Quality)
        # ==========================================
        plt.rcParams['pdf.fonttype'] = 42
        plt.rcParams['ps.fonttype'] = 42

        FS_TITLE = 20
        FS_LABEL = 18
        FS_TICK = 16
        FS_LEGEND = 14
        LW_AXIS = 2.5
        LW_PLOT = 2.5

        plt.rcParams.update({
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial"],
            "font.weight": "normal",
            "axes.labelweight": "normal",
            "axes.linewidth": LW_AXIS,
            "xtick.major.width": LW_AXIS,
            "ytick.major.width": LW_AXIS,
            "xtick.major.size": 6,
            "ytick.major.size": 6,
        })

        # ==========================================
        # Plot 1: Design Parameters Evolution
        # ==========================================
        plt.figure(figsize=(5.5, 3.8))

        plt.plot(d['Z_full'],
                 d['f_full'] * 1e3,
                 label='Beam width $f$',
                 linestyle='--',
                 color='gray',
                 alpha=0.8,
                 linewidth=LW_PLOT)
        plt.plot(d['Z_full'],
                 d['F_full'] * 1e3,
                 label='Component $F$',
                 linestyle=':',
                 color='gray',
                 alpha=0.8,
                 linewidth=LW_PLOT)
        plt.plot(d['Z_full'],
                 d['W_full'] * 1e3,
                 label='Component $W$',
                 linestyle='-.',
                 color='gray',
                 alpha=0.8,
                 linewidth=LW_PLOT)
        plt.plot(d['Z_full'],
                 d['R_full'] * 1e3,
                 label='Mapping Radius $R$',
                 color='black',
                 linewidth=3.5,
                 zorder=10)

        z_valid_start, z_valid_end = d['Z_v'][0], d['Z_v'][-1]

        plt.axvspan(z_valid_start,
                    z_valid_end,
                    color='orange',
                    alpha=0.05,
                    label='Design Region')
        plt.axhline(self.width * 1e3,
                    color='r',
                    linestyle='--',
                    label=f'Aperture ({self.width*1e3:.0f} mm)',
                    linewidth=LW_PLOT)

        plt.xlabel('$z$ (m)', fontsize=FS_LABEL)
        plt.ylabel('Value (mm)', fontsize=FS_LABEL)
        plt.ylim([-0.5, 4.5])

        ax = plt.gca()
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.tick_params(axis='both', which='major', labelsize=FS_TICK, pad=6)

        handles, labels = ax.get_legend_handles_labels()
        order = [3, 0, 1, 2, 5, 4]

        plt.legend([handles[idx] for idx in order],
                   [labels[idx] for idx in order],
                   loc='upper left',
                   shadow=False,
                   frameon=True,
                   fontsize=FS_LEGEND,
                   prop={
                       'weight': 'normal',
                       'size': FS_LEGEND
                   },
                   handlelength=1.5,
                   borderpad=0.4,
                   labelspacing=0.3)

        plt.grid(True, linestyle=':', alpha=0.6, linewidth=1.5)
        plt.savefig(f"{prefix}_{case_name}_Parameters.pdf",
                    dpi=300,
                    bbox_inches='tight',
                    transparent=True)
        plt.close()

        # ==========================================
        # Plot 2: Source Plane Phase Mask
        # ==========================================
        fig, ax = plt.subplots(figsize=(4.2, 3.5))
        phase_plot = np.angle(self.E0)
        phase_plot[np.abs(self.E0) == 0] = np.nan

        im = ax.imshow(phase_plot,
                       cmap='hsv',
                       extent=[
                           -self.width * 1e3, self.width * 1e3,
                           -self.width * 1e3, self.width * 1e3
                       ])

        ax.set_xlabel('$x_0$ (mm)', fontsize=FS_LABEL)
        ax.set_ylabel('$y_0$ (mm)', fontsize=FS_LABEL)
        ax.tick_params(axis='both', labelsize=FS_TICK)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3))

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("left", size="6%", pad=0.9)
        cbar = plt.colorbar(im, cax=cax)
        cbar.ax.yaxis.set_ticks_position('left')
        cbar.ax.yaxis.set_label_position('left')
        cbar.set_label('Phase (rad)',
                       fontsize=FS_LABEL,
                       weight='normal',
                       labelpad=5)
        cbar.set_ticks([-np.pi, 0, np.pi])
        cbar.set_ticklabels([r'$-\pi$', r'$0$', r'$\pi$'])
        cbar.ax.tick_params(labelsize=FS_TICK)

        plt.savefig(f"{prefix}_{case_name}_Phase.pdf",
                    dpi=300,
                    bbox_inches='tight',
                    transparent=True)
        plt.close()

        # ==========================================
        # Plot 3: Longitudinal Intensity Evolution
        # ==========================================
        z_scan = np.linspace(d['Z_v'][0], d['Z_v'][-1], 200)
        xz_plane = []
        center_idx = self.N // 2
        for z in z_scan:
            Ez = self.propagate_angular_spectrum(z)
            xz_plane.append(np.abs(Ez[center_idx, :])**2)
        xz_plane = np.array(xz_plane).T

        plt.figure(figsize=(5.5, 2.8))
        plt.imshow(xz_plane,
                   aspect='auto',
                   cmap='inferno',
                   extent=[
                       z_scan[0] * 1e2, z_scan[-1] * 1e2, -self.width * 1e3,
                       self.width * 1e3
                   ],
                   origin='lower',
                   vmin=0,
                   vmax=np.percentile(xz_plane, 99.5))

        plt.plot(d['Z_v'] * 1e2,
                 d['f_v'] * 1e3,
                 color='cyan',
                 linestyle='--',
                 linewidth=2.5,
                 alpha=0.9)
        plt.plot(d['Z_v'] * 1e2,
                 -d['f_v'] * 1e3,
                 color='cyan',
                 linestyle='--',
                 linewidth=2.5,
                 alpha=0.9)

        plt.xlabel('$z$ (cm)', fontsize=FS_LABEL)
        plt.ylabel('$x$ (mm)', fontsize=FS_LABEL)
        ax = plt.gca()
        ax.tick_params(labelsize=FS_TICK)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3))

        plt.savefig(f"{prefix}_{case_name}_Longitudinal.pdf",
                    dpi=300,
                    bbox_inches='tight',
                    transparent=True)
        plt.close()

        # ==========================================
        # Plot 4 & 5: Transverse Slices & OAM Spectra
        # ==========================================
        z_slices = [0.2, 0.4, 0.6]
        slice_names = ['Start', 'Mid', 'End']
        disp_w = d['cal_range'] * 1e3

        print(f"\n--- Processing Transverse Slices for [{case_name}] ---")

        for i, z in enumerate(z_slices):
            Ez = self.propagate_angular_spectrum(z)
            l_sim, spec_data = self.measure_global_oam_spectrum(Ez)
            win_freqs, win_power = spec_data

            I = np.abs(Ez)**2
            phi = np.angle(Ez)
            I_norm = np.clip((I / np.max(I))**0.5, 0, 1)
            hsv_img = np.zeros((self.N, self.N, 3))
            hsv_img[:, :, 0] = (phi + np.pi) / (2 * np.pi)
            hsv_img[:, :, 1] = 1.0
            hsv_img[:, :, 2] = I_norm
            rgb_img = hsv_to_rgb(hsv_img)

            fig_w = 5
            fig_h = 3.0
            fig = plt.figure(figsize=(fig_w, fig_h))

            # Transverse Field Main Image
            ax_main = fig.add_axes([0, 0, 3.0 / fig_w, 1])
            ax_main.imshow(rgb_img,
                           extent=[
                               -self.width * 1e3, self.width * 1e3,
                               -self.width * 1e3, self.width * 1e3
                           ])

            r_pix, coords = self.locate_main_lobe(Ez)
            x_plot = (coords[0] - self.N / 2) * (2 * self.width / self.N) * 1e3
            y_plot = (coords[1] - self.N / 2) * (2 * self.width / self.N) * 1e3
            ax_main.plot(x_plot, y_plot, 'w--', linewidth=1.5, alpha=0.8)

            ax_main.set_xlim([-disp_w, disp_w])
            ax_main.set_ylim([-disp_w, disp_w])
            ax_main.axis('off')

            # 2D Colorbar for Phase-Intensity Composite
            bar_left = 3.25 / fig_w
            bar_width = 0.42 / fig_w
            ax_bar = fig.add_axes([bar_left, 0, bar_width, 1])

            res_y, res_x = 200, 20
            YY, XX = np.mgrid[-np.pi:np.pi:complex(0, res_y),
                              0:1:complex(0, res_x)]
            H_bar = (YY + np.pi) / (2 * np.pi)
            S_bar = np.ones_like(YY)
            V_bar = XX
            hsv_bar = np.zeros((res_y, res_x, 3))
            hsv_bar[:, :, 0] = H_bar
            hsv_bar[:, :, 1] = S_bar
            hsv_bar[:, :, 2] = V_bar
            rgb_bar = hsv_to_rgb(hsv_bar)

            ax_bar.imshow(rgb_bar,
                          origin='lower',
                          extent=[0, 1, -np.pi, np.pi],
                          aspect='auto')
            ax_bar.yaxis.tick_right()
            ax_bar.yaxis.set_label_position("right")
            ax_bar.set_yticks([-np.pi, 0, np.pi])
            ax_bar.set_yticklabels([r'$-\pi$', r'$0$', r'$\pi$'],
                                   fontsize=FS_TICK,
                                   weight='normal')
            ax_bar.set_ylabel('Phase',
                              rotation=270,
                              labelpad=10,
                              fontsize=FS_LABEL,
                              weight='normal')

            ax_bar.set_xticks([0, 1])
            ax_bar.set_xticklabels(['0', '1'],
                                   fontsize=FS_TICK,
                                   weight='normal')
            ax_bar.set_xlabel('Amp', fontsize=FS_LABEL, weight='normal')

            plt.savefig(
                f"{prefix}_{case_name}_Slice_{slice_names[i]}_Composite.pdf",
                dpi=300,
                bbox_inches='tight',
                transparent=True)
            plt.close()

            # OAM Spectrum Plot
            if len(win_freqs) > 0:
                plt.figure(figsize=(3.0, 1.8))
                power_norm = win_power / np.max(win_power)
                plt.bar(win_freqs,
                        power_norm,
                        width=0.6,
                        color='royalblue',
                        edgecolor='navy',
                        alpha=0.8,
                        linewidth=1.5)
                plt.axvline(l_sim, color='red', linestyle='--', linewidth=2.5)

                plt.xlabel('Topo. Charge $l$', fontsize=FS_LABEL)
                plt.ylabel('Norm. Power', fontsize=FS_LABEL)
                plt.ylim([0, 1.1])

                ax = plt.gca()
                ax.tick_params(labelsize=FS_TICK)
                ax.xaxis.set_major_locator(MultipleLocator(1))
                ax.yaxis.set_major_locator(MultipleLocator(0.5))

                plt.text(0.99,
                         0.9,
                         f'$\\bar{{l}} \\approx {l_sim:.2f}$',
                         transform=ax.transAxes,
                         ha='right',
                         va='top',
                         color='red',
                         fontsize=FS_LABEL,
                         weight='normal')

                plt.savefig(
                    f"{prefix}_{case_name}_Slice_{slice_names[i]}_Spectrum.pdf",
                    dpi=300,
                    bbox_inches='tight',
                    transparent=True)
                plt.close()

            print(
                f"  Slice {slice_names[i]}: Expectation l={l_sim:.4f} saved.")


# ================= Main Execution =================
if __name__ == "__main__":
    # Initialize the generator with parameters defined in the paper
    sim = CausticGenerator(N=1001, width=4e-3)

    # Example runs. Uncomment 'linear' or 'square' to reproduce Fig. 2(a) and (b).
    sim.solve_and_generate('linear')
    # sim.solve_and_generate('square')

    # Reproduces the inverse expansion target shown in Fig. 2(c-f)
    # sim.solve_and_generate('inverse')
    sim.export_plots_for_illustrator('Fig')
