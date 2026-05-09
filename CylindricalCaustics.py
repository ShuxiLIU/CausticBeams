"""
Numerical Validation of Z-Invariant Cylindrical Caustics

This script corresponds to Section 3.B of the Supplemental Document for the paper:
"Engineering geometrical optical singularities: inverse design of 3D asymmetric caustic surfaces"

It validates the analytical mapping solutions for z-invariant cylindrical caustics
(non-diffracting beams), focusing specifically on:
1. Circular Cylindrical Caustic (Higher-order Bessel-like, Eq. S32)
2. Parabolic Translating Surface (Eq. S33)

Key Implementations:
- Exact 2D parameterization mapping (propagation distance 'u' & transverse parameter 'v').
- Strict topological charge quantization to prevent fractional vortex splitting.
- High-precision Fresnel matrix multiplication for propagation to overcome sampling limits.
- Exports publication-quality figures (Fig. S2).
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from abc import ABC, abstractmethod
from scipy.interpolate import griddata
from scipy.fft import fft2, ifft2, fftfreq
from scipy.ndimage import binary_erosion
import matplotlib as mpl
from matplotlib.path import Path
from mpl_toolkits.mplot3d.art3d import Line3DCollection

# ==========================================
# Global Academic Plotting Settings
# ==========================================
mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'axes.linewidth': 0.8
})


# ==========================================
# 1. Target Definition Base Class (Cylindrical Caustic Mapping)
# ==========================================
class CylindricalCaustic(ABC):
    """
    Base class for cylindrical caustics, dual-parameterized by the 
    propagation distance 'u' and the transverse parameter 'v'.
    """

    def __init__(self,
                 name="Generic",
                 u_range=(0.0, 0.8),
                 v_range=(0, 2 * np.pi)):
        self.name = name
        self.u_range = u_range
        self.v_range = v_range
        self.has_hole = False

    @abstractmethod
    def parameterization(self, u, v, k):
        """
        Returns: Xc, Yc, Zc (Target 3D curve), X, Y (Source Mapping), Phi (Analytical Phase)
        """
        pass


class CircularCylindrical(CylindricalCaustic):
    """
    Circular Cylindrical Surface (Generates higher-order Bessel-like beams).
    (Refer to Eq. S32 in the Supplemental Document)
    """

    def __init__(self):
        super().__init__(name="Circular Cylindrical",
                         u_range=(0.0, 0.8),
                         v_range=(0, 2 * np.pi))
        self.has_hole = True  # A cylinder maps to a hollow annulus on the source plane

    def parameterization(self, u, v, k):
        R = 1e-3
        C_theory = 4e-3

        # [PHYSICS PRESERVATION] Force quantize the topological charge 'm' to an integer.
        # This prevents beam splitting/instability caused by fractional vortices during propagation.
        m = np.round(-k * C_theory * R)
        C = -m / (
            k * R
        )  # Dynamically fine-tune C to strictly satisfy physical continuity

        # Theoretical Caustic Surface: Cylinder
        Xc = R * np.cos(v)
        Yc = R * np.sin(v)
        Zc = u

        # Analytical Mapping Grid
        X = R * np.cos(v) - C * u * np.sin(v)
        Y = R * np.sin(v) + C * u * np.cos(v)

        # Analytical Phase (Vortex term + Axicon term)
        Phi = m * v - k * (C**2) * u

        return Xc, Yc, Zc, X, Y, Phi


class ParabolicCylindrical(CylindricalCaustic):
    """
    Parabolic Translating Cylindrical Surface.
    (Refer to Eq. S33 in the Supplemental Document)
    """

    def __init__(self):
        super().__init__(name="Parabolic Cylindrical",
                         u_range=(0.0, 0.8),
                         v_range=(-1e-3, 1e-3))
        self.has_hole = False

    def parameterization(self, u, v, k):
        a = 1e3
        C = 4e-3

        # Theoretical Caustic Surface: Parabolic Cylinder
        Xc = a * v**2
        Yc = v
        Zc = u

        denom = np.sqrt(1 + 4 * a**2 * v**2)
        X = a * v**2 + C * u * (2 * a * v) / denom
        Y = v + C * u * 1.0 / denom

        # Phase Integration: int sqrt(1 + 4a^2 xi^2) dxi
        A = 2 * a

        # Antiderivative of the integral
        def integral_func(xi):
            return 0.5 * xi * np.sqrt(1 + A**2 * xi**2) + (
                1 / (2 * A)) * np.log(A * xi + np.sqrt(1 + A**2 * xi**2))

        integral_val = integral_func(v) - integral_func(self.v_range[0])

        # Analytical Phase
        Phi = -k * C * integral_val - k * (C**2) * u

        return Xc, Yc, Zc, X, Y, Phi


# ==========================================
# 2. Analytical Mapper Core
# ==========================================
class AnalyticalMapper:

    def __init__(self,
                 target: CylindricalCaustic,
                 wl,
                 N_grid=1024,
                 bounds_mm=4.0):
        self.target = target
        self.N = N_grid
        self.bounds = bounds_mm * 1e-3
        self.k = 2 * np.pi / wl

        u_min, u_max = target.u_range
        v_min, v_max = target.v_range
        self.u_vec = np.linspace(u_min, u_max, self.N)
        self.v_vec = np.linspace(v_min, v_max, self.N)
        self.U, self.V = np.meshgrid(self.u_vec, self.v_vec, indexing='ij')

        print(
            f"[*] Generating analytical cylindrical mapping for [{target.name}]..."
        )
        self.Xc, self.Yc, self.Zc, self.X_sol, self.Y_sol, self.Phi_map = self.target.parameterization(
            self.U, self.V, self.k)


# ==========================================
# 3. Phase Interpolation & Wavefront Propagation
# ==========================================
class WavePropagator:

    def __init__(self, mapper: AnalyticalMapper, wl=532e-9, N_sim=1024):
        self.opt = mapper
        self.L = mapper.bounds
        self.N_sim = N_sim
        self.dx = 2 * self.L / N_sim
        self.wl = wl
        self.k = 2 * np.pi / wl

    def interpolate_and_mask(self):
        """
        Applies linear interpolation on the real phase array and utilizes rigorous topological masking.
        Relies on dense mapping grids to ensure precision.
        """
        print("    -> Generating topological mask and interpolating phase...")
        x = np.linspace(-self.L, self.L, self.N_sim)
        y = np.linspace(-self.L, self.L, self.N_sim)
        self.xx, self.yy = np.meshgrid(x, y)

        X_t, Y_t = self.opt.X_sol, self.opt.Y_sol

        # Extract boundary for masking
        bx_1, by_1 = X_t[0, :], Y_t[0, :]
        bx_2, by_2 = X_t[:, -1], Y_t[:, -1]
        bx_3, by_3 = X_t[-1, ::-1], Y_t[-1, ::-1]
        bx_4, by_4 = X_t[::-1, 0], Y_t[::-1, 0]
        boundary_verts = np.column_stack(
            (np.concatenate([bx_1, bx_2, bx_3,
                             bx_4]), np.concatenate([by_1, by_2, by_3, by_4])))
        path = Path(boundary_verts)

        pts_grid = np.column_stack((self.xx.flatten(), self.yy.flatten()))
        is_inside = path.contains_points(pts_grid)

        # Handle inner hole exclusion for tubular topologies
        if self.opt.target.has_hole:
            path_in = Path(np.column_stack((X_t[0, :], Y_t[0, :])))
            in_in = path_in.contains_points(pts_grid)
            self.mask = (is_inside & ~in_in).reshape(self.N_sim,
                                                     self.N_sim).astype(float)
        else:
            self.mask = is_inside.reshape(self.N_sim, self.N_sim).astype(float)

        points = np.column_stack((X_t.flatten(), Y_t.flatten()))
        values = self.opt.Phi_map.flatten()

        # [NUMERICAL FIX]: Eliminate ~10^-16 floating point truncation errors between v=0 and v=2pi.
        # This prevents duplicate physical coordinates with conflicting phases, which would cause
        # griddata to yield infinite gradients (singularities) along the stitch line.
        points_rounded = np.round(points, decimals=9)
        _, unique_indices = np.unique(points_rounded,
                                      axis=0,
                                      return_index=True)
        points = points[unique_indices]
        values = values[unique_indices]

        # Perform rigorous linear interpolation on the real phase array
        phi_grid = griddata(points,
                            values, (self.xx, self.yy),
                            method='linear',
                            fill_value=0.0)
        self.phi_grid = np.nan_to_num(phi_grid, 0)
        self.E0 = self.mask * np.exp(1j * self.phi_grid * self.mask)

    def propagate(self, z):
        """
        [PROPAGATION UPDATE]: Replaced FFT-based Angular Spectrum Method with discrete 
        Fresnel matrix multiplication to achieve precise control over the high-resolution
        output window without zero-padding overhead.
        """
        # Set calculation observation window to 4mm total width (-2mm to 2mm)
        L_out = 2e-3
        N_out = self.N_sim  # Maintain high grid sampling count for output resolution

        # Source plane coordinates (matching self.E0, -4mm to 4mm)
        x0 = np.linspace(-self.L, self.L, self.N_sim)
        y0 = np.linspace(-self.L, self.L, self.N_sim)

        # Target plane coordinates (2mm window)
        x_out = np.linspace(-L_out, L_out, N_out)
        y_out = np.linspace(-L_out, L_out, N_out)
        XX, YY = np.meshgrid(x_out, y_out)

        # Matrix Construction: M_y @ M_3 @ M_x using physically separated slices
        # M1: Constants and quadratic phase at the target plane
        M1 = np.exp(1j * self.k * z) / (1j * self.wl * z) * np.exp(
            1j * self.k / (2 * z) * (XX**2 + YY**2))

        # My: Fourier kernel operator in y-direction [N_out, N_sim]
        My = np.exp(-1j * self.k / z *
                    np.dot(y_out.reshape(N_out, 1), y0.reshape(1, self.N_sim)))

        # M3: Source field enveloped by initial quadratic phase [N_sim, N_sim]
        M3 = self.E0 * np.exp(1j * self.k / (2 * z) *
                              (self.xx**2 + self.yy**2))

        # Mx: Fourier kernel operator in x-direction [N_sim, N_out]
        Mx = np.exp(-1j * self.k / z *
                    np.dot(x0.reshape(self.N_sim, 1), x_out.reshape(1, N_out)))

        # Match micro-area integration steps
        dx0 = x0[1] - x0[0]
        dy0 = y0[1] - y0[0]

        Ez = M1 * (My @ M3 @ Mx) * (dx0 * dy0)
        return Ez

    def export_publication_figures(self, prefix=""):
        """Exports separated, high-resolution figures for the paper."""
        print(
            f"[*] Exporting high-resolution publication figures for [{prefix}]..."
        )
        script_dir = os.path.dirname(os.path.abspath(__file__))

        def save_fig_tight(fig, suffix):
            filepath = os.path.join(script_dir, f"{prefix}_{suffix}.pdf")
            fig.savefig(filepath,
                        bbox_inches='tight',
                        transparent=True,
                        pad_inches=0.02)
            print(f"      -> Exported: {filepath}")

        # ==================================================
        # 1. Source Plane Mapping Grid
        # ==================================================
        fig_map = plt.figure(figsize=(4, 4))
        ax_map = fig_map.add_axes([0.15, 0.15, 0.75, 0.75])

        skip_m = max(1, self.opt.N // 64)

        sc = ax_map.scatter(self.opt.X_sol[::skip_m, ::skip_m].flatten() * 1e3,
                            self.opt.Y_sol[::skip_m, ::skip_m].flatten() * 1e3,
                            c=self.opt.U[::skip_m, ::skip_m].flatten(),
                            s=1.5,
                            cmap='viridis',
                            alpha=0.8)

        ax_map.plot(self.opt.X_sol[-1, :] * 1e3,
                    self.opt.Y_sol[-1, :] * 1e3,
                    'k-',
                    lw=1.5)
        if self.opt.target.has_hole:
            ax_map.plot(self.opt.X_sol[0, :] * 1e3,
                        self.opt.Y_sol[0, :] * 1e3,
                        'k-',
                        lw=1.5)

        ax_map.set_aspect('equal')
        ax_map.set_xlim([-self.L * 1e3, self.L * 1e3])
        ax_map.set_ylim([-self.L * 1e3, self.L * 1e3])

        ax_map.set_xticks([-4, 0, 4])
        ax_map.set_yticks([-4, 0, 4])
        ax_map.set_xticklabels([])
        ax_map.set_yticklabels([])
        ax_map.tick_params(direction='in', length=4, width=0.8)
        save_fig_tight(fig_map, "1_Mapping")

        # ==================================================
        # 2. Source Phase Map
        # ==================================================
        fig_ph = plt.figure(figsize=(4, 4.5))
        ax_ph = fig_ph.add_axes([0.15, 0.20, 0.75, 0.75 * (4 / 4.5)])

        phase_display = np.mod(self.phi_grid + np.pi, 2 * np.pi) - np.pi
        phase_display[self.mask == 0] = np.nan

        im_ph = ax_ph.imshow(
            phase_display,
            extent=[-self.L * 1e3, self.L * 1e3, -self.L * 1e3, self.L * 1e3],
            cmap='twilight',
            origin='lower',
            vmin=-np.pi,
            vmax=np.pi)

        ax_ph.set_xticks([-4, 0, 4])
        ax_ph.set_yticks([-4, 0, 4])
        ax_ph.set_xticklabels([])
        ax_ph.set_yticklabels([])
        ax_ph.tick_params(direction='in', length=4, width=0.8)

        cax_ph = fig_ph.add_axes([0.15, 0.08, 0.75, 0.04])
        cbar_ph = fig_ph.colorbar(im_ph,
                                  cax=cax_ph,
                                  orientation='horizontal',
                                  ticks=[-np.pi, 0, np.pi])
        cbar_ph.ax.set_xticklabels([r'$-\pi$', r'$0$', r'$\pi$'], fontsize=22)
        cbar_ph.ax.xaxis.set_ticks_position('bottom')
        save_fig_tight(fig_ph, "2_Phase")

        # ==================================================
        # 3. 3D Ray Tracing Visualization
        # ==================================================
        fig_ray = plt.figure(figsize=(4, 4))
        ax_ray = fig_ray.add_subplot(111, projection='3d')

        grad_y, grad_x = np.gradient(self.phi_grid, self.dx, self.dx)

        slope_x = grad_x / self.k
        slope_y = grad_y / self.k

        safe_mask = binary_erosion(self.mask > 0.5, iterations=8)
        physical_mask = (np.abs(slope_x) < 0.5) & (np.abs(slope_y) < 0.5)

        skip_s = self.N_sim // 60
        sample_mask = np.zeros_like(safe_mask)
        sample_mask[::skip_s, ::skip_s] = True

        final_mask = safe_mask & physical_mask & sample_mask

        X_s = self.xx[final_mask]
        Y_s = self.yy[final_mask]
        gx = grad_x[final_mask]
        gy = grad_y[final_mask]

        slope_x = gx / self.k
        slope_y = gy / self.k

        Z_ext = self.opt.target.u_range[1]

        X_ext = X_s + Z_ext * slope_x
        Y_ext = Y_s + Z_ext * slope_y

        segments = [((X_s[i] * 1e3, Y_s[i] * 1e3, 0), (X_ext[i] * 1e3,
                                                       Y_ext[i] * 1e3, Z_ext))
                    for i in range(len(X_s))]
        lc = Line3DCollection(segments,
                              colors="#72bff7",
                              linewidths=0.6,
                              alpha=0.08)
        ax_ray.add_collection3d(lc)

        z_targets = [0.2, 0.4, 0.6]
        for z_t in z_targets:
            v_dense = np.linspace(self.opt.target.v_range[0],
                                  self.opt.target.v_range[1], 500)
            u_dense = np.full_like(v_dense, z_t)
            Xc_th, Yc_th, _, _, _, _ = self.opt.target.parameterization(
                u_dense, v_dense, self.k)
            ax_ray.plot(Xc_th * 1e3,
                        Yc_th * 1e3,
                        z_t,
                        color='#d62728',
                        linewidth=1.5,
                        zorder=10)

            X_t = X_s + z_t * slope_x
            Y_t = Y_s + z_t * slope_y
            ax_ray.scatter(X_t * 1e3,
                           Y_t * 1e3,
                           z_t,
                           color='#0033aa',
                           s=6,
                           alpha=0.8,
                           edgecolor='none',
                           zorder=20)

        ax_ray.set_xlim(-self.L * 1e3, self.L * 1e3)
        ax_ray.set_ylim(-self.L * 1e3, self.L * 1e3)
        ax_ray.set_zlim(0, Z_ext)
        ax_ray.set_axis_off()
        save_fig_tight(fig_ray, "3_RayTracing")

        # ==================================================
        # 4, 5, 6. Propagation Planes (Non-diffracting observation)
        # ==================================================
        for z in z_targets:
            fig_z = plt.figure(figsize=(5.2, 4))
            ax_z = fig_z.add_axes([0.15, 0.15, 0.65, 0.8])

            Ez = self.propagate(z)
            I = np.abs(Ez)**2
            vmax = np.percentile(I, 99.8)

            I_norm = I / vmax

            # Matched coordinate extent for the 4mm observation window (-2mm to 2mm)
            window_mm = 2.0
            im_z = ax_z.imshow(
                I_norm,
                extent=[-window_mm, window_mm, -window_mm, window_mm],
                cmap='inferno',
                vmax=1.0,
                origin='lower')

            v_dense = np.linspace(self.opt.target.v_range[0],
                                  self.opt.target.v_range[1], 500)
            u_dense = np.full_like(v_dense, z)
            Xc_th, Yc_th, _, _, _, _ = self.opt.target.parameterization(
                u_dense, v_dense, self.k)
            ax_z.plot(Xc_th * 1e3, Yc_th * 1e3, 'w--', lw=1.2, alpha=0.7)

            # Crop theoretical overlay if it falls outside the zoomed 2mm observation window
            ax_z.set_xlim([-window_mm, window_mm])
            ax_z.set_ylim([-window_mm, window_mm])

            ax_z.set_xticks([])
            ax_z.set_yticks([])
            ax_z.tick_params(length=0)

            if abs(z - z_targets[-1]) < 1e-3:
                cax_z = fig_z.add_axes([0.83, 0.15, 0.04, 0.8])
                cbar_z = fig_z.colorbar(im_z, cax=cax_z)
                cbar_z.set_ticks([0.0, 0.5, 1.0])
                cbar_z.ax.tick_params(labelsize=14)

            filepath = os.path.join(script_dir,
                                    f"{prefix}_4_Plane_{int(z*100)}cm.pdf")
            fig_z.savefig(filepath, transparent=True)
            plt.close(fig_z)
            print(f"      -> Exported: {filepath}")


# ==========================================
# 4. Main Execution
# ==========================================
def run_validation(target_class, prefix, bounds_mm):
    target = target_class()
    mapper = AnalyticalMapper(target,
                              wl=532e-9,
                              N_grid=1024,
                              bounds_mm=bounds_mm)
    prop = WavePropagator(mapper, wl=532e-9, N_sim=1024)
    prop.interpolate_and_mask()
    prop.export_publication_figures(prefix=prefix)


if __name__ == "__main__":
    print("====== Validation 1/2: Cylindrical Circular (Bessel-like) ======")
    run_validation(CircularCylindrical, "Validation_Cyl_Circle", bounds_mm=4.0)

    print("\n====== Validation 2/2: Cylindrical Parabolic ======")
    run_validation(ParabolicCylindrical,
                   "Validation_Cyl_Parabolic",
                   bounds_mm=4.0)

    print(
        "\n[*] Cylindrical caustics rendering completed. Inspecting visual output windows..."
    )
    print("    (Script will exit automatically once windows are closed.)")
    plt.show()
