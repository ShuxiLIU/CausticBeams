"""
Numerical Validation of Transverse Caustic Lines

This script corresponds to Section 3.A of the Supplemental Document for the paper:
"Engineering geometrical optical singularities: inverse design of 3D asymmetric caustic surfaces"

It validates the newly derived analytical mapping solutions for transverse caustic lines,
focusing specifically on a Circular caustic (Eq. S27) and an Astroid caustic (Eq. S28).
The script performs analytical parameterization, rigorous topological masking, 
wave-optic propagation, and exports publication-quality figures (Fig. S1).
"""

import os
import numpy as np
import matplotlib.pyplot as plt
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
    'axes.linewidth': 0.8  # Uniform border thickness
})


# ==========================================
# 1. Target Definition Base Class (Analytical Mapping)
# ==========================================
class AnalyticalCaustic(ABC):
    """
    Abstract base class for analytical caustic mapping.
    """

    def __init__(self,
                 name="Generic",
                 C_range=(0.0, 3.0),
                 v_range=(0, 2 * np.pi)):
        self.name = name
        self.C_range = C_range
        self.v_range = v_range
        # has_hole indicates if the inner boundary should be masked out as a topological hole
        self.has_hole = False

    @abstractmethod
    def parameterization(self, C, v, k):
        """
        Returns: Xc, Yc, Zc (Target), X, Y (Source plane), Phi (Analytical Phase)
        """
        pass


class CircleCaustic(AnalyticalCaustic):
    """
    Circular transverse caustic line.
    (Refer to Eq. S27 in the Supplemental Document)
    """

    def __init__(self):
        super().__init__(name="Circle",
                         C_range=(-1.0, 0.95),
                         v_range=(0, 2 * np.pi))
        self.has_hole = False

    def parameterization(self, C, v, k):
        R = 2e-3  # Base radius = 2 mm
        u0 = 0.2  # Focal plane distance

        Xc = R * np.cos(v)
        Yc = R * np.sin(v)
        Zc = np.full_like(v, u0)

        X = (1 + C) * R * np.cos(v)
        Y = (1 + C) * R * np.sin(v)

        # Analytical quadratic-like phase for circle mapping
        Phi = -(k / (2 * u0)) * (C**2) * (R**2)
        return Xc, Yc, Zc, X, Y, Phi


class AstroidCaustic(AnalyticalCaustic):
    """
    Astroid (star-shaped) transverse caustic line.
    (Refer to Eq. S28 in the Supplemental Document)
    """

    def __init__(self):
        super().__init__(name="Astroid",
                         C_range=(0.0, 1.0),
                         v_range=(0, 2 * np.pi))
        self.has_hole = True

    def parameterization(self, C, v, k):
        R = 2e-3  # Base radius = 2 mm
        u0 = 0.2  # Focal plane distance

        Xc = R * np.cos(v)**3
        Yc = R * np.sin(v)**3
        Zc = np.full_like(v, u0)

        X = R * np.cos(v)**3 + 3 * C * R * np.sin(v)**2 * np.cos(v)
        Y = R * np.sin(v)**3 + 3 * C * R * np.cos(v)**2 * np.sin(v)

        # Analytical phase: f'^2 + g'^2 = 9R^2 cos^2(v) sin^2(v)
        Phi = -(k / (2 * u0)) * (C**2) * 9 * (R**2) * (np.cos(v)**
                                                       2) * (np.sin(v)**2)
        return Xc, Yc, Zc, X, Y, Phi


# ==========================================
# 2. Analytical Mapper Core
# ==========================================
class AnalyticalMapper:
    """
    Generates the discrete source plane mapping grid and phase profile 
    from the target parameterization.
    """

    def __init__(self,
                 target: AnalyticalCaustic,
                 wl,
                 N_grid=512,
                 bounds_mm=4.0):
        self.target = target
        self.N = N_grid
        self.bounds = bounds_mm * 1e-3
        self.k = 2 * np.pi / wl

        C_min, C_max = target.C_range
        v_min, v_max = target.v_range
        self.C_vec = np.linspace(C_min, C_max, self.N)
        self.v_vec = np.linspace(v_min, v_max, self.N)
        self.C, self.V = np.meshgrid(self.C_vec, self.v_vec, indexing='ij')

        print(
            f"[*] Generating analytical caustic mapping for [{target.name}]..."
        )
        self.Xc, self.Yc, self.Zc, self.X_sol, self.Y_sol, self.Phi_map = self.target.parameterization(
            self.C, self.V, self.k)
        self.Phi_map -= np.min(self.Phi_map)


# ==========================================
# 3. Topological Masking & Wave Propagation
# ==========================================
class WavePropagator:
    """
    Handles Cartesian interpolation, topological masking, and angular spectrum propagation.
    """

    def __init__(self, mapper: AnalyticalMapper, wl=532e-9, N_sim=1024):
        self.opt = mapper
        self.L = mapper.bounds
        self.N_sim = N_sim
        self.dx = 2 * self.L / N_sim
        self.wl = wl
        self.k = 2 * np.pi / wl

    def interpolate_and_mask(self):
        """
        Applies strict Cartesian interpolation and topology-based masking.
        Holes (e.g., inside the astroid) are rigorously excluded using ray-casting algorithms (Path).
        """
        print(
            "    -> Generating adaptive topological mask and interpolating...")
        x = np.linspace(-self.L, self.L, self.N_sim)
        y = np.linspace(-self.L, self.L, self.N_sim)
        self.xx, self.yy = np.meshgrid(x, y)
        pts_grid = np.column_stack((self.xx.flatten(), self.yy.flatten()))

        X_t, Y_t = self.opt.X_sol, self.opt.Y_sol

        # Define outer boundary path
        path_out = Path(np.column_stack((X_t[-1, :], Y_t[-1, :])))
        in_out = path_out.contains_points(pts_grid)

        if self.opt.target.has_hole:
            # Define inner boundary path (exclude the topological hole)
            path_in = Path(np.column_stack((X_t[0, :], Y_t[0, :])))
            in_in = path_in.contains_points(pts_grid)
            self.mask = (in_out & ~in_in).reshape(self.N_sim,
                                                  self.N_sim).astype(float)
        else:
            self.mask = in_out.reshape(self.N_sim, self.N_sim).astype(float)

        points = np.column_stack((X_t.flatten(), Y_t.flatten()))
        values = self.opt.Phi_map.flatten()

        # Interpolate scattered mapping data onto a uniform grid
        phi_grid = griddata(points,
                            values, (self.xx, self.yy),
                            method='linear',
                            fill_value=0.0)
        self.phi_grid = np.nan_to_num(phi_grid, 0)
        self.E0 = self.mask * np.exp(1j * self.phi_grid * self.mask)

    def propagate(self, z):
        """Free-space propagation using the Angular Spectrum Method (ASM)."""
        fx = fftfreq(self.N_sim, self.dx)
        FX, FY = np.meshgrid(fx, fx)
        H = np.exp(
            1j * self.k * z *
            np.sqrt(np.maximum(0, 1 - (self.wl * FX)**2 - (self.wl * FY)**2)))
        return ifft2(fft2(self.E0) * H)

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
            # plt.close(fig) is deliberately omitted to allow final visual inspection via plt.show()
            print(f"      -> Exported: {filepath}")

        # ==================================================
        # 1. Source Plane Mapping Grid
        # ==================================================
        fig_map = plt.figure(figsize=(4, 4))
        ax_map = fig_map.add_axes([0.15, 0.15, 0.75, 0.75])
        skip_m = 4
        sc = ax_map.scatter(self.opt.X_sol[::skip_m, ::skip_m].flatten() * 1e3,
                            self.opt.Y_sol[::skip_m, ::skip_m].flatten() * 1e3,
                            c=self.opt.C[::skip_m, ::skip_m].flatten(),
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
        ax_map.set_xlim([-4.0, 4.0])
        ax_map.set_ylim([-4.0, 4.0])

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
        # Elevate plot area to make room for the colorbar
        ax_ph = fig_ph.add_axes([0.15, 0.20, 0.75, 0.75 * (4 / 4.5)])

        phase_display = np.angle(self.E0)
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

        # Horizontal Colorbar
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
        safe_mask = binary_erosion(self.mask > 0.5, iterations=2)

        skip_s = self.N_sim // 80
        sample_mask = np.zeros_like(safe_mask)
        sample_mask[::skip_s, ::skip_s] = True
        final_mask = safe_mask & sample_mask

        X_s = self.xx[final_mask]
        Y_s = self.yy[final_mask]
        gx = grad_x[final_mask]
        gy = grad_y[final_mask]

        slope_x = gx / self.k
        slope_y = gy / self.k

        Z_target = self.opt.Zc[0, 0]  # Focal plane location (0.2m)
        Z_ext = 0.28  # Upward extension

        X_ext = X_s + Z_ext * slope_x
        Y_ext = Y_s + Z_ext * slope_y

        # Ray envelope: rendered as an ultra-transparent blue mist
        segments = [((X_s[i] * 1e3, Y_s[i] * 1e3, 0), (X_ext[i] * 1e3,
                                                       Y_ext[i] * 1e3, Z_ext))
                    for i in range(len(X_s))]
        lc = Line3DCollection(segments,
                              colors="#72bff7",
                              linewidths=0.6,
                              alpha=0.08)
        ax_ray.add_collection3d(lc)

        # Theoretical curve at the focal plane
        v_dense = np.linspace(0, 2 * np.pi, 500)
        Xc_th, Yc_th, _, _, _, _ = self.opt.target.parameterization(
            np.zeros_like(v_dense), v_dense, self.k)
        ax_ray.plot(Xc_th * 1e3,
                    Yc_th * 1e3,
                    Z_target,
                    color='#d62728',
                    linewidth=1.5,
                    zorder=10)

        # Scatter points at specific observation planes (15cm, 20cm, 25cm)
        z_scatters = [0.15, 0.20, 0.25]
        for z_t in z_scatters:
            X_t = X_s + z_t * slope_x
            Y_t = Y_s + z_t * slope_y
            ax_ray.scatter(X_t * 1e3,
                           Y_t * 1e3,
                           z_t,
                           color='#0033aa',
                           s=15,
                           alpha=0.95,
                           edgecolor='none',
                           zorder=20)

        ax_ray.set_xlim(-self.L * 1e3, self.L * 1e3)
        ax_ray.set_ylim(-self.L * 1e3, self.L * 1e3)
        ax_ray.set_zlim(0, Z_ext)
        ax_ray.set_axis_off()
        save_fig_tight(fig_ray, "3_RayTracing")

        # ==================================================
        # 4, 5, 6. Transverse Intensity Distributions
        # ==================================================
        z_planes = [0.15, 0.20, 0.25]
        for z in z_planes:
            fig_z = plt.figure(figsize=(5.2, 4))
            ax_z = fig_z.add_axes([0.15, 0.15, 0.65, 0.8])

            Ez = self.propagate(z)
            I = np.abs(Ez)**2
            vmax = np.percentile(I, 99.8)

            I_norm = I / vmax

            im_z = ax_z.imshow(I_norm,
                               extent=[
                                   -self.L * 1e3, self.L * 1e3, -self.L * 1e3,
                                   self.L * 1e3
                               ],
                               cmap='inferno',
                               vmax=1.0,
                               origin='lower')
            ax_z.plot(Xc_th * 1e3, Yc_th * 1e3, 'w--', lw=1.2, alpha=0.7)

            # Remove all axes labels and ticks for a clean look
            ax_z.set_xticks([])
            ax_z.set_yticks([])
            ax_z.tick_params(length=0)

            # Append precise colorbar only for the furthest plane (Plane_25)
            if abs(z - 0.25) < 1e-3:
                cax_z = fig_z.add_axes([0.83, 0.15, 0.04, 0.8])
                cbar_z = fig_z.colorbar(im_z, cax=cax_z)
                cbar_z.set_ticks([0.0, 0.5, 1.0])
                cbar_z.ax.tick_params(labelsize=14)

            filepath = os.path.join(script_dir,
                                    f"{prefix}_4_Plane_{int(z*100)}cm.pdf")

            fig_z.savefig(filepath, transparent=True)
            print(f"      -> Exported: {filepath}")


# ==========================================
# 4. Main Execution
# ==========================================
def run_validation(target_class, prefix, bounds_mm):
    target = target_class()
    mapper = AnalyticalMapper(target,
                              wl=532e-9,
                              N_grid=512,
                              bounds_mm=bounds_mm)
    prop = WavePropagator(mapper, wl=532e-9, N_sim=1024)
    prop.interpolate_and_mask()
    prop.export_publication_figures(prefix=prefix)


if __name__ == "__main__":
    print("====== Validation 1/2: Analytical Circle Mapping ======")
    run_validation(CircleCaustic, "Validation_Circle", bounds_mm=4.0)

    print("\n====== Validation 2/2: Analytical Astroid Mapping ======")
    run_validation(AstroidCaustic, "Validation_Astroid", bounds_mm=4.0)

    print(
        "\n[*] All renderings completed. Inspecting visual output windows...")
    print("    (Script will exit automatically once windows are closed.)")
    plt.show()
