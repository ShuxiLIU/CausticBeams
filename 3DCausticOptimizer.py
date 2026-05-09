import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy.sparse as sp
import os
from abc import ABC, abstractmethod
from matplotlib.path import Path
from scipy.optimize import least_squares
from scipy.sparse.linalg import lsqr
from scipy.interpolate import griddata
from scipy.fft import fft2, ifft2, fftfreq
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter, distance_transform_edt
from scipy.io import savemat

# ==========================================
# 1. Base Class and Target Definitions
# ==========================================


class CausticTarget(ABC):
    """
    Base class for 3D caustic targets.
    Inherit this class to define parametric generating equations.
    """

    def __init__(self,
                 name="Generic",
                 u_range=(0.05, 0.8),
                 v_range=(0, 2 * np.pi),
                 is_cyclic_v=False):
        """
        is_cyclic_v: Determines if periodic conditions are applied along the v direction.
        """
        self.name = name
        self.u_range = u_range
        self.v_range = v_range
        self.is_cyclic_v = is_cyclic_v

    @abstractmethod
    def parameterization(self, u, v):
        """
        Define the parametric equations C(u, v).
        u: Longitudinal parameter (propagation axis, z)
        v: Transverse/Angular parameter
        return: xc, yc, zc
        """
        pass

    @abstractmethod
    def homotopy_initial(self, u, v):
        """
        Dynamically allocates the structurally matching canonical initial surface
        based on the topological cyclic nature of the target.
        return: xc_init, yc_init, zc_init, x_init, y_init
        """
        pass


class LinearExpandingEllipse(CausticTarget):
    """ Target from Fig. 1: a demo of linearly expanding ellipse """

    def __init__(self, name="Expanding Ellipse"):
        super().__init__(name,
                         u_range=(0.05, 0.8),
                         v_range=(0, 2 * np.pi),
                         is_cyclic_v=True)

    def parameterization(self, u, v):
        scale = (0.3 + 1.5 * u) * 1e-3
        a = 1.2 * scale
        b = 0.8 * scale
        xc = a * np.cos(v)
        yc = b * np.sin(v)
        zc = u
        return xc, yc, zc

    def homotopy_initial(self, u, v):
        R = 1e-3
        theta = v
        xc_init = R * np.cos(theta)
        yc_init = R * np.sin(theta)
        zc_init = u
        sign = 1
        long = 4e-3
        x_init = xc_init - sign * long * u * np.sin(theta)
        y_init = yc_init + sign * long * u * np.cos(theta)
        return xc_init, yc_init, zc_init, x_init, y_init


class CubicSurface(CausticTarget):
    """ Target from Fig. 3 & Fig. 4: cubic surface """

    def __init__(self, name="Cubic Surface"):
        super().__init__(name,
                         u_range=(0.05, 0.8),
                         v_range=(-1, 1),
                         is_cyclic_v=False)

    def parameterization(self, u, v):
        scale = 2e-3
        xc = scale * v
        yc = 2 * scale * (v**3 / 3 - u * v)
        zc = u
        return xc, yc, zc

    def homotopy_initial(self, u, v):
        R = 0.5e-3
        theta = v * 2 + np.pi * 0.5
        xc_init = R * np.cos(theta) - 1e-3
        # rotated by 180 degrees
        # theta = v * 2 - np.pi * 0.5
        # xc_init = R * np.cos(theta) - 2e-3
        yc_init = R * np.sin(theta)
        zc_init = u
        sign = 1
        long = 3.5e-3
        x_init = xc_init - sign * long * u * np.sin(theta)
        y_init = yc_init + sign * long * u * np.cos(theta)
        return xc_init, yc_init, zc_init, x_init, y_init


class ConvergingCardioid(CausticTarget):
    """ Target from Fig. 5a: Converging Cardioid """

    def __init__(self, name="Converging Cardioid"):
        super().__init__(name,
                         u_range=(0.05, 0.8),
                         v_range=(0, 2 * np.pi),
                         is_cyclic_v=True)

    def parameterization(self, u, v):
        R_base = 1e-3 * (1 - 0.6 * u)  # Shrinking radius along propagation
        xc = R_base * (1 - np.cos(v)) * np.cos(v) + R_base
        yc = R_base * (1 - np.cos(v)) * np.sin(v)
        zc = u
        return xc, yc, zc

    def homotopy_initial(self, u, v):
        R = 1e-3
        theta = v
        xc_init = R * np.cos(theta)
        yc_init = R * np.sin(theta)
        zc_init = u
        sign = 1
        long = 4e-3
        x_init = xc_init - sign * long * u * np.sin(theta)
        y_init = yc_init + sign * long * u * np.cos(theta)
        return xc_init, yc_init, zc_init, x_init, y_init


class AstigmaticEnvelope(CausticTarget):
    """ Target from Fig. 5b: Astigmatic Envelope """

    def __init__(self, name="Astigmatic Envelope"):
        super().__init__(name,
                         u_range=(0.05, 0.8),
                         v_range=(0, 2 * np.pi),
                         is_cyclic_v=True)

    def parameterization(self, u, v):
        a0, b0 = 1.2e-3, 0.6e-3
        k_a, k_b = -0.8e-3, 0.8e-3  # Cross astigmatism effect
        a_u = np.maximum(a0 + k_a * (u - self.u_range[0]), 1e-4)
        b_u = np.maximum(b0 + k_b * (u - self.u_range[0]), 1e-4)
        xc = a_u * np.cos(v)
        yc = b_u * np.sin(v)
        zc = u
        return xc, yc, zc

    def homotopy_initial(self, u, v):
        R = 1e-3
        theta = v
        xc_init = R * np.cos(theta)
        yc_init = R * np.sin(theta)
        zc_init = u
        sign = 1
        long = 4e-3
        x_init = xc_init - sign * long * u * np.sin(theta)
        y_init = yc_init + sign * long * u * np.cos(theta)
        return xc_init, yc_init, zc_init, x_init, y_init


class SkewedParabolicTrough(CausticTarget):
    """ Target from Fig. 5c: Skewed Parabolic Trough """

    def __init__(self, name="Skewed Parabolic Trough"):
        super().__init__(name,
                         u_range=(0.05, 0.8),
                         v_range=(-1.0, 1.0),
                         is_cyclic_v=False)

    def parameterization(self, u, v):
        L = 1e-3
        xc = L * v
        A = (1.5 - 1.2 * u) * 1e-3  # Curvature
        B = 1e-3 * u**2  # Asymmetric skewness
        C = 0.5e-3 * np.sqrt(u)  # Baseline shift
        yc = A * v**2 + B * v + C
        zc = u
        return xc, yc, zc

    def homotopy_initial(self, u, v):
        R = 0.5e-3
        theta = v - np.pi * 0.5
        xc_init = R * np.cos(theta)
        yc_init = R * np.sin(theta) - 1e-3
        zc_init = u
        sign = 1
        long = 4e-3
        x_init = xc_init - sign * long * u * np.sin(theta)
        y_init = yc_init + sign * long * u * np.cos(theta)
        return xc_init, yc_init, zc_init, x_init, y_init


class DriftingOvoidalCone(CausticTarget):
    """ Target from Fig. 5d: Drifting Ovoidal Cone """

    def __init__(self, name="Drifting Ovoidal Cone"):
        super().__init__(name,
                         u_range=(0.05, 0.8),
                         v_range=(0, 2 * np.pi),
                         is_cyclic_v=True)

    def parameterization(self, u, v):
        scale = (0.5 + 0.6 * u) * 1e-3
        a, b = 1.2 * scale, 0.8 * scale
        k = 0.15  # Egg-shape modulation factor
        drift_x = 0.2e-3 * np.sin(4 * u)
        drift_y = 0.2e-3 * u**2
        xc = a * np.cos(v) + drift_x
        yc = b * np.sin(v) * (1 + k * np.cos(v)) + drift_y
        zc = u
        return xc, yc, zc

    def homotopy_initial(self, u, v):
        R = 0.5e-3
        theta = v
        xc_init = R * np.cos(theta)
        yc_init = R * np.sin(theta)
        zc_init = u
        sign = 1
        long = 4e-3
        x_init = xc_init - sign * long * u * np.sin(theta)
        y_init = yc_init + sign * long * u * np.cos(theta)
        return xc_init, yc_init, zc_init, x_init, y_init


class PearceyLikeCaustic(CausticTarget):
    """ Target from Fig. 5e: Pearcey-like Caustic """

    def __init__(self, name="Pearcey-like Caustic"):
        super().__init__(name,
                         u_range=(0.05, 0.8),
                         v_range=(-1.5, 1.5),
                         is_cyclic_v=False)

    def parameterization(self, u, v):
        k = 2 * np.pi / 532e-9
        u_max_val = self.u_range[1]
        l0 = np.sqrt(u_max_val / (4 * k))
        xc = 4 * (2 - 4 * u / u_max_val) * v**3 * l0
        yc = -3 * (2 - 4 * u / u_max_val) * v**2 * l0
        zc = u
        return xc, yc, zc

    def homotopy_initial(self, u, v):
        R = 1e-3
        theta = v
        xc_init = R * np.cos(theta)
        yc_init = R * np.sin(theta)
        zc_init = u
        sign = 1
        long = 4e-3
        x_init = xc_init - sign * long * u * np.sin(theta)
        y_init = yc_init + sign * long * u * np.cos(theta)
        return xc_init, yc_init, zc_init, x_init, y_init


class HelicoidalTrefoil(CausticTarget):
    """ Target from Fig. 5f: Helicoidal Trefoil """

    def __init__(self, name="Helicoidal Trefoil"):
        super().__init__(name,
                         u_range=(0.05, 0.8),
                         v_range=(0, 2 * np.pi),
                         is_cyclic_v=True)

    def parameterization(self, u, v):
        R_base = 1e-3
        angle = v + 1.5 * u
        distortion = 0.25 * np.cos(3 * angle)
        r = R_base * (1 + distortion)
        xc = r * np.cos(v)
        yc = r * np.sin(v)
        zc = u
        return xc, yc, zc

    def homotopy_initial(self, u, v):
        R = 1e-3
        theta = v
        xc_init = R * np.cos(theta)
        yc_init = R * np.sin(theta)
        zc_init = u
        sign = 1
        long = 4e-3
        x_init = xc_init - sign * long * u * np.sin(theta)
        y_init = yc_init + sign * long * u * np.cos(theta)
        return xc_init, yc_init, zc_init, x_init, y_init


# ==========================================
# 2. Optimization Solver (Homotopy Continuation & Newton Method)
# ==========================================


class CausticOptimizer:

    def __init__(self,
                 target: CausticTarget,
                 init_mode='homotopy',
                 N_opt=64,
                 bounds_mm=4.0,
                 loss_weights=None):
        self.target = target
        self.initial = init_mode
        self.N = N_opt
        self.bounds = bounds_mm * 1e-3

        default_weights = {
            'eq1': 1e0,
            'eq2': 1e0,
            'jac': 1e0,
            'bnd': 1e3,
            'lap': 5e-3,
            'cyclic': 1e3
        }
        self.weights = loss_weights if loss_weights is not None else default_weights

        self.u_min, self.u_max = self.target.u_range
        self.v_min, self.v_max = self.target.v_range

        self.u_vec = np.linspace(self.u_min, self.u_max, self.N)
        self.v_vec = np.linspace(self.v_min, self.v_max, self.N)
        self.U, self.V = np.meshgrid(self.u_vec, self.v_vec, indexing='ij')

        self.du = self.u_vec[1] - self.u_vec[0]
        self.dv = self.v_vec[1] - self.v_vec[0]

        self.Xc, self.Yc, self.Zc = self.target.parameterization(
            self.U, self.V)

        self.X_sol = None
        self.Y_sol = None

        print(f"[*] Target caustic [{target.name}] initialized...")
        print(f"    u range: [{self.u_min:.2f}, {self.u_max:.2f}]")
        print(f"    v range: [{self.v_min:.2f}, {self.v_max:.2f}]")
        print(f"    v periodicity: {target.is_cyclic_v}")
        print(f"    Grid resolution: {N_opt}x{N_opt}")

    def save_results(self, filename="solver_cache.npz"):
        """ Serialize computing results and historical data """
        if self.X_sol is None or self.Y_sol is None:
            print("[!] No data available to save.")
            return

        save_dict = {'X_sol': self.X_sol, 'Y_sol': self.Y_sol}
        if hasattr(self, 'history'):
            for k, v in self.history.items():
                save_dict[f'hist_{k}'] = np.array(v)

        np.savez(filename, **save_dict)
        print(f"[*] Optimization results cached in: {filename}")

    def load_results(self, filename="solver_cache.npz"):
        try:
            data = np.load(filename, allow_pickle=True)
            self.X_sol = data['X_sol']
            self.Y_sol = data['Y_sol']

            self.history = {}
            for k in data.files:
                if k.startswith('hist_'):
                    self.history[k.replace('hist_', '')] = list(data[k])

            print(f"[*] Successfully loaded cached results from: {filename}")
            return True
        except Exception as e:
            print(f"[!] Failed to load cache: {e}")
            return False

    def get_gradients(self, F, is_cyclic=False, axis=0):
        """ Compute unified 2nd-order gradients with correct cyclic boundary padding """
        if axis == 0:
            return np.gradient(F, self.du, axis=0, edge_order=2)
        if axis == 1:
            if is_cyclic:
                pad_width = [(0, 0), (1, 1)]
                F_padded = np.pad(F, pad_width, mode='wrap')
                grad_padded = np.gradient(F_padded,
                                          self.dv,
                                          axis=1,
                                          edge_order=2)
                return grad_padded[:, 1:-1]
            else:
                return np.gradient(F, self.dv, axis=1, edge_order=2)

    def make_sparsity_matrix(self):
        """ Construct the sparse pattern of the Jacobian matrix (Vectorized) """
        N = self.N
        N2 = N**2
        radius = 2
        idx_grid = np.arange(N2).reshape(N, N)

        local_rows, local_cols = [], []
        i_idx, j_idx = np.indices((N, N))

        for di in range(-radius, radius + 1):
            for dj in range(-radius, radius + 1):
                ni = i_idx + di
                nj = j_idx + dj

                if self.target.is_cyclic_v:
                    nj = nj % N
                    valid_mask = (ni >= 0) & (ni < N)
                else:
                    valid_mask = (ni >= 0) & (ni < N) & (nj >= 0) & (nj < N)

                current_indices = idx_grid[valid_mask]
                neighbor_indices = ni[valid_mask] * N + nj[valid_mask]
                local_rows.append(current_indices)
                local_cols.append(neighbor_indices)

        all_rows = np.concatenate(local_rows)
        all_cols = np.concatenate(local_cols)
        structure_block = sp.coo_matrix(
            (np.ones(len(all_rows)), (all_rows, all_cols)), shape=(N2, N2))

        blocks = []
        # Number: (Eq1, Eq2, Jac, BndX, BndY, LapX, LapY = 7)
        num_standard_residual_blocks = 7
        for _ in range(num_standard_residual_blocks):
            blocks.append([structure_block, structure_block])

        if self.target.is_cyclic_v:
            rows_cyc = np.repeat(np.arange(N), 2)
            cols_cyc = []
            for i in range(N):
                cols_cyc.append(i * N)
                cols_cyc.append(i * N + N - 1)
            cols_cyc = np.array(cols_cyc)
            data_cyc = np.ones(len(cols_cyc))

            cyc_block = sp.coo_matrix((data_cyc, (rows_cyc, cols_cyc)),
                                      shape=(N, N2))
            empty_block = sp.coo_matrix(([], ([], [])), shape=(N, N2))

            blocks.append([cyc_block, empty_block])
            blocks.append([empty_block, cyc_block])

        return sp.bmat(blocks)

    def loss_function(self, params, dXmid_du, dXmid_dv, dYmid_du, dYmid_dv,
                      dZmid_du, dZmid_dv, J_yc_zc, J_zc_xc, J_xc_yc, Xmid,
                      Ymid, Zmid):
        """ Optimization loss function integrating governing PDEs and constraints. """
        is_cyclic_v = self.target.is_cyclic_v

        X = params[:self.N**2].reshape(self.N, self.N)
        Y = params[self.N**2:].reshape(self.N, self.N)
        dX_du = self.get_gradients(X, is_cyclic=False, axis=0)
        dX_dv = self.get_gradients(X, is_cyclic=is_cyclic_v, axis=1)
        dY_du = self.get_gradients(Y, is_cyclic=False, axis=0)
        dY_dv = self.get_gradients(Y, is_cyclic=is_cyclic_v, axis=1)

        # Inline function for fast determinant calculation
        def fast_det(df_du, df_dv, dg_du, dg_dv):
            return df_du * dg_dv - df_dv * dg_du

        # Eq 1: Tangency condition (Refer to Eq. 13 in the paper)
        eq1 = (Xmid - X) * J_yc_zc + (Ymid - Y) * J_zc_xc + Zmid * J_xc_yc
        eq1_weight = (1e6 / self.N) * self.weights['eq1']

        # Eq 2: Paraxial irrotational condition (Refer to Eq. 14 in the paper)
        J_yc_y = fast_det(dYmid_du, dYmid_dv, dY_du, dY_dv)
        J_zc_y = fast_det(dZmid_du, dZmid_dv, dY_du, dY_dv)
        J_x_xc = fast_det(dX_du, dX_dv, dXmid_du, dXmid_dv)
        J_x_zc = fast_det(dX_du, dX_dv, dZmid_du, dZmid_dv)
        eq2 = (-Zmid * J_yc_y + (Ymid - Y) * J_zc_y) - \
              (-Zmid * J_x_xc + (Xmid - X) * J_x_zc)
        eq2_weight = (1e6 / self.N) * self.weights['eq2']

        # Constraint 1: Diffeomorphic constraint (Enforce positive Jacobian to prevent topological folding)
        J_map = fast_det(dX_du, dX_dv, dY_du, dY_dv)
        min_abs_jac = 1e-7
        penalty_jac = np.maximum(0, min_abs_jac - J_map)
        jac_weight = (1e6 / self.N) * self.weights['jac']

        # Constraint 2: Boundary constraint
        penalty_boundary_x = np.maximum(0, np.abs(X) - self.bounds)
        penalty_boundary_y = np.maximum(0, np.abs(Y) - self.bounds)
        boundary_weight = (1e3 / self.N) * self.weights['bnd']

        # Constraint 3: Smoothness Regularization (Laplacian)
        d2X_du2 = self.get_gradients(dX_du, is_cyclic=False, axis=0)
        d2X_dv2 = self.get_gradients(dX_dv, is_cyclic=is_cyclic_v, axis=1)
        laplacian_x = np.abs(d2X_du2 + d2X_dv2)

        d2Y_du2 = self.get_gradients(dY_du, is_cyclic=False, axis=0)
        d2Y_dv2 = self.get_gradients(dY_dv, is_cyclic=is_cyclic_v, axis=1)
        laplacian_y = np.abs(d2Y_du2 + d2Y_dv2)

        laplacian_weight = (1e3 / self.N) * self.weights['lap']

        residuals = [
            eq1_weight * eq1.flatten(),
            eq2_weight * eq2.flatten(),
            jac_weight * penalty_jac.flatten(),
            boundary_weight * penalty_boundary_x.flatten(),
            boundary_weight * penalty_boundary_y.flatten(),
            laplacian_weight * laplacian_x.flatten(),
            laplacian_weight * laplacian_y.flatten(),
        ]

        if is_cyclic_v:
            cyclic_weight = (1e3 / np.sqrt(self.N)) * self.weights['cyclic']
            residuals.append(cyclic_weight * (X[:, 0] - X[:, -1]).flatten())
            residuals.append(cyclic_weight * (Y[:, 0] - Y[:, -1]).flatten())

        return np.concatenate(residuals)

    def initial_guess(self, mode='homotopy'):
        """ Generate initial mapping functions for solvers """
        if mode == 'homotopy':
            return self.target.homotopy_initial(self.U, self.V)

        elif mode == 'linear':
            scaling_factor = 0.7 * self.bounds / max(np.max(np.abs(self.Xc)),
                                                     np.max(np.abs(self.Yc)))
            X_init = scaling_factor * self.Xc
            Y_init = scaling_factor * self.Yc
            return None, None, None, X_init, Y_init

        elif mode == 'projection':
            safe_Zc = np.maximum(np.abs(self.Zc), 1e-6)
            ratio_x = self.Xc / safe_Zc
            ratio_y = self.Yc / safe_Zc
            max_ratio = max(np.max(np.abs(ratio_x)), np.max(np.abs(ratio_y)))
            virtual_focal_length = 0.95 * self.bounds / max_ratio
            X_init = ratio_x * virtual_focal_length
            Y_init = ratio_y * virtual_focal_length
            return None, None, None, X_init, Y_init

        elif mode == 'tangent':
            is_cyclic_v = self.target.is_cyclic_v
            # 1. Calculate tangent vectors of the caustic surface
            # dC/du
            dXc_du = self.get_gradients(self.Xc, is_cyclic=False, axis=0)
            dYc_du = self.get_gradients(self.Yc, is_cyclic=False, axis=0)
            dZc_du = self.get_gradients(self.Zc, is_cyclic=False, axis=0)
            # dC/dv
            dXc_dv = self.get_gradients(self.Xc, is_cyclic=is_cyclic_v, axis=1)
            dYc_dv = self.get_gradients(self.Yc, is_cyclic=is_cyclic_v, axis=1)
            dZc_dv = self.get_gradients(self.Zc, is_cyclic=is_cyclic_v, axis=1)

            # 2. Compute normal vector n = Tu x Tv
            nx = dYc_du * dZc_dv - dZc_du * dYc_dv
            ny = dZc_du * dXc_dv - dXc_du * dZc_dv
            nz = dXc_du * dYc_dv - dYc_du * dXc_dv

            # Normalize normal vector (avoid numerical instability)
            norm_n = np.sqrt(nx**2 + ny**2 + nz**2)
            norm_n[norm_n == 0] = 1.0
            nx, ny, nz = nx / norm_n, ny / norm_n, nz / norm_n

            # 3. Construct ray direction in the tangent plane
            # Find the direction in the tangent plane with the smallest angle to the Z-axis (0,0,1)
            # i.e., the projection of the Z-axis onto the tangent plane: v = k - (k·n)n
            vx = 0 - nz * nx
            vy = 0 - nz * ny
            vz = 1 - nz * nz
            norm_v = np.sqrt(vx**2 + vy**2 + vz**2)
            norm_v[norm_v == 0] = 1.0
            vx, vy, vz = vx / norm_v, vy / norm_v, vz / norm_v

            tx = -ny
            ty = nx
            tz = 0
            norm_t = np.sqrt(tx**2 + ty**2 + tz**2)
            norm_t[norm_t == 0] = 1.0
            tx, ty, tz = tx / norm_t, ty / norm_t, tz / norm_t

            A = 0.001
            combine_x = vx + A * tx
            combine_y = vy + A * ty
            combine_z = vz + A * tz
            combine_z[np.abs(combine_z) < 1e-6] = 1e-6
            slope_x = combine_x / combine_z
            slope_y = combine_y / combine_z

            # Back-calculate source coordinates
            X_init = self.Xc - self.Zc * slope_x
            Y_init = self.Yc - self.Zc * slope_y

            return None, None, None, X_init, Y_init

        else:
            raise ValueError(f"Unknown initial guess mode: {mode}")

    def solve(self, max_iter=100, tol=1e-4, steps=10, verbose=2):
        """ Solve mapping optimization """
        print("[*] Starting mapping optimization...")
        print(f"    Initial guess mode: {self.initial}")
        print(f"    tol = {tol:.1e}, max_iter = {max_iter}")

        jac_sparsity = self.make_sparsity_matrix()

        self.history = {
            'alpha': [],
            'Xmid': [],
            'Ymid': [],
            'Zmid': [],
            'X_init': [],
            'Y_init': [],
            'X_opt': [],
            'Y_opt': [],
            'res_init': [],
            'res_opt': []
        }

        Xc_init, Yc_init, Zc_init, X_init, Y_init = self.initial_guess(
            mode=self.initial)
        if Xc_init is None or Yc_init is None or Zc_init is None:
            Xc_init, Yc_init, Zc_init = self.Xc.copy(), self.Yc.copy(
            ), self.Zc.copy()
            steps = 2

        p0 = np.concatenate([X_init.flatten(), Y_init.flatten()])
        is_cyclic = self.target.is_cyclic_v
        lim = self.bounds * 1.5  # hard bound for optimization variables to prevent divergence

        def fast_det(df_du, df_dv, dg_du, dg_dv):
            return df_du * dg_dv - df_dv * dg_du

        for i, alpha in enumerate(np.linspace(0, 1, steps + 1)):
            print(f"    > Step {i}/{steps} (alpha={alpha:.2f})")

            # Intermediate homotopy target
            Xmid = (1 - alpha) * Xc_init + alpha * self.Xc
            Ymid = (1 - alpha) * Yc_init + alpha * self.Yc
            Zmid = (1 - alpha) * Zc_init + alpha * self.Zc

            dXmid_du = self.get_gradients(Xmid, is_cyclic=False, axis=0)
            dXmid_dv = self.get_gradients(Xmid, is_cyclic=is_cyclic, axis=1)
            dYmid_du = self.get_gradients(Ymid, is_cyclic=False, axis=0)
            dYmid_dv = self.get_gradients(Ymid, is_cyclic=is_cyclic, axis=1)
            dZmid_du = self.get_gradients(Zmid, is_cyclic=False, axis=0)
            dZmid_dv = self.get_gradients(Zmid, is_cyclic=is_cyclic, axis=1)

            J_yc_zc = fast_det(dYmid_du, dYmid_dv, dZmid_du, dZmid_dv)
            J_zc_xc = fast_det(dZmid_du, dZmid_dv, dXmid_du, dXmid_dv)
            J_xc_yc = fast_det(dXmid_du, dXmid_dv, dYmid_du, dYmid_dv)

            current_init_res = self.loss_function(p0, dXmid_du, dXmid_dv,
                                                  dYmid_du, dYmid_dv, dZmid_du,
                                                  dZmid_dv, J_yc_zc, J_zc_xc,
                                                  J_xc_yc, Xmid, Ymid, Zmid)

            self.history['alpha'].append(alpha)
            self.history['Xmid'].append(Xmid.copy())
            self.history['Ymid'].append(Ymid.copy())
            self.history['Zmid'].append(Zmid.copy())
            self.history['X_init'].append(p0[:self.N**2].reshape(
                self.N, self.N).copy())
            self.history['Y_init'].append(p0[self.N**2:].reshape(
                self.N, self.N).copy())
            self.history['res_init'].append(np.linalg.norm(current_init_res))

            if i == 0:
                print(
                    f"      Initial overall residual: {np.linalg.norm(current_init_res):.2e}"
                )

            res = least_squares(self.loss_function,
                                p0,
                                args=(dXmid_du, dXmid_dv, dYmid_du, dYmid_dv,
                                      dZmid_du, dZmid_dv, J_yc_zc, J_zc_xc,
                                      J_xc_yc, Xmid, Ymid, Zmid),
                                jac_sparsity=jac_sparsity,
                                bounds=(-lim, lim),
                                method='trf',
                                ftol=tol,
                                xtol=tol,
                                gtol=tol,
                                max_nfev=max_iter,
                                verbose=verbose)
            p0 = res.x

            self.history['X_opt'].append(res.x[:self.N**2].reshape(
                self.N, self.N).copy())
            self.history['Y_opt'].append(res.x[self.N**2:].reshape(
                self.N, self.N).copy())
            self.history['res_opt'].append(np.linalg.norm(res.fun))

        self.X_sol = res.x[:self.N**2].reshape(self.N, self.N)
        self.Y_sol = res.x[self.N**2:].reshape(self.N, self.N)

        res_norm = np.linalg.norm(res.fun)
        res_eq1_norm = np.linalg.norm(res.fun[0:self.N**2])
        res_eq2_norm = np.linalg.norm(res.fun[self.N**2:2 * self.N**2])
        res_jac_norm = np.linalg.norm(res.fun[2 * self.N**2:3 * self.N**2])
        res_boundx_norm = np.linalg.norm(res.fun[3 * self.N**2:4 * self.N**2])
        res_boundy_norm = np.linalg.norm(res.fun[4 * self.N**2:5 * self.N**2])
        res_lapx_norm = np.linalg.norm(res.fun[5 * self.N**2:6 * self.N**2])
        res_lapy_norm = np.linalg.norm(res.fun[6 * self.N**2:7 * self.N**2])

        print(f"   Optimization finished, final residual norm: {res_norm:.2e}")
        print(f"   Eq1 residual: {res_eq1_norm:.2e}")
        print(f"   Eq2 residual: {res_eq2_norm:.2e}")
        print(f"   Jac residual: {res_jac_norm:.2e}")
        print(f"   Boundary X residual: {res_boundx_norm:.2e}")
        print(f"   Boundary Y residual: {res_boundy_norm:.2e}")
        print(f"   Laplacian X residual: {res_lapx_norm:.2e}")
        print(f"   Laplacian Y residual: {res_lapy_norm:.2e}")
        print(
            f"   X range: [{self.X_sol.min():.2e}, {self.X_sol.max():.2e}] m")
        print(
            f"   Y range: [{self.Y_sol.min():.2e}, {self.Y_sol.max():.2e}] m")

        return self.X_sol, self.Y_sol

    def visualize_results(self):
        fig = plt.figure(figsize=(10, 5))
        ax1 = plt.subplot(121, projection='3d')
        skip = 1
        ax1.plot_surface(self.Xc[::skip, ::skip] * 1e3,
                         self.Yc[::skip, ::skip] * 1e3,
                         self.Zc[::skip, ::skip],
                         alpha=0.7,
                         cmap='plasma')
        ax1.set_xlabel("x (mm)")
        ax1.set_ylabel("y (mm)")
        ax1.set_zlabel("z (m)")
        ax1.set_title("Target Caustic Surface")

        ax2 = fig.add_subplot(122)
        dX_du, dX_dv = self.get_gradients(self.X_sol, is_cyclic=False, axis=0), \
            self.get_gradients(self.X_sol, self.target.is_cyclic_v, axis=1)
        dY_du, dY_dv = self.get_gradients(self.Y_sol, is_cyclic=False, axis=0), \
            self.get_gradients(self.Y_sol, self.target.is_cyclic_v, axis=1)
        J_map = dX_du * dY_dv - dX_dv * dY_du
        im = ax2.imshow(
            J_map,
            extent=[self.v_min, self.v_max, self.u_min, self.u_max],
            aspect='auto',
            origin='lower',
            cmap='RdBu')
        ax2.set_xlabel("v")
        ax2.set_ylabel("u")
        ax2.set_title("Jacobian Determinant")
        plt.colorbar(im, ax=ax2)
        plt.tight_layout()

        positive_ratio = np.sum(np.sign(J_map) > 0) / J_map.size * 100
        print(f"[*] Positive Jacobian ratio: {positive_ratio:.1f}%")

    def visualize_mapping(self):
        if self.X_sol is None or self.Y_sol is None:
            print(
                "No data to display, please run solve() or load_results() first."
            )
            return

        Xc_init, Yc_init, Zc_init, X_init, Y_init = self.initial_guess(
            mode=self.initial)
        if X_init is None: X_init, Y_init = self.Xc, self.Yc

        fig = plt.figure(figsize=(12, 5))
        ax1 = fig.add_subplot(121)
        sc1 = ax1.scatter(X_init.flatten() * 1e3,
                          Y_init.flatten() * 1e3,
                          c=self.U.flatten(),
                          s=5,
                          cmap='viridis',
                          alpha=0.6)
        ax1.set_xlabel("x (mm)")
        ax1.set_ylabel("y (mm)")
        ax1.set_title(f"Initial Mapping ({self.initial})")
        ax1.axis('equal')
        plt.colorbar(sc1, ax=ax1, label='u (normalized)')

        ax2 = fig.add_subplot(122)
        sc2 = ax2.scatter(self.X_sol.flatten() * 1e3,
                          self.Y_sol.flatten() * 1e3,
                          c=self.U.flatten(),
                          s=5,
                          cmap='viridis',
                          alpha=0.6)
        ax2.set_xlabel("x (mm)")
        ax2.set_ylabel("y (mm)")
        ax2.set_title("Optimized Mapping")
        ax2.axis('equal')
        plt.colorbar(sc2, ax=ax2, label='u (normalized)')
        plt.tight_layout()

    def create_animation(self, filename="homotopy_process.gif"):
        if not hasattr(self, 'history') or not self.history.get('alpha'):
            print(
                "[*] No history data or data is empty, skipping animation generation."
            )
            return

        rc_params = {
            'font.family': 'sans-serif',
            'font.sans-serif': ['Arial'],
            'axes.titlesize': 16,
            'axes.labelsize': 14,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'legend.fontsize': 12
        }

        with plt.rc_context(rc_params):
            fig = plt.figure(figsize=(16, 5))
            ax1 = fig.add_subplot(131, projection='3d')
            ax2 = fig.add_subplot(132)
            ax3 = fig.add_subplot(133)
            steps = len(self.history['alpha'])

            def update(frame):
                ax1.clear()
                ax2.clear()
                ax3.clear()
                alpha = self.history['alpha'][frame]

                Xmid, Ymid, Zmid = self.history['Xmid'][frame], self.history[
                    'Ymid'][frame], self.history['Zmid'][frame]
                skip = max(1, self.N // 15)
                ax1.plot_surface(Xmid[::skip, ::skip] * 1e3,
                                 Ymid[::skip, ::skip] * 1e3,
                                 Zmid[::skip, ::skip],
                                 alpha=0.7,
                                 cmap='plasma')
                ax1.set_xlabel("x (mm)")
                ax1.set_ylabel("y (mm)")
                ax1.set_zlabel("z (m)")
                ax1.set_title(f"Target Surface ($\\alpha$={alpha:.2f})")
                lim = self.bounds * 1e3
                ax1.set_xlim([-lim, lim])
                ax1.set_ylim([-lim, lim])
                ax1.set_zlim([self.u_min, self.u_max])

                X_opt, Y_opt = self.history['X_opt'][frame], self.history[
                    'Y_opt'][frame]
                ax2.scatter(X_opt.flatten() * 1e3,
                            Y_opt.flatten() * 1e3,
                            c=self.U.flatten(),
                            s=5,
                            cmap='viridis',
                            alpha=0.6)
                ax2.set_xlabel("x (mm)")
                ax2.set_ylabel("y (mm)")
                ax2.set_title("Mapping Function")
                ax2.set_aspect('equal', adjustable='box')
                ax2.set_xlim([-lim, lim])
                ax2.set_ylim([-lim, lim])

                alphas = np.array(self.history['alpha'][:frame + 1])
                res_init = np.array(self.history['res_init'][:frame + 1])
                res_opt = np.array(self.history['res_opt'][:frame + 1])
                x_data = np.repeat(alphas, 2)
                y_data = np.empty(2 * len(alphas))
                y_data[0::2] = res_init
                y_data[1::2] = res_opt

                ax3.plot(x_data,
                         y_data,
                         'b-',
                         marker='o',
                         markersize=4,
                         linewidth=1.5,
                         label='Residual Track')
                ax3.set_xlim([-0.05, 1.05])
                ax3.set_yscale('log')
                ax3.set_xlabel("Homotopy $\\alpha$")
                ax3.set_ylabel("Residual Norm (Log)")
                ax3.set_title("Residual Evolution")
                ax3.legend(loc='upper right')
                plt.tight_layout()

            print(f"[*] Generating homotopy animation ({steps} frames)...")
            anim = animation.FuncAnimation(fig,
                                           update,
                                           frames=steps,
                                           interval=250)
            try:
                anim.save(filename, writer='pillow')
                print(f"[*] Animation saved to: {filename}")
            except Exception as e:
                print(f"[!] Failed to save animation: {e}")
                pass
            finally:
                plt.close(fig)


# ==========================================
# 3. Phase Retrieval & Wave Propagation
# ==========================================


class WavePropagator:

    def __init__(self, optimizer: CausticOptimizer, wl=532e-9, N_sim=1024):
        self.opt = optimizer
        self.L = optimizer.bounds
        self.N_sim = N_sim
        self.dx = 2 * self.L / N_sim
        self.wl = wl
        self.k = 2 * np.pi / wl

    def recover_phase(self, method='global_lsqr'):
        print(f"[*] Starting phase recovery using {method}...")
        X, Y = self.opt.X_sol, self.opt.Y_sol
        Xc, Yc, Zc = self.opt.Xc, self.opt.Yc, self.opt.Zc

        # Geometrical Optics Ray Law: grad_S(Phi) = k * (C - S) / z
        mask = np.abs(Zc) > 1e-12
        dPhi_dx, dPhi_dy = np.zeros_like(Zc), np.zeros_like(Zc)
        dPhi_dx[mask] = self.k * (Xc[mask] - X[mask]) / Zc[mask]
        dPhi_dy[mask] = self.k * (Yc[mask] - Y[mask]) / Zc[mask]

        if method == 'global_lsqr':
            return self.recover_phase_global_lsqr(dPhi_dx, dPhi_dy, X, Y)
        else:
            raise ValueError(f"Unknown phase recovery method: {method}")

    def recover_phase_global_lsqr(self, dPhi_dx, dPhi_dy, X, Y):
        dX_du, dX_dv = self.opt.get_gradients(X, is_cyclic=False, axis=0), \
            self.opt.get_gradients(X, self.opt.target.is_cyclic_v, axis=1)
        dY_du, dY_dv = self.opt.get_gradients(Y, is_cyclic=False, axis=0), \
            self.opt.get_gradients(Y, self.opt.target.is_cyclic_v, axis=1)

        G_u = dPhi_dx * dX_du + dPhi_dy * dY_du
        G_v = dPhi_dx * dX_dv + dPhi_dy * dY_dv

        Nu, Nv = X.shape
        du, dv = self.opt.du, self.opt.dv

        rows, cols, data, b_vec = [], [], [], []
        row_idx = 0
        for i in range(Nu - 1):
            for j in range(Nv):
                idx, idx_next = i * Nv + j, (i + 1) * Nv + j
                rows.extend([row_idx, row_idx])
                cols.extend([idx_next, idx])
                data.extend([1.0, -1.0])
                b_vec.append(G_u[i, j] * du)
                row_idx += 1
        for i in range(Nu):
            for j in range(Nv - 1):
                idx, idx_next = i * Nv + j, i * Nv + (j + 1)
                rows.extend([row_idx, row_idx])
                cols.extend([idx_next, idx])
                data.extend([1.0, -1.0])
                b_vec.append(G_v[i, j] * dv)
                row_idx += 1

        A = sp.coo_matrix((data, (rows, cols)), shape=(row_idx, Nu * Nv))
        b = np.array(b_vec)

        phi_flat = lsqr(A, b, damp=1e-8, atol=1e-10, btol=1e-10, show=False)[0]
        self.Phi_map = phi_flat.reshape(Nu, Nv)
        self.Phi_map -= np.min(self.Phi_map)

        G_u_recon = np.gradient(self.Phi_map, du, axis=0)
        G_v_recon = np.gradient(self.Phi_map, dv, axis=1)
        error_u = np.median(np.abs(G_u_recon - G_u))
        error_v = np.median(np.abs(G_v_recon - G_v))

        print("   Phase calculation completed:")
        print(f"     - Median error of u-direction gradient: {error_u:.2e}")
        print(f"     - Median error of v-direction gradient: {error_v:.2e}")
        print(
            f"     - Phase range: [{self.Phi_map.min():.2f}, {self.Phi_map.max():.2f}] rad"
        )
        return self.Phi_map

    def interpolate_and_mask(self):
        print("[*] Interpolating phase over Cartesian grid...")
        x = y = np.linspace(-self.L, self.L, self.N_sim)
        self.xx, self.yy = np.meshgrid(x, y)

        if self.opt.target.is_cyclic_v:
            X_train = self.opt.X_sol[:, :-1]
            Y_train = self.opt.Y_sol[:, :-1]
            Phi_train = self.Phi_map[:, :-1]
        else:
            X_train = self.opt.X_sol.copy()
            Y_train = self.opt.Y_sol.copy()
            Phi_train = self.Phi_map.copy()

        points = np.column_stack((X_train.flatten(), Y_train.flatten()))
        phi_grid = griddata(points,
                            Phi_train.flatten(), (self.xx, self.yy),
                            method='linear',
                            fill_value=0)

        X, Y = self.opt.X_sol, self.opt.Y_sol
        boundary_verts = np.column_stack(
            (np.concatenate([X[0, :], X[:, -1], X[-1, ::-1], X[::-1, 0]]),
             np.concatenate([Y[0, :], Y[:, -1], Y[-1, ::-1], Y[::-1, 0]])))
        path = Path(boundary_verts)

        pts_grid = np.column_stack((self.xx.flatten(), self.yy.flatten()))
        is_inside = path.contains_points(pts_grid)
        mask = is_inside.reshape(self.N_sim, self.N_sim).astype(float)

        self.E0 = mask * np.exp(1j * phi_grid * mask)

        target_name_clean = self.opt.target.name.replace(' ', '_').replace(
            '-', '_')
        print("   Phase interpolation completed.")
        savemat(f"PhaseMask_{target_name_clean}.mat", {'E0': self.E0})
        return self.E0

    def visualize_source(self):
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        im1 = axes[0].imshow(
            np.angle(self.E0),
            extent=[-self.L * 1e3, self.L * 1e3, -self.L * 1e3, self.L * 1e3],
            cmap='twilight',
            origin='lower')
        axes[0].set_title("Source Phase (rad)")
        axes[0].set_xlabel("x (mm)")
        axes[0].set_ylabel("y (mm)")
        plt.colorbar(im1, ax=axes[0])

        im2 = axes[1].imshow(
            np.abs(self.E0),
            extent=[-self.L * 1e3, self.L * 1e3, -self.L * 1e3, self.L * 1e3],
            cmap='hot',
            origin='lower')
        axes[1].set_title("Source Amplitude")
        axes[1].set_xlabel("x (mm)")
        axes[1].set_ylabel("y (mm)")
        plt.colorbar(im2, ax=axes[1])
        plt.tight_layout()

    def visualize_uv_phase(self):
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        im1 = ax.imshow(self.Phi_map,
                        extent=[
                            self.opt.v_min, self.opt.v_max, self.opt.u_min,
                            self.opt.u_max
                        ],
                        aspect='auto',
                        cmap='twilight',
                        origin='lower')
        ax.set_title("Phase in (u,v) Space (rad)")
        ax.set_xlabel("v")
        ax.set_ylabel("u")
        plt.colorbar(im1, ax=ax)

    def propagate(self, z):
        fx = fftfreq(self.N_sim, 2 * self.L / self.N_sim)
        FX, FY = np.meshgrid(fx, fx)
        H = np.exp(
            1j * self.k * z *
            np.sqrt(np.maximum(0, 1 - (self.wl * FX)**2 - (self.wl * FY)**2)))
        return ifft2(fft2(self.E0) * H)

    def evaluate_metrics(self,
                         z,
                         I_sim,
                         threshold_rel=0.1,
                         confinement_width_mm=0.1,
                         pcc_type='gaussian',
                         pcc_sigma=3):
        """
        Compute quantitative evaluation metrics.

        Parameters:
            z: propagation distance
            I_sim: simulated intensity distribution
            threshold_rel: relative intensity threshold for RMSD calculation
            confinement_width_mm: width of tubular region for ECE (mm)
            pcc_type: 'binary' (ideal binary) or 'gaussian' (Gaussian blur)
            pcc_sigma: Gaussian blur sigma (pixels)

        Returns:
            iw_rmsd: intensity-weighted RMSD (mm)
            ece: energy confinement efficiency (0-1)
            pcc: Pearson correlation coefficient (-1 to 1)
        """
        target = self.opt.target
        v_hires = np.linspace(target.v_range[0], target.v_range[1], 2000)
        xc_t, yc_t, _ = target.parameterization(np.full_like(v_hires, z),
                                                v_hires)
        curve_points = np.column_stack((xc_t, yc_t))

        X, Y = np.meshgrid(np.linspace(-self.L, self.L, self.N_sim),
                           np.linspace(-self.L, self.L, self.N_sim))

        I_max = np.max(I_sim)
        if I_max < 1e-12: return 0, 0, 0

        mask_rmsd = I_sim > (I_max * threshold_rel)
        if not np.any(mask_rmsd):
            iw_rmsd = float('inf')
        else:
            dists, _ = cKDTree(curve_points).query(
                np.column_stack((X[mask_rmsd], Y[mask_rmsd])))
            iw_rmsd = np.sqrt(
                np.sum(I_sim[mask_rmsd] * dists**2) /
                np.sum(I_sim[mask_rmsd])) * 1e3

        idx_x = ((xc_t + self.L) / (2 * self.L) * (self.N_sim - 1)).astype(int)
        idx_y = ((yc_t + self.L) / (2 * self.L) * (self.N_sim - 1)).astype(int)
        valid_idx = (idx_x >= 0) & (idx_x < self.N_sim) & (idx_y >= 0) & (
            idx_y < self.N_sim)
        idx_x, idx_y = idx_x[valid_idx], idx_y[valid_idx]

        binary_curve = np.zeros_like(I_sim)
        binary_curve[idx_y, idx_x] = 1

        dist_map_mm = distance_transform_edt(1 - binary_curve) * (
            2 * self.L / self.N_sim) * 1e3
        ece = np.sum(
            I_sim[dist_map_mm <= confinement_width_mm]) / np.sum(I_sim)

        reference_img = gaussian_filter(
            binary_curve.astype(float), sigma=pcc_sigma
        ) if pcc_type == 'gaussian' else binary_curve.astype(float)
        pcc = np.corrcoef(I_sim.flatten(), reference_img.flatten())[0, 1]

        return iw_rmsd, ece, pcc

    def visualize_propagation_series(self, z_planes, pcc_type='gaussian'):
        print("[*] Propagating and evaluating metrics...")
        n_planes = len(z_planes)
        plt.figure(figsize=(3 * n_planes, 6))

        for i, z in enumerate(z_planes):
            I = np.abs(self.propagate(z))**2
            rmsd, ece, pcc = self.evaluate_metrics(z, I, pcc_type=pcc_type)

            plt.subplot(1, n_planes, i + 1)
            plt.title(
                f"Z={z}m\nRMSD={rmsd:.3f}mm\nECE={ece:.1%}\nPCC={pcc:.3f}",
                fontsize=9)
            extent = [-self.L * 1e3, self.L * 1e3, -self.L * 1e3, self.L * 1e3]
            plt.imshow(I,
                       extent=extent,
                       cmap='inferno',
                       vmax=np.percentile(I, 99),
                       origin='lower')

            v_hires = np.linspace(self.opt.target.v_range[0],
                                  self.opt.target.v_range[1], 200)
            xc_t, yc_t, _ = self.opt.target.parameterization(
                np.full_like(v_hires, z), v_hires)
            plt.plot(xc_t * 1e3,
                     yc_t * 1e3,
                     'w--',
                     lw=1.5,
                     alpha=0.6,
                     label='Theory')
            plt.axis('off')

        plt.tight_layout()


# ==========================================
# 4. Main Execution
# ==========================================


def main(config):
    FORCE_RERUN = False
    script_dir = os.path.dirname(os.path.abspath(__file__))
    CACHE_FILE = os.path.join(script_dir, "solver_cache.npz")

    p_grid, p_solver, p_weights, p_prop = config['grid'], config[
        'solver'], config['weights'], config['propagation']

    # Step 1: Select target geometry corresponding to figures in the paper
    # target = LinearExpandingEllipse()  # Fig 1
    # target = CubicSurface()  # Fig 3 & Fig 4
    # target = ConvergingCardioid()       # Fig 5a
    # target = AstigmaticEnvelope()       # Fig 5b
    # target = SkewedParabolicTrough()    # Fig 5c
    target = DriftingOvoidalCone()      # Fig 5d
    # target = PearceyLikeCaustic()       # Fig 5e
    # target = HelicoidalTrefoil()  # Fig 5f

    # Step 2: Initialize Optimizer
    opt = CausticOptimizer(target,
                           init_mode=p_solver['init_mode'],
                           N_opt=p_grid['N_opt'],
                           bounds_mm=p_grid['bounds_mm'],
                           loss_weights=p_weights)

    # Step 3: Solve Mapping (With Local Cache Support)
    if os.path.exists(CACHE_FILE) and not FORCE_RERUN:
        print(f"[*] Cache file detected: {CACHE_FILE}")
        if not opt.load_results(CACHE_FILE):
            print("   Loading failed, recalculating...")
            opt.solve(tol=p_solver['tol'],
                      max_iter=p_solver['max_iter'],
                      steps=p_solver['homotopy_steps'],
                      verbose=1)
            opt.save_results(CACHE_FILE)
        else:
            print(
                f"[*] Skipping optimization, using cached results: {CACHE_FILE}"
            )
    else:
        print(f"[*] Starting new computation...")
        opt.solve(tol=p_solver['tol'],
                  max_iter=p_solver['max_iter'],
                  steps=p_solver['homotopy_steps'],
                  verbose=1)
        opt.save_results(CACHE_FILE)

    opt.visualize_results()
    opt.visualize_mapping()
    opt.create_animation(
        filename=os.path.join(script_dir, "homotopy_process.gif"))

    # Step 4: Phase Retrieval & Synthesis
    prop = WavePropagator(opt, wl=532e-9, N_sim=p_prop['N_sim'])
    prop.recover_phase(method='global_lsqr')
    prop.visualize_uv_phase()
    prop.interpolate_and_mask()
    prop.visualize_source()

    # Step 5: Field Propagation & Evaluation
    prop.visualize_propagation_series(p_prop['z_planes'], pcc_type='gaussian')
    plt.show()


if __name__ == "__main__":
    CONFIG = {
        'grid': {
            'N_opt': 64,
            'bounds_mm': 4.0
        },
        'solver': {
            'init_mode': 'homotopy',
            # 'init_mode': 'linear',
            # 'init_mode': 'projection',
            # 'init_mode': 'tangent',
            'tol': 1e-6,
            'max_iter': 500,
            'homotopy_steps': 100
        },
        'weights': {
            'eq1': 1e0,
            'eq2': 1e0,
            'jac': 1e0,
            'bnd': 1e3,
            'lap': 5e-3,
            'cyclic': 1e3
        },
        'propagation': {
            'N_sim': 1024,
            'z_planes': [0.16, 0.32, 0.4, 0.48, 0.64]
        }
    }
    main(CONFIG)
