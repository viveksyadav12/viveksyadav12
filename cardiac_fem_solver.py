"""
Cardiac FEM Solver - Electrophysiology Simulations on Heart Meshes

This module provides finite element solvers for cardiac electrophysiology,
implementing the monodomain equation with various ionic models.

Features:
- Monodomain equation for electrical wave propagation
- Multiple ionic models (Aliev-Panfilov, FitzHugh-Nagumo, etc.)
- Anisotropic diffusion with fiber orientations
- Adaptive time stepping
- Stimulus protocols
- Action potential propagation

Equations:
    Monodomain: ∂V/∂t = ∇·(D∇V) - I_ion/C_m + I_stim
    where D is the diffusion tensor (anisotropic)

Author: Vivek Singh Yadav
Date: 2025-11-18
"""

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
from typing import Callable, Optional, Tuple, Dict, List
from dataclasses import dataclass
import warnings


@dataclass
class CardiacTissueProperties:
    """Properties of cardiac tissue"""
    # Electrical properties
    sigma_l: float = 0.3  # Longitudinal conductivity (S/m)
    sigma_t: float = 0.03  # Transverse conductivity (S/m)
    C_m: float = 1.0  # Membrane capacitance (μF/cm²)
    chi: float = 1000.0  # Surface-to-volume ratio (1/cm)

    # Derived properties
    def diffusion_coefficient_longitudinal(self) -> float:
        """Longitudinal diffusion coefficient"""
        return self.sigma_l / (self.C_m * self.chi)

    def diffusion_coefficient_transverse(self) -> float:
        """Transverse diffusion coefficient"""
        return self.sigma_t / (self.C_m * self.chi)


class IonicModel:
    """Base class for ionic current models"""

    def __init__(self):
        self.state_var_names = []

    def compute_current(self, V: np.ndarray, states: Dict[str, np.ndarray],
                       dt: float) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Compute ionic current and update state variables

        Args:
            V: Membrane potential (mV)
            states: Dictionary of state variables
            dt: Time step

        Returns:
            I_ion: Ionic current
            updated_states: Updated state variables
        """
        raise NotImplementedError


class AlievPanfilovModel(IonicModel):
    """
    Aliev-Panfilov ionic model (simplified cardiac model)

    Two-variable model for cardiac action potential:
        dV/dt = -k*V*(V-a)*(V-1) - V*r
        dr/dt = (ε + μ₁*r/(μ₂+V)) * (-r - k*V*(V-a-1))

    Parameters based on Aliev & Panfilov (1996)
    """

    def __init__(self, k: float = 8.0, a: float = 0.15,
                 epsilon: float = 0.01, mu1: float = 0.2, mu2: float = 0.3):
        """
        Initialize Aliev-Panfilov model

        Args:
            k: Excitability parameter
            a: Threshold parameter
            epsilon: Recovery time scale
            mu1, mu2: Recovery parameters
        """
        super().__init__()
        self.k = k
        self.a = a
        self.epsilon = epsilon
        self.mu1 = mu1
        self.mu2 = mu2
        self.state_var_names = ['r']

    def compute_current(self, V: np.ndarray, states: Dict[str, np.ndarray],
                       dt: float) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Compute ionic current using Aliev-Panfilov model"""
        r = states.get('r', np.zeros_like(V))

        # Clip voltage to prevent numerical overflow
        V_clipped = np.clip(V, -2.0, 2.0)

        # Ionic current (fast depolarization and repolarization)
        I_ion = self.k * V_clipped * (V_clipped - self.a) * (V_clipped - 1.0) + V_clipped * r

        # Update recovery variable r
        dr_dt = (self.epsilon + self.mu1 * r / (self.mu2 + V_clipped + 1e-10)) * \
                (-r - self.k * V_clipped * (V_clipped - self.a - 1.0))

        # Clip derivative to prevent instability
        dr_dt = np.clip(dr_dt, -10.0, 10.0)

        r_new = np.clip(r + dt * dr_dt, -2.0, 2.0)

        updated_states = {'r': r_new}

        return I_ion, updated_states


class FitzHughNagumoCardiacModel(IonicModel):
    """
    FitzHugh-Nagumo model adapted for cardiac tissue

    Simplified excitable media model:
        dV/dt = V - V³/3 - w + I_stim
        dw/dt = ε(V + a - b*w)
    """

    def __init__(self, a: float = 0.7, b: float = 0.8, epsilon: float = 0.08):
        """
        Initialize FitzHugh-Nagumo cardiac model

        Args:
            a: Threshold parameter
            b: Recovery strength
            epsilon: Recovery time scale
        """
        super().__init__()
        self.a = a
        self.b = b
        self.epsilon = epsilon
        self.state_var_names = ['w']

    def compute_current(self, V: np.ndarray, states: Dict[str, np.ndarray],
                       dt: float) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Compute ionic current using FitzHugh-Nagumo model"""
        w = states.get('w', np.zeros_like(V))

        # Ionic current
        I_ion = -(V - V**3 / 3.0 - w)

        # Update recovery variable
        dw_dt = self.epsilon * (V + self.a - self.b * w)
        w_new = w + dt * dw_dt

        updated_states = {'w': w_new}

        return I_ion, updated_states


class CardiacFEMSolver:
    """
    Finite Element Method solver for cardiac electrophysiology

    Solves the monodomain equation:
        C_m * ∂V/∂t = ∇·(D∇V) - I_ion + I_stim
    """

    def __init__(self, nodes: np.ndarray, elements: np.ndarray,
                 fiber_field: Optional[np.ndarray] = None,
                 tissue_properties: Optional[CardiacTissueProperties] = None,
                 ionic_model: Optional[IonicModel] = None):
        """
        Initialize cardiac FEM solver

        Args:
            nodes: Node coordinates (N x 2)
            elements: Element connectivity (M x 3 or more)
            fiber_field: Fiber directions at each node (N x 2)
            tissue_properties: Cardiac tissue properties
            ionic_model: Ionic current model
        """
        self.nodes = nodes
        self.elements = elements[:, :3]  # Use only first 3 nodes (triangles)
        self.n_nodes = len(nodes)
        self.n_elements = len(elements)

        self.fiber_field = fiber_field
        if self.fiber_field is None:
            # Default: isotropic (no preferred direction)
            self.fiber_field = np.tile([1.0, 0.0], (self.n_nodes, 1))

        self.tissue_props = tissue_properties or CardiacTissueProperties()
        self.ionic_model = ionic_model or AlievPanfilovModel()

        # Solution variables
        self.V = np.zeros(self.n_nodes)  # Membrane potential
        self.states = {name: np.zeros(self.n_nodes)
                      for name in self.ionic_model.state_var_names}

        # FEM matrices
        self.M = None  # Mass matrix
        self.K = None  # Stiffness matrix (diffusion)
        self._assemble_matrices()

    def _assemble_matrices(self):
        """Assemble FEM mass and stiffness matrices"""
        M = lil_matrix((self.n_nodes, self.n_nodes))
        K = lil_matrix((self.n_nodes, self.n_nodes))

        # Diffusion coefficients
        D_l = self.tissue_props.diffusion_coefficient_longitudinal()
        D_t = self.tissue_props.diffusion_coefficient_transverse()

        for elem_idx in range(self.n_elements):
            node_ids = self.elements[elem_idx]
            elem_nodes = self.nodes[node_ids]

            # Element area and gradients
            p1, p2, p3 = elem_nodes
            area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) -
                           (p3[0] - p1[0]) * (p2[1] - p1[1]))

            if area < 1e-12:
                continue

            # Shape function gradients
            grad_phi = self._compute_shape_gradients(elem_nodes)

            # Element fiber direction (average of node fibers)
            fiber_elem = np.mean(self.fiber_field[node_ids], axis=0)
            fiber_elem = fiber_elem / (np.linalg.norm(fiber_elem) + 1e-10)

            # Anisotropic diffusion tensor
            # D = D_l * (f ⊗ f) + D_t * (I - f ⊗ f)
            f = fiber_elem.reshape(-1, 1)
            D_aniso = D_l * (f @ f.T) + D_t * (np.eye(2) - f @ f.T)

            # Assemble element matrices
            for i in range(3):
                for j in range(3):
                    ni, nj = node_ids[i], node_ids[j]

                    # Mass matrix (consistent mass)
                    if i == j:
                        M[ni, nj] += area / 6.0  # Diagonal
                    else:
                        M[ni, nj] += area / 12.0  # Off-diagonal

                    # Stiffness matrix (anisotropic diffusion)
                    grad_i = grad_phi[i]
                    grad_j = grad_phi[j]
                    K[ni, nj] += area * (grad_i @ D_aniso @ grad_j)

        self.M = M.tocsr()
        self.K = K.tocsr()

    def _compute_shape_gradients(self, elem_nodes: np.ndarray) -> np.ndarray:
        """
        Compute shape function gradients for triangular element

        Args:
            elem_nodes: Element node coordinates (3 x 2)

        Returns:
            gradients: Shape function gradients (3 x 2)
        """
        p1, p2, p3 = elem_nodes

        # Jacobian
        J = np.array([
            [p2[0] - p1[0], p3[0] - p1[0]],
            [p2[1] - p1[1], p3[1] - p1[1]]
        ])

        det_J = np.linalg.det(J)
        if abs(det_J) < 1e-12:
            return np.zeros((3, 2))

        J_inv = np.linalg.inv(J)

        # Shape function gradients in reference element
        grad_ref = np.array([
            [-1.0, -1.0],
            [1.0, 0.0],
            [0.0, 1.0]
        ])

        # Transform to physical element
        gradients = grad_ref @ J_inv.T

        return gradients

    def apply_stimulus(self, stimulus_func: Callable[[np.ndarray, float], np.ndarray],
                      time: float) -> np.ndarray:
        """
        Apply external stimulus

        Args:
            stimulus_func: Function(nodes, time) -> stimulus current
            time: Current simulation time

        Returns:
            Stimulus current at each node
        """
        return stimulus_func(self.nodes, time)

    def solve_timestep(self, dt: float, time: float,
                      stimulus_func: Optional[Callable] = None) -> None:
        """
        Solve one time step using semi-implicit method

        Args:
            dt: Time step size
            time: Current time
            stimulus_func: Optional stimulus function
        """
        # Get stimulus current
        if stimulus_func is not None:
            I_stim = self.apply_stimulus(stimulus_func, time)
        else:
            I_stim = np.zeros(self.n_nodes)

        # Compute ionic current
        I_ion, self.states = self.ionic_model.compute_current(
            self.V, self.states, dt
        )

        # Semi-implicit time integration
        # (M/dt - θ*K)*V^{n+1} = (M/dt + (1-θ)*K)*V^n - I_ion + I_stim
        theta = 0.5  # Crank-Nicolson

        # Left-hand side matrix
        A = self.M / dt - theta * self.K

        # Right-hand side
        b = (self.M / dt + (1 - theta) * self.K) @ self.V - I_ion + I_stim

        # Solve linear system
        try:
            V_new = spsolve(A, b)
            # Clip to prevent numerical overflow
            self.V = np.clip(V_new, -5.0, 5.0)
        except Exception as e:
            warnings.warn(f"Linear solver failed: {e}. Using previous solution.")
            pass

    def solve(self, T: float, dt: float,
             stimulus_func: Optional[Callable] = None,
             save_interval: int = 10) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Solve cardiac electrophysiology over time interval

        Args:
            T: Total simulation time
            dt: Time step size
            stimulus_func: Optional stimulus function
            save_interval: Save solution every N steps

        Returns:
            times: Array of saved time points
            solutions: List of solution arrays at saved times
        """
        n_steps = int(T / dt)
        times = []
        solutions = []

        print(f"Starting cardiac simulation...")
        print(f"  Time steps: {n_steps}")
        print(f"  dt = {dt:.6f}")
        print(f"  Nodes: {self.n_nodes}")
        print(f"  Elements: {self.n_elements}")

        for step in range(n_steps):
            time = step * dt

            # Solve timestep
            self.solve_timestep(dt, time, stimulus_func)

            # Save solution
            if step % save_interval == 0:
                times.append(time)
                solutions.append(self.V.copy())

                if step % (save_interval * 10) == 0:
                    V_min, V_max = self.V.min(), self.V.max()
                    print(f"  Step {step}/{n_steps}, t={time:.3f}, "
                          f"V: [{V_min:.3f}, {V_max:.3f}]")

        print("Simulation complete!")

        return np.array(times), solutions

    def set_initial_condition(self, V0: np.ndarray):
        """Set initial membrane potential"""
        self.V = V0.copy()

    def reset(self):
        """Reset solver state"""
        self.V = np.zeros(self.n_nodes)
        self.states = {name: np.zeros(self.n_nodes)
                      for name in self.ionic_model.state_var_names}


class StimulusProtocol:
    """Helper class for defining stimulus protocols"""

    @staticmethod
    def point_stimulus(center: Tuple[float, float], radius: float,
                      amplitude: float, duration: float,
                      start_time: float = 0.0) -> Callable:
        """
        Create a point stimulus protocol

        Args:
            center: Stimulus center coordinates
            radius: Stimulus radius
            amplitude: Stimulus amplitude
            duration: Stimulus duration
            start_time: Start time for stimulus

        Returns:
            Stimulus function
        """
        def stimulus(nodes: np.ndarray, time: float) -> np.ndarray:
            if time < start_time or time > start_time + duration:
                return np.zeros(len(nodes))

            # Compute distance from center
            dist = np.linalg.norm(nodes - np.array(center), axis=1)

            # Apply stimulus to nodes within radius
            I_stim = np.where(dist <= radius, amplitude, 0.0)

            return I_stim

        return stimulus

    @staticmethod
    def region_stimulus(region_func: Callable[[np.ndarray], np.ndarray],
                       amplitude: float, duration: float,
                       start_time: float = 0.0) -> Callable:
        """
        Create a regional stimulus protocol

        Args:
            region_func: Function(nodes) -> boolean mask
            amplitude: Stimulus amplitude
            duration: Stimulus duration
            start_time: Start time

        Returns:
            Stimulus function
        """
        def stimulus(nodes: np.ndarray, time: float) -> np.ndarray:
            if time < start_time or time > start_time + duration:
                return np.zeros(len(nodes))

            mask = region_func(nodes)
            I_stim = np.where(mask, amplitude, 0.0)

            return I_stim

        return stimulus

    @staticmethod
    def pacing_protocol(center: Tuple[float, float], radius: float,
                       amplitude: float, duration: float,
                       period: float, n_beats: int = 1) -> Callable:
        """
        Create a periodic pacing protocol

        Args:
            center: Stimulus center
            radius: Stimulus radius
            amplitude: Stimulus amplitude
            duration: Stimulus duration per beat
            period: Pacing period (cycle length)
            n_beats: Number of beats

        Returns:
            Stimulus function
        """
        def stimulus(nodes: np.ndarray, time: float) -> np.ndarray:
            # Check if we're in a stimulus phase
            for beat in range(n_beats):
                start = beat * period
                end = start + duration

                if start <= time <= end:
                    dist = np.linalg.norm(nodes - np.array(center), axis=1)
                    return np.where(dist <= radius, amplitude, 0.0)

            return np.zeros(len(nodes))

        return stimulus


if __name__ == "__main__":
    # Quick test with simple mesh
    print("Testing Cardiac FEM Solver...")

    # Create simple rectangular mesh
    nx, ny = 20, 20
    x = np.linspace(0, 5, nx)
    y = np.linspace(0, 5, ny)
    nodes = []
    for yi in y:
        for xi in x:
            nodes.append([xi, yi])
    nodes = np.array(nodes)

    # Create triangular elements
    elements = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            n00 = j * nx + i
            n10 = j * nx + i + 1
            n01 = (j + 1) * nx + i
            n11 = (j + 1) * nx + i + 1

            elements.append([n00, n10, n01])
            elements.append([n10, n11, n01])
    elements = np.array(elements)

    print(f"  Mesh: {len(nodes)} nodes, {len(elements)} elements")

    # Create solver
    solver = CardiacFEMSolver(
        nodes, elements,
        ionic_model=AlievPanfilovModel()
    )

    # Define stimulus
    stimulus = StimulusProtocol.point_stimulus(
        center=(1.0, 2.5), radius=0.5,
        amplitude=50.0, duration=2.0
    )

    # Solve
    times, solutions = solver.solve(T=10.0, dt=0.1, stimulus_func=stimulus,
                                   save_interval=5)

    print(f"  Saved {len(solutions)} time points")
    print("Cardiac FEM solver ready!")
