"""
Time-Dependent Reaction-Diffusion PDE Solver

This module implements numerical solvers for reaction-diffusion partial differential equations (PDEs)
of the form:
    ∂u/∂t = D∇²u + R(u, v, ...)
    ∂v/∂t = D'∇²v + R'(u, v, ...)

where D, D' are diffusion coefficients and R, R' are reaction terms.

Author: Vivek Singh Yadav
"""

import numpy as np
from typing import Callable, Tuple, Optional, List
from dataclasses import dataclass


@dataclass
class ReactionDiffusionSystem:
    """Configuration for a reaction-diffusion system."""

    # Spatial domain
    Lx: float = 1.0  # Domain length in x
    Ly: float = 1.0  # Domain length in y
    Nx: int = 128    # Number of grid points in x
    Ny: int = 128    # Number of grid points in y

    # Time domain
    T: float = 100.0      # Total simulation time
    dt: float = 0.01      # Time step

    # Diffusion coefficients
    Du: float = 1e-5      # Diffusion coefficient for u
    Dv: float = 5e-6      # Diffusion coefficient for v

    # Boundary conditions: 'periodic', 'neumann', 'dirichlet'
    bc_type: str = 'periodic'

    def __post_init__(self):
        """Calculate derived quantities."""
        self.dx = self.Lx / self.Nx
        self.dy = self.Ly / self.Ny
        self.x = np.linspace(0, self.Lx, self.Nx)
        self.y = np.linspace(0, self.Ly, self.Ny)
        self.X, self.Y = np.meshgrid(self.x, self.y)

        # Stability check for explicit schemes
        self.r_x = self.Du * self.dt / (self.dx ** 2)
        self.r_y = self.Du * self.dt / (self.dy ** 2)
        if self.r_x + self.r_y > 0.5:
            print(f"Warning: Stability condition violated. rx={self.r_x:.4f}, ry={self.r_y:.4f}")
            print(f"Consider reducing dt or increasing grid spacing.")


class ReactionDiffusionSolver:
    """
    Solves time-dependent reaction-diffusion PDEs using finite difference methods.

    The solver uses explicit Euler for time integration and second-order central
    differences for spatial derivatives.
    """

    def __init__(self, system: ReactionDiffusionSystem):
        """
        Initialize the solver.

        Args:
            system: Configuration object for the reaction-diffusion system
        """
        self.sys = system

        # Storage for solution
        self.u = None
        self.v = None
        self.time = 0.0
        self.step = 0

        # History for visualization
        self.history_u = []
        self.history_v = []
        self.history_times = []

    def set_initial_conditions(self, u0: np.ndarray, v0: Optional[np.ndarray] = None):
        """
        Set initial conditions for the system.

        Args:
            u0: Initial condition for u field
            v0: Initial condition for v field (optional, for coupled systems)
        """
        self.u = u0.copy()
        self.v = v0.copy() if v0 is not None else None
        self.time = 0.0
        self.step = 0

        # Store initial condition
        self.history_u = [self.u.copy()]
        self.history_v = [self.v.copy()] if self.v is not None else []
        self.history_times = [0.0]

    def laplacian(self, field: np.ndarray) -> np.ndarray:
        """
        Compute the Laplacian ∇²field using second-order finite differences.

        Args:
            field: 2D array representing the field

        Returns:
            Laplacian of the field
        """
        lap = np.zeros_like(field)

        if self.sys.bc_type == 'periodic':
            # Periodic boundary conditions
            lap = (
                (np.roll(field, 1, axis=0) - 2 * field + np.roll(field, -1, axis=0)) / self.sys.dx**2 +
                (np.roll(field, 1, axis=1) - 2 * field + np.roll(field, -1, axis=1)) / self.sys.dy**2
            )
        elif self.sys.bc_type == 'neumann':
            # Neumann (zero-flux) boundary conditions
            lap[1:-1, 1:-1] = (
                (field[2:, 1:-1] - 2 * field[1:-1, 1:-1] + field[:-2, 1:-1]) / self.sys.dx**2 +
                (field[1:-1, 2:] - 2 * field[1:-1, 1:-1] + field[1:-1, :-2]) / self.sys.dy**2
            )
            # Boundaries (Neumann: derivative = 0)
            lap[0, :] = lap[1, :]
            lap[-1, :] = lap[-2, :]
            lap[:, 0] = lap[:, 1]
            lap[:, -1] = lap[:, -2]
        elif self.sys.bc_type == 'dirichlet':
            # Dirichlet (fixed value) boundary conditions
            lap[1:-1, 1:-1] = (
                (field[2:, 1:-1] - 2 * field[1:-1, 1:-1] + field[:-2, 1:-1]) / self.sys.dx**2 +
                (field[1:-1, 2:] - 2 * field[1:-1, 1:-1] + field[1:-1, :-2]) / self.sys.dy**2
            )
            # Keep boundaries fixed (already zero in lap)

        return lap

    def step_forward(self, reaction_func: Callable, **params):
        """
        Advance the solution by one time step using forward Euler.

        Args:
            reaction_func: Function that computes reaction terms R(u, v, **params)
            **params: Additional parameters for the reaction function
        """
        # Compute Laplacians
        lap_u = self.laplacian(self.u)

        # Get reaction terms
        if self.v is not None:
            lap_v = self.laplacian(self.v)
            Ru, Rv = reaction_func(self.u, self.v, **params)

            # Update both fields
            self.u = self.u + self.sys.dt * (self.sys.Du * lap_u + Ru)
            self.v = self.v + self.sys.dt * (self.sys.Dv * lap_v + Rv)
        else:
            # Single equation system
            Ru = reaction_func(self.u, **params)
            self.u = self.u + self.sys.dt * (self.sys.Du * lap_u + Ru)

        self.time += self.sys.dt
        self.step += 1

    def solve(self,
              reaction_func: Callable,
              save_every: int = 100,
              **params) -> Tuple[List[np.ndarray], List[np.ndarray], List[float]]:
        """
        Solve the reaction-diffusion system over the entire time domain.

        Args:
            reaction_func: Function that computes reaction terms
            save_every: Save solution every N steps
            **params: Additional parameters for the reaction function

        Returns:
            Tuple of (u_history, v_history, times)
        """
        n_steps = int(self.sys.T / self.sys.dt)

        print(f"Solving reaction-diffusion system...")
        print(f"Grid: {self.sys.Nx}x{self.sys.Ny}, Time steps: {n_steps}")
        print(f"dx={self.sys.dx:.6f}, dy={self.sys.dy:.6f}, dt={self.sys.dt:.6f}")

        for n in range(n_steps):
            self.step_forward(reaction_func, **params)

            # Save snapshots
            if n % save_every == 0:
                self.history_u.append(self.u.copy())
                if self.v is not None:
                    self.history_v.append(self.v.copy())
                self.history_times.append(self.time)

                if n % (save_every * 10) == 0:
                    print(f"Step {n}/{n_steps}, t={self.time:.2f}")

        print(f"Simulation complete! Final time: {self.time:.2f}")

        return self.history_u, self.history_v, self.history_times

    def get_current_state(self) -> Tuple[np.ndarray, Optional[np.ndarray], float]:
        """
        Get the current state of the system.

        Returns:
            Tuple of (u, v, time)
        """
        return self.u.copy(), self.v.copy() if self.v is not None else None, self.time


# ==============================================================================
# Reaction Functions for Classic Systems
# ==============================================================================

def fisher_reaction(u: np.ndarray, r: float = 1.0, K: float = 1.0) -> np.ndarray:
    """
    Fisher's equation (logistic growth): R(u) = r*u*(1 - u/K)

    Models population dynamics with logistic growth.
    """
    return r * u * (1 - u / K)


def gray_scott_reaction(u: np.ndarray, v: np.ndarray,
                        F: float = 0.04, k: float = 0.06) -> Tuple[np.ndarray, np.ndarray]:
    """
    Gray-Scott reaction terms:
        R_u = -u*v² + F*(1-u)
        R_v = u*v² - (F+k)*v

    Classic model that produces diverse patterns including spots, stripes, and spirals.

    Args:
        u, v: Concentration fields
        F: Feed rate (typically 0.01-0.08)
        k: Kill rate (typically 0.04-0.08)
    """
    uvv = u * v * v
    Ru = -uvv + F * (1 - u)
    Rv = uvv - (F + k) * v
    return Ru, Rv


def fitzhugh_nagumo_reaction(u: np.ndarray, v: np.ndarray,
                             a: float = 0.0, b: float = 1.0,
                             tau: float = 10.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    FitzHugh-Nagumo reaction terms:
        R_u = u - u³/3 - v + a
        R_v = (u + b) / tau

    Simplified model of neuronal excitation and inhibition.
    """
    Ru = u - u**3 / 3 - v + a
    Rv = (u + b) / tau
    return Ru, Rv


def brusselator_reaction(u: np.ndarray, v: np.ndarray,
                        A: float = 4.5, B: float = 8.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Brusselator reaction terms (theoretical chemical oscillator):
        R_u = A + u²*v - B*u - u
        R_v = B*u - u²*v

    Exhibits oscillatory behavior and Turing patterns.
    """
    u2v = u * u * v
    Ru = A + u2v - B * u - u
    Rv = B * u - u2v
    return Ru, Rv


def schnakenberg_reaction(u: np.ndarray, v: np.ndarray,
                         a: float = 0.1, b: float = 0.9) -> Tuple[np.ndarray, np.ndarray]:
    """
    Schnakenberg (trimolecular) reaction terms:
        R_u = a - u + u²*v
        R_v = b - u²*v

    Another model exhibiting Turing patterns.
    """
    u2v = u * u * v
    Ru = a - u + u2v
    Rv = b - u2v
    return Ru, Rv


# ==============================================================================
# Initial Condition Generators
# ==============================================================================

def random_perturbation(system: ReactionDiffusionSystem,
                       u_base: float = 0.0, v_base: float = 0.0,
                       noise_level: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """Generate random initial conditions with small perturbations."""
    u0 = u_base + noise_level * np.random.random((system.Ny, system.Nx))
    v0 = v_base + noise_level * np.random.random((system.Ny, system.Nx))
    return u0, v0


def localized_perturbation(system: ReactionDiffusionSystem,
                           u_base: float = 0.0, v_base: float = 0.0,
                           center_value_u: float = 1.0,
                           center_value_v: float = 1.0,
                           radius: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """Generate initial conditions with a localized perturbation at the center."""
    u0 = np.ones((system.Ny, system.Nx)) * u_base
    v0 = np.ones((system.Ny, system.Nx)) * v_base

    # Add localized perturbation
    center_x, center_y = system.Lx / 2, system.Ly / 2
    r_squared = (system.X - center_x)**2 + (system.Y - center_y)**2
    mask = r_squared < radius**2

    u0[mask] = center_value_u
    v0[mask] = center_value_v

    return u0, v0


def stripe_perturbation(system: ReactionDiffusionSystem,
                       u_base: float = 0.0, v_base: float = 0.0,
                       amplitude: float = 0.2,
                       wavelength: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """Generate initial conditions with sinusoidal stripes."""
    k = 2 * np.pi / wavelength
    u0 = u_base + amplitude * np.sin(k * system.X)
    v0 = v_base + amplitude * np.cos(k * system.Y)
    return u0, v0
