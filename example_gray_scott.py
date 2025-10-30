"""
Gray-Scott Model Example

The Gray-Scott model produces fascinating pattern formation including:
- Spots
- Stripes
- Spirals
- Chaotic patterns

This example demonstrates the classic "mitosis" pattern.

Author: Vivek Singh Yadav
"""

import numpy as np
from reaction_diffusion_solver import (
    ReactionDiffusionSystem,
    ReactionDiffusionSolver,
    gray_scott_reaction,
    random_perturbation,
    localized_perturbation
)
from visualization import plot_evolution, create_animation, plot_snapshot

# Set random seed for reproducibility
np.random.seed(42)

# System configuration
system = ReactionDiffusionSystem(
    Lx=2.5,
    Ly=2.5,
    Nx=256,
    Ny=256,
    T=10000,         # Long simulation time to see pattern development
    dt=1.0,
    Du=2e-5,         # Diffusion coefficient for u
    Dv=1e-5,         # Diffusion coefficient for v (slower)
    bc_type='periodic'
)

# Create solver
solver = ReactionDiffusionSolver(system)

# Initial conditions: uniform state with small random perturbations
# For Gray-Scott: u=1 (substrate), v=0 (activator) at equilibrium
u0, v0 = random_perturbation(system, u_base=1.0, v_base=0.0, noise_level=0.01)

# Add a localized perturbation to seed pattern formation
center_u, center_v = localized_perturbation(
    system,
    u_base=0.0,
    v_base=0.0,
    center_value_u=0.5,
    center_value_v=0.25,
    radius=0.2
)
u0 = u0 - center_u + 0.5
v0 = v0 + center_v

solver.set_initial_conditions(u0, v0)

# Parameters for interesting "mitosis" pattern
# Experiment with different F and k values:
# - F=0.04, k=0.06: mitosis (spots that divide)
# - F=0.03, k=0.06: stripes
# - F=0.05, k=0.065: spirals
# - F=0.014, k=0.054: maze patterns
F = 0.04
k = 0.06

print(f"\nGray-Scott Model Simulation")
print(f"Parameters: F={F}, k={k}")
print(f"Expected pattern: Mitosis (dividing spots)\n")

# Solve the system
u_history, v_history, times = solver.solve(
    gray_scott_reaction,
    F=F,
    k=k,
    save_every=100
)

# Visualizations
print("\n" + "="*60)
print("Creating visualizations...")
print("="*60)

# Plot evolution
plot_evolution(
    u_history,
    v_history,
    times,
    n_snapshots=6,
    title=f"Gray-Scott Model (F={F}, k={k})",
    cmap='RdYlBu_r'
)

# Plot final state
plot_snapshot(
    u_history[-1],
    v_history[-1],
    times[-1],
    title=f"Gray-Scott Final State (F={F}, k={k})",
    cmap='RdYlBu_r'
)

# Create animation (optional - uncomment to generate)
# create_animation(
#     u_history,
#     v_history,
#     times,
#     title=f"Gray-Scott Model (F={F}, k={k})",
#     save_path="gray_scott_animation.gif",
#     fps=10,
#     cmap='RdYlBu_r'
# )

print("\n" + "="*60)
print("Try different parameter values:")
print("  F=0.03, k=0.06   -> Stripes")
print("  F=0.05, k=0.065  -> Spirals")
print("  F=0.014, k=0.054 -> Maze patterns")
print("="*60)
