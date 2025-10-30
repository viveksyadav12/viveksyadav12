"""
Turing Pattern Formation Example

Demonstrates spontaneous pattern formation in the Schnakenberg model,
a classic example of Turing instability.

Turing patterns arise from the interaction between:
- Short-range activation (autocatalysis)
- Long-range inhibition (different diffusion rates)

Author: Vivek Singh Yadav
"""

import numpy as np
from reaction_diffusion_solver import (
    ReactionDiffusionSystem,
    ReactionDiffusionSolver,
    schnakenberg_reaction,
    brusselator_reaction,
    random_perturbation
)
from visualization import (
    plot_evolution,
    plot_snapshot,
    plot_power_spectrum,
    create_animation
)

# Set random seed for reproducibility
np.random.seed(123)

print("="*70)
print("TURING PATTERN FORMATION - SCHNAKENBERG MODEL")
print("="*70)

# System configuration
# Key point: Different diffusion coefficients (Du << Dv) enable Turing patterns
system = ReactionDiffusionSystem(
    Lx=2.0,
    Ly=2.0,
    Nx=256,
    Ny=256,
    T=5000,
    dt=0.1,
    Du=1e-4,          # Activator diffuses slowly
    Dv=1e-3,          # Inhibitor diffuses quickly (10x faster)
    bc_type='neumann' # Zero-flux boundaries
)

# Create solver
solver = ReactionDiffusionSolver(system)

# Schnakenberg parameters
a = 0.1
b = 0.9

# Calculate the homogeneous steady state
u_star = a + b
v_star = b / (a + b)**2

print(f"\nSystem Parameters:")
print(f"  Domain: [{system.Lx} × {system.Ly}]")
print(f"  Grid: {system.Nx} × {system.Ny}")
print(f"  Diffusion ratio (Dv/Du): {system.Dv/system.Du:.1f}")
print(f"  Steady state: u*={u_star:.4f}, v*={v_star:.4f}")

# Initial conditions: small random perturbations around steady state
u0, v0 = random_perturbation(
    system,
    u_base=u_star,
    v_base=v_star,
    noise_level=0.05
)

solver.set_initial_conditions(u0, v0)

# Solve the system
print("\nSolving Schnakenberg model...")
u_history, v_history, times = solver.solve(
    schnakenberg_reaction,
    a=a,
    b=b,
    save_every=50
)

# Visualizations
print("\n" + "="*70)
print("VISUALIZATION")
print("="*70)

# Plot evolution showing pattern formation
plot_evolution(
    u_history,
    v_history,
    times,
    n_snapshots=6,
    title=f"Turing Pattern Formation (Schnakenberg, a={a}, b={b})",
    cmap='twilight'
)

# Plot final pattern
plot_snapshot(
    u_history[-1],
    v_history[-1],
    times[-1],
    title=f"Turing Pattern - Final State",
    cmap='twilight'
)

# Analyze the pattern wavelength using power spectrum
plot_power_spectrum(
    v_history[-1],
    title="Power Spectrum Analysis - Characteristic Wavelength"
)

# Optional: Create animation
# create_animation(
#     u_history,
#     v_history,
#     times,
#     title=f"Turing Pattern Formation",
#     save_path="turing_pattern_animation.gif",
#     fps=15,
#     cmap='twilight'
# )

print("\n" + "="*70)
print("ANALYSIS:")
print("  - Patterns emerge spontaneously from random initial conditions")
print("  - Pattern wavelength is determined by diffusion ratio and parameters")
print("  - These patterns model animal coat patterns (zebra stripes, leopard spots)")
print("="*70)


# Bonus: Brusselator Turing patterns
print("\n\n" + "="*70)
print("BONUS: BRUSSELATOR MODEL")
print("="*70)

# New system for Brusselator
system2 = ReactionDiffusionSystem(
    Lx=2.0,
    Ly=2.0,
    Nx=256,
    Ny=256,
    T=2000,
    dt=0.05,
    Du=8e-4,
    Dv=4e-3,
    bc_type='periodic'
)

solver2 = ReactionDiffusionSolver(system2)

# Brusselator parameters
A = 4.5
B = 8.0

# Steady state
u_star2 = A
v_star2 = B / A

u0_2, v0_2 = random_perturbation(
    system2,
    u_base=u_star2,
    v_base=v_star2,
    noise_level=0.1
)

solver2.set_initial_conditions(u0_2, v0_2)

print(f"\nBrusselator Parameters: A={A}, B={B}")
print("Solving...")

u_history2, v_history2, times2 = solver2.solve(
    brusselator_reaction,
    A=A,
    B=B,
    save_every=20
)

plot_evolution(
    u_history2,
    v_history2,
    times2,
    n_snapshots=6,
    title=f"Brusselator Turing Patterns (A={A}, B={B})",
    cmap='plasma'
)

print("\n" + "="*70)
print("Simulation complete!")
print("="*70)
