"""
FitzHugh-Nagumo Model Example

The FitzHugh-Nagumo model is a simplified model of neuronal excitation and inhibition.
It demonstrates:
- Excitable media
- Spiral waves
- Pulse propagation

This model is widely used in cardiac electrophysiology and neuroscience.

Author: Vivek Singh Yadav
"""

import numpy as np
from reaction_diffusion_solver import (
    ReactionDiffusionSystem,
    ReactionDiffusionSolver,
    fitzhugh_nagumo_reaction,
    localized_perturbation
)
from visualization import (
    plot_evolution,
    plot_snapshot,
    plot_phase_space,
    create_animation
)

print("="*70)
print("FITZHUGH-NAGUMO MODEL - EXCITABLE MEDIA")
print("="*70)

# System configuration
system = ReactionDiffusionSystem(
    Lx=3.0,
    Ly=3.0,
    Nx=256,
    Ny=256,
    T=300,
    dt=0.05,
    Du=1e-3,          # Diffusion for activator (u)
    Dv=0.0,           # Inhibitor (v) doesn't diffuse
    bc_type='neumann'
)

# FitzHugh-Nagumo parameters
a = 0.0
b = 0.5
tau = 10.0

print(f"\nParameters:")
print(f"  a = {a}")
print(f"  b = {b}")
print(f"  tau = {tau} (recovery timescale)")
print(f"  Du = {system.Du} (u diffuses)")
print(f"  Dv = {system.Dv} (v doesn't diffuse)")

# Create solver
solver = ReactionDiffusionSolver(system)

# Initial conditions: small perturbation to trigger excitation
# Rest state: u ≈ 0, v ≈ b
u0 = np.zeros((system.Ny, system.Nx))
v0 = np.ones((system.Ny, system.Nx)) * b

# Add localized stimulus (super-threshold perturbation)
u_stim, v_stim = localized_perturbation(
    system,
    u_base=0.0,
    v_base=0.0,
    center_value_u=1.0,
    center_value_v=0.0,
    radius=0.3
)

u0 += u_stim

# Add a second stimulus at different location to create interesting dynamics
center_x2 = system.Lx * 0.7
center_y2 = system.Ly * 0.7
r_squared2 = (system.X - center_x2)**2 + (system.Y - center_y2)**2
mask2 = r_squared2 < 0.2**2
u0[mask2] = 0.8

solver.set_initial_conditions(u0, v0)

print("\nInitial condition: Localized stimuli to trigger excitation waves")
print("Solving FitzHugh-Nagumo model...")

# Solve
u_history, v_history, times = solver.solve(
    fitzhugh_nagumo_reaction,
    a=a,
    b=b,
    tau=tau,
    save_every=20
)

# Visualizations
print("\n" + "="*70)
print("VISUALIZATION")
print("="*70)

# Plot evolution
plot_evolution(
    u_history,
    v_history,
    times,
    n_snapshots=6,
    title=f"FitzHugh-Nagumo: Excitation Waves (a={a}, b={b}, τ={tau})",
    cmap='seismic'
)

# Plot final state
plot_snapshot(
    u_history[-1],
    v_history[-1],
    times[-1],
    title="FitzHugh-Nagumo - Excitable Media",
    cmap='seismic'
)

# Phase space analysis
plot_phase_space(
    u_history,
    v_history,
    sample_points=2000,
    title="FitzHugh-Nagumo Phase Space (u vs v)"
)

# Optional: Create animation
# create_animation(
#     u_history,
#     v_history,
#     times,
#     title="FitzHugh-Nagumo Excitation Waves",
#     save_path="fitzhugh_nagumo_animation.gif",
#     fps=15,
#     cmap='seismic'
# )

print("\n" + "="*70)
print("ANALYSIS:")
print("  - u: Fast activator (membrane potential)")
print("  - v: Slow inhibitor (recovery variable)")
print("  - Exhibits threshold behavior: small perturbations decay,")
print("    large perturbations trigger propagating waves")
print("  - Models action potential propagation in nerves/heart")
print("="*70)


# Bonus: Spiral wave formation
print("\n\n" + "="*70)
print("BONUS: SPIRAL WAVE FORMATION")
print("="*70)

system_spiral = ReactionDiffusionSystem(
    Lx=5.0,
    Ly=5.0,
    Nx=256,
    Ny=256,
    T=500,
    dt=0.05,
    Du=1e-3,
    Dv=0.0,
    bc_type='neumann'
)

solver_spiral = ReactionDiffusionSolver(system_spiral)

# Create initial condition for spiral: broken wave front
u0_spiral = np.zeros((system_spiral.Ny, system_spiral.Nx))
v0_spiral = np.ones((system_spiral.Ny, system_spiral.Nx)) * b

# Create a broken wave front
midx = system_spiral.Nx // 2
midy = system_spiral.Ny // 2

# Vertical front on left half
u0_spiral[:, :midx] = 1.0
v0_spiral[:, :midx] = 0.0

# Remove bottom half to create break
u0_spiral[midy:, :midx//2] = 0.0
v0_spiral[midy:, :midx//2] = b

solver_spiral.set_initial_conditions(u0_spiral, v0_spiral)

print("Creating spiral wave from broken wave front...")
u_hist_spiral, v_hist_spiral, times_spiral = solver_spiral.solve(
    fitzhugh_nagumo_reaction,
    a=a,
    b=b,
    tau=tau,
    save_every=30
)

plot_evolution(
    u_hist_spiral,
    v_hist_spiral,
    times_spiral,
    n_snapshots=6,
    title="FitzHugh-Nagumo: Spiral Wave Formation",
    cmap='seismic'
)

print("\n" + "="*70)
print("Spiral waves are important in:")
print("  - Cardiac arrhythmias (may lead to fibrillation)")
print("  - Chemical oscillators (Belousov-Zhabotinsky reaction)")
print("  - Cortical spreading depression")
print("="*70)

print("\nSimulation complete!")
