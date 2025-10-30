"""
Fisher's Equation Example

Fisher's equation models the spatial spread of a population with logistic growth:
    ∂u/∂t = D∇²u + r*u*(1 - u/K)

This is a fundamental equation in population dynamics and invasion biology.
Demonstrates traveling wave solutions.

Author: Vivek Singh Yadav
"""

import numpy as np
from reaction_diffusion_solver import (
    ReactionDiffusionSystem,
    ReactionDiffusionSolver,
    fisher_reaction
)
from visualization import (
    plot_evolution,
    plot_snapshot,
    plot_concentration_profile
)

print("="*70)
print("FISHER'S EQUATION - POPULATION INVASION WAVE")
print("="*70)

# System configuration - 1D simulation on a 2D domain
system = ReactionDiffusionSystem(
    Lx=10.0,         # Large domain to see wave propagation
    Ly=1.0,          # Narrow in y-direction (quasi-1D)
    Nx=512,
    Ny=32,
    T=200,
    dt=0.01,
    Du=1e-3,         # Diffusion coefficient
    Dv=0.0,          # Not used (single equation)
    bc_type='neumann'
)

# Fisher equation parameters
r = 1.0          # Growth rate
K = 1.0          # Carrying capacity

print(f"\nParameters:")
print(f"  Growth rate r: {r}")
print(f"  Carrying capacity K: {K}")
print(f"  Diffusion coefficient D: {system.Du}")
print(f"  Theoretical wave speed c = 2*sqrt(r*D) = {2*np.sqrt(r*system.Du):.4f}")

# Create solver
solver = ReactionDiffusionSolver(system)

# Initial condition: localized population on the left side
u0 = np.zeros((system.Ny, system.Nx))
# Population on left 10% of domain
u0[:, :int(0.1 * system.Nx)] = K  # At carrying capacity

# Add small random perturbations
u0 += 0.01 * np.random.random((system.Ny, system.Nx))
u0 = np.clip(u0, 0, K)  # Keep in valid range

solver.set_initial_conditions(u0, v0=None)

print(f"\nInitial condition: Population at carrying capacity on left side")
print("Solving Fisher's equation...")

# Solve
u_history, _, times = solver.solve(
    fisher_reaction,
    r=r,
    K=K,
    save_every=50
)

# Visualizations
print("\n" + "="*70)
print("VISUALIZATION")
print("="*70)

# Plot evolution
plot_evolution(
    u_history,
    [],
    times,
    n_snapshots=6,
    title=f"Fisher's Equation: Population Invasion (r={r}, K={K})",
    cmap='YlOrRd'
)

# Plot 1D profiles at different times
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(12, 6))

# Plot profiles at several time points
time_indices = np.linspace(0, len(u_history)-1, 8, dtype=int)
colors = plt.cm.viridis(np.linspace(0, 1, len(time_indices)))

# Average over y-direction for 1D profile
for idx, (ti, color) in enumerate(zip(time_indices, colors)):
    u_1d = np.mean(u_history[ti], axis=0)
    ax.plot(system.x, u_1d, color=color, linewidth=2,
           label=f't={times[ti]:.1f}')

ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('Population density u', fontsize=12)
ax.set_title("Fisher's Equation: Traveling Wave Solution", fontsize=14, fontweight='bold')
ax.legend(fontsize=9, ncol=2)
ax.grid(True, alpha=0.3)
ax.set_ylim([-0.1, 1.2])

plt.tight_layout()
plt.show()

# Final state
plot_snapshot(
    u_history[-1],
    None,
    times[-1],
    title=f"Fisher's Equation - Final State",
    cmap='YlOrRd'
)

print("\n" + "="*70)
print("ANALYSIS:")
print("  - Population spreads as a traveling wave")
print("  - Wave speed determined by growth rate and diffusion")
print("  - Classic model for biological invasions")
print("  - Related to Allen-Cahn equation in phase transitions")
print("="*70)


# Bonus: Compare different growth rates
print("\n\n" + "="*70)
print("BONUS: Effect of Growth Rate")
print("="*70)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for idx, r_test in enumerate([0.5, 1.0, 2.0]):
    system_test = ReactionDiffusionSystem(
        Lx=10.0, Ly=1.0, Nx=512, Ny=32,
        T=100, dt=0.01, Du=1e-3, bc_type='neumann'
    )

    solver_test = ReactionDiffusionSolver(system_test)
    u0_test = np.zeros((system_test.Ny, system_test.Nx))
    u0_test[:, :int(0.1 * system_test.Nx)] = K
    solver_test.set_initial_conditions(u0_test, v0=None)

    print(f"Solving with r={r_test}...")
    u_hist, _, t_hist = solver_test.solve(fisher_reaction, r=r_test, K=K, save_every=100)

    # Plot final state
    u_1d = np.mean(u_hist[-1], axis=0)
    axes[idx].plot(system_test.x, u_1d, 'b-', linewidth=2)
    axes[idx].set_title(f'r={r_test}, c={2*np.sqrt(r_test*system_test.Du):.3f}', fontsize=12)
    axes[idx].set_xlabel('x')
    axes[idx].set_ylabel('u')
    axes[idx].grid(True, alpha=0.3)
    axes[idx].set_ylim([-0.1, 1.2])

plt.suptitle("Fisher's Equation: Wave Speed vs Growth Rate", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

print("\n" + "="*70)
print("Complete!")
print("="*70)
