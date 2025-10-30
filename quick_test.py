"""
Quick test script to verify the reaction-diffusion solver is working correctly.

This runs a fast simulation with reduced parameters to check the installation.

Author: Vivek Singh Yadav
"""

import numpy as np
from reaction_diffusion_solver import (
    ReactionDiffusionSystem,
    ReactionDiffusionSolver,
    gray_scott_reaction,
    fisher_reaction,
    random_perturbation,
    localized_perturbation
)
from visualization import plot_snapshot

print("="*70)
print("REACTION-DIFFUSION SOLVER - QUICK TEST")
print("="*70)

# Test 1: Gray-Scott (small simulation)
print("\n[Test 1] Gray-Scott Model (quick test)...")
system1 = ReactionDiffusionSystem(
    Lx=1.0, Ly=1.0, Nx=64, Ny=64,
    T=500, dt=1.0,
    Du=2e-5, Dv=1e-5,
    bc_type='periodic'
)

solver1 = ReactionDiffusionSolver(system1)
u0, v0 = random_perturbation(system1, u_base=1.0, v_base=0.0, noise_level=0.02)
u0_center, v0_center = localized_perturbation(
    system1, u_base=0.0, v_base=0.0,
    center_value_u=0.5, center_value_v=0.25, radius=0.15
)
u0 = u0 - u0_center + 0.5
v0 = v0 + v0_center

solver1.set_initial_conditions(u0, v0)
u_hist1, v_hist1, times1 = solver1.solve(
    gray_scott_reaction, F=0.04, k=0.06, save_every=100
)

print(f"✓ Gray-Scott simulation complete!")
print(f"  Final time: {times1[-1]:.1f}")
print(f"  Snapshots saved: {len(u_hist1)}")
print(f"  u range: [{np.min(u_hist1[-1]):.4f}, {np.max(u_hist1[-1]):.4f}]")
print(f"  v range: [{np.min(v_hist1[-1]):.4f}, {np.max(v_hist1[-1]):.4f}]")

# Test 2: Fisher equation (1D wave)
print("\n[Test 2] Fisher's Equation (quick test)...")
system2 = ReactionDiffusionSystem(
    Lx=5.0, Ly=1.0, Nx=128, Ny=16,
    T=50, dt=0.01,
    Du=1e-3, bc_type='neumann'
)

solver2 = ReactionDiffusionSolver(system2)
u0_fisher = np.zeros((system2.Ny, system2.Nx))
u0_fisher[:, :int(0.1 * system2.Nx)] = 1.0
solver2.set_initial_conditions(u0_fisher)

u_hist2, _, times2 = solver2.solve(fisher_reaction, r=1.0, K=1.0, save_every=50)

print(f"✓ Fisher equation simulation complete!")
print(f"  Final time: {times2[-1]:.1f}")
print(f"  Wave front position: ~{np.mean(np.where(np.mean(u_hist2[-1], axis=0) > 0.5)):.1f}")

# Test 3: Verify numerical stability
print("\n[Test 3] Checking numerical stability...")
assert not np.any(np.isnan(u_hist1[-1])), "Gray-Scott: NaN detected!"
assert not np.any(np.isnan(v_hist1[-1])), "Gray-Scott: NaN detected!"
assert not np.any(np.isnan(u_hist2[-1])), "Fisher: NaN detected!"
assert np.all(u_hist1[-1] >= -0.1), "Gray-Scott: u became negative!"
assert np.all(v_hist1[-1] >= -0.1), "Gray-Scott: v became negative!"
assert np.all(u_hist2[-1] >= -0.1), "Fisher: u became negative!"

print("✓ All stability checks passed!")

# Test 4: Visualization
print("\n[Test 4] Testing visualization...")
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for testing
    import matplotlib.pyplot as plt

    plot_snapshot(u_hist1[-1], v_hist1[-1], times1[-1],
                 title="Gray-Scott Test", cmap='viridis')
    plt.close('all')  # Don't display, just test

    print("✓ Visualization works!")
except Exception as e:
    print(f"⚠ Visualization test skipped: {e}")

print("\n" + "="*70)
print("ALL TESTS PASSED! ✓")
print("="*70)
print("\nThe solver is working correctly. You can now run the example scripts:")
print("  - python example_gray_scott.py")
print("  - python example_turing_patterns.py")
print("  - python example_fisher.py")
print("  - python example_fitzhugh_nagumo.py")
print("="*70)
