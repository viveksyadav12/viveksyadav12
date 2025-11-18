"""
Heart FEM Simulation Example

This script demonstrates cardiac electrophysiology simulation on a heart mesh.
It creates a realistic heart geometry, sets up the FEM solver, and simulates
electrical wave propagation (action potential).

Features demonstrated:
- Heart mesh generation (left ventricle or biventricular)
- Cardiac tissue properties
- Ionic models (Aliev-Panfilov)
- Stimulus protocols
- Anisotropic conduction with fiber orientations
- Visualization of action potential propagation

Author: Vivek Singh Yadav
Date: 2025-11-18
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.tri import Triangulation
import sys

from heart_mesh_generator import (
    HeartMeshGenerator, HeartGeometry,
    create_simple_lv_mesh, create_biventricular_mesh
)
from cardiac_fem_solver import (
    CardiacFEMSolver, AlievPanfilovModel, FitzHughNagumoCardiacModel,
    CardiacTissueProperties, StimulusProtocol
)


def example_lv_propagation():
    """
    Example 1: Action potential propagation in left ventricle

    Simulates electrical wave propagation from apex to base
    """
    print("=" * 60)
    print("Example 1: LV Action Potential Propagation")
    print("=" * 60)

    # Create left ventricle mesh
    print("\n1. Generating left ventricle mesh...")
    heart_mesh = create_simple_lv_mesh(n_radial=10, n_angular=30)

    # Get mesh data
    mesh_data = heart_mesh.export_for_fem()
    nodes = mesh_data['nodes']
    elements = mesh_data['elements']
    fibers = mesh_data['fibers']

    print(f"   Nodes: {len(nodes)}")
    print(f"   Elements: {len(elements)}")

    # Visualize mesh
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot mesh
    ax = axes[0]
    tri = Triangulation(nodes[:, 0], nodes[:, 1], elements[:, :3])
    ax.triplot(tri, 'k-', linewidth=0.3, alpha=0.5)
    ax.plot(nodes[:, 0], nodes[:, 1], 'b.', markersize=2)
    ax.set_aspect('equal')
    ax.set_title('Left Ventricle Mesh')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.grid(True, alpha=0.3)

    # Plot fiber orientations
    ax = axes[1]
    ax.triplot(tri, 'k-', linewidth=0.3, alpha=0.3)

    # Subsample fibers for visualization
    step = max(1, len(nodes) // 50)
    for i in range(0, len(nodes), step):
        x, y = nodes[i]
        fx, fy = fibers[i] if fibers is not None else [1, 0]
        ax.arrow(x, y, fx * 0.3, fy * 0.3, head_width=0.1,
                head_length=0.05, fc='red', ec='red', alpha=0.7)

    ax.set_aspect('equal')
    ax.set_title('Fiber Orientations')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('heart_mesh_lv.png', dpi=150, bbox_inches='tight')
    print("   Saved: heart_mesh_lv.png")
    plt.close()

    # Set up FEM solver
    print("\n2. Setting up cardiac FEM solver...")

    tissue_props = CardiacTissueProperties(
        sigma_l=0.3,  # Longitudinal conductivity
        sigma_t=0.03,  # Transverse conductivity (10x less)
        C_m=1.0
    )

    ionic_model = AlievPanfilovModel(
        k=8.0, a=0.15, epsilon=0.01
    )

    solver = CardiacFEMSolver(
        nodes, elements,
        fiber_field=fibers,
        tissue_properties=tissue_props,
        ionic_model=ionic_model
    )

    # Define stimulus at apex
    print("\n3. Defining stimulus protocol...")

    # Find apex (lowest y-coordinate)
    apex_y = nodes[:, 1].min()
    apex_center = (0.0, apex_y + 0.5)

    stimulus = StimulusProtocol.point_stimulus(
        center=apex_center,
        radius=0.5,
        amplitude=50.0,
        duration=2.0,
        start_time=0.0
    )

    print(f"   Stimulus at: {apex_center}")
    print(f"   Radius: 0.5")
    print(f"   Duration: 2.0 ms")

    # Solve
    print("\n4. Running simulation...")
    T = 50.0  # Total time (ms)
    dt = 0.05  # Time step (ms)

    times, solutions = solver.solve(
        T=T, dt=dt,
        stimulus_func=stimulus,
        save_interval=10
    )

    # Visualize results
    print("\n5. Creating visualizations...")

    # Plot action potential propagation
    n_frames = min(12, len(solutions))
    frame_indices = np.linspace(0, len(solutions) - 1, n_frames, dtype=int)

    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()

    for idx, frame_idx in enumerate(frame_indices):
        ax = axes[idx]
        V = solutions[frame_idx]
        t = times[frame_idx]

        # Plot voltage on mesh
        tcf = ax.tripcolor(tri, V, shading='flat', cmap='hot',
                          vmin=-0.2, vmax=1.0)
        ax.triplot(tri, 'k-', linewidth=0.2, alpha=0.3)
        ax.set_aspect('equal')
        ax.set_title(f't = {t:.1f} ms')
        ax.set_xlabel('x')
        ax.set_ylabel('y')

        if idx == n_frames - 1:
            plt.colorbar(tcf, ax=ax, label='Voltage (normalized)')

    plt.tight_layout()
    plt.savefig('heart_propagation_lv.png', dpi=150, bbox_inches='tight')
    print("   Saved: heart_propagation_lv.png")
    plt.close()

    # Create activation map
    print("\n6. Creating activation map...")

    # Compute activation time (when V crosses threshold)
    activation_time = np.full(len(nodes), np.inf)
    threshold = 0.5

    for t_idx, V in enumerate(solutions):
        activated = (V > threshold) & (activation_time == np.inf)
        activation_time[activated] = times[t_idx]

    # Mask unactivated regions
    activation_time[activation_time == np.inf] = np.nan

    fig, ax = plt.subplots(figsize=(8, 8))
    tcf = ax.tripcolor(tri, activation_time, shading='flat', cmap='jet')
    ax.triplot(tri, 'k-', linewidth=0.2, alpha=0.3)
    ax.set_aspect('equal')
    ax.set_title('Activation Map (ms)')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.colorbar(tcf, ax=ax, label='Activation Time (ms)')

    plt.tight_layout()
    plt.savefig('heart_activation_map.png', dpi=150, bbox_inches='tight')
    print("   Saved: heart_activation_map.png")
    plt.close()

    print("\n" + "=" * 60)
    print("LV simulation complete!")
    print("=" * 60)


def example_biventricular_pacing():
    """
    Example 2: Biventricular pacing simulation

    Simulates electrical pacing of biventricular heart mesh
    """
    print("\n" + "=" * 60)
    print("Example 2: Biventricular Pacing")
    print("=" * 60)

    # Create biventricular mesh
    print("\n1. Generating biventricular mesh...")
    heart_mesh = create_biventricular_mesh()

    mesh_data = heart_mesh.export_for_fem()
    nodes = mesh_data['nodes']
    elements = mesh_data['elements']
    fibers = mesh_data['fibers']

    print(f"   Nodes: {len(nodes)}")
    print(f"   Elements: {len(elements)}")

    # Visualize mesh
    fig, ax = plt.subplots(figsize=(8, 8))
    tri = Triangulation(nodes[:, 0], nodes[:, 1], elements[:, :3])
    ax.triplot(tri, 'k-', linewidth=0.3)
    ax.plot(nodes[:, 0], nodes[:, 1], 'b.', markersize=1)
    ax.set_aspect('equal')
    ax.set_title('Biventricular Heart Mesh')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('heart_mesh_biventricular.png', dpi=150, bbox_inches='tight')
    print("   Saved: heart_mesh_biventricular.png")
    plt.close()

    # Set up solver
    print("\n2. Setting up FEM solver...")

    tissue_props = CardiacTissueProperties()
    ionic_model = AlievPanfilovModel()

    solver = CardiacFEMSolver(
        nodes, elements,
        fiber_field=fibers,
        tissue_properties=tissue_props,
        ionic_model=ionic_model
    )

    # Dual-site pacing stimulus
    print("\n3. Defining biventricular pacing protocol...")

    # Find two pacing sites
    center = np.mean(nodes, axis=0)
    site1 = center + [-1.5, -1.0]
    site2 = center + [1.5, -1.0]

    def dual_stimulus(nodes_array, time):
        stim1 = StimulusProtocol.point_stimulus(
            site1, 0.4, 40.0, 2.0, 0.0
        )(nodes_array, time)
        stim2 = StimulusProtocol.point_stimulus(
            site2, 0.4, 40.0, 2.0, 0.0
        )(nodes_array, time)
        return stim1 + stim2

    print(f"   Pacing site 1: {site1}")
    print(f"   Pacing site 2: {site2}")

    # Solve
    print("\n4. Running biventricular simulation...")
    times, solutions = solver.solve(
        T=60.0, dt=0.05,
        stimulus_func=dual_stimulus,
        save_interval=10
    )

    # Visualize
    print("\n5. Creating visualizations...")

    n_frames = min(12, len(solutions))
    frame_indices = np.linspace(0, len(solutions) - 1, n_frames, dtype=int)

    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()

    for idx, frame_idx in enumerate(frame_indices):
        ax = axes[idx]
        V = solutions[frame_idx]
        t = times[frame_idx]

        tcf = ax.tripcolor(tri, V, shading='flat', cmap='hot',
                          vmin=-0.2, vmax=1.0)
        ax.triplot(tri, 'k-', linewidth=0.2, alpha=0.2)
        ax.set_aspect('equal')
        ax.set_title(f't = {t:.1f} ms')
        ax.set_xlabel('x')
        ax.set_ylabel('y')

        # Mark pacing sites
        ax.plot(*site1, 'g*', markersize=10, label='Pace 1' if idx == 0 else '')
        ax.plot(*site2, 'b*', markersize=10, label='Pace 2' if idx == 0 else '')

        if idx == 0:
            ax.legend()

    plt.tight_layout()
    plt.savefig('heart_biventricular_pacing.png', dpi=150, bbox_inches='tight')
    print("   Saved: heart_biventricular_pacing.png")
    plt.close()

    print("\n" + "=" * 60)
    print("Biventricular simulation complete!")
    print("=" * 60)


def example_reentry_simulation():
    """
    Example 3: Cardiac reentry (spiral wave) simulation

    Demonstrates pathological reentrant wave patterns
    """
    print("\n" + "=" * 60)
    print("Example 3: Reentrant Wave (Spiral Wave)")
    print("=" * 60)

    # Create heart mesh
    print("\n1. Generating heart mesh...")
    heart_mesh = create_simple_lv_mesh(n_radial=12, n_angular=36)

    mesh_data = heart_mesh.export_for_fem()
    nodes = mesh_data['nodes']
    elements = mesh_data['elements']

    print(f"   Nodes: {len(nodes)}")
    print(f"   Elements: {len(elements)}")

    # Set up solver with modified parameters for reentry
    print("\n2. Setting up solver for reentry...")

    tissue_props = CardiacTissueProperties(
        sigma_l=0.25,
        sigma_t=0.025
    )

    # Modified parameters to support reentry
    ionic_model = AlievPanfilovModel(
        k=8.0, a=0.12, epsilon=0.02
    )

    solver = CardiacFEMSolver(
        nodes, elements,
        tissue_properties=tissue_props,
        ionic_model=ionic_model
    )

    # Create initial condition for reentry (S1-S2 protocol)
    print("\n3. Setting up S1-S2 protocol for reentry initiation...")

    V0 = np.zeros(len(nodes))

    # S1: Stimulate entire lower half
    lower_half = nodes[:, 1] < np.mean(nodes[:, 1])
    V0[lower_half] = 1.0

    # S2: Block a region to create conduction block
    block_region = (nodes[:, 0] > 0) & (nodes[:, 1] < 0) & lower_half
    V0[block_region] = 0.0

    solver.set_initial_condition(V0)

    # Solve
    print("\n4. Running reentry simulation...")
    times, solutions = solver.solve(
        T=100.0, dt=0.05,
        stimulus_func=None,
        save_interval=10
    )

    # Visualize
    print("\n5. Creating visualizations...")

    tri = Triangulation(nodes[:, 0], nodes[:, 1], elements[:, :3])

    n_frames = min(16, len(solutions))
    frame_indices = np.linspace(0, len(solutions) - 1, n_frames, dtype=int)

    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    axes = axes.flatten()

    for idx, frame_idx in enumerate(frame_indices):
        ax = axes[idx]
        V = solutions[frame_idx]
        t = times[frame_idx]

        tcf = ax.tripcolor(tri, V, shading='flat', cmap='hot',
                          vmin=-0.2, vmax=1.0)
        ax.triplot(tri, 'k-', linewidth=0.2, alpha=0.2)
        ax.set_aspect('equal')
        ax.set_title(f't = {t:.1f} ms')
        ax.set_xlabel('x')
        ax.set_ylabel('y')

    plt.tight_layout()
    plt.savefig('heart_reentry_spiral.png', dpi=150, bbox_inches='tight')
    print("   Saved: heart_reentry_spiral.png")
    plt.close()

    print("\n" + "=" * 60)
    print("Reentry simulation complete!")
    print("=" * 60)


def run_all_examples():
    """Run all heart FEM examples"""
    print("\n" + "=" * 60)
    print("HEART FEM SIMULATION EXAMPLES")
    print("=" * 60)
    print("\nThis will run three cardiac electrophysiology examples:")
    print("  1. Left ventricle action potential propagation")
    print("  2. Biventricular pacing")
    print("  3. Reentrant wave (spiral wave)")
    print("\n" + "=" * 60)

    try:
        # Example 1: LV propagation
        example_lv_propagation()

        # Example 2: Biventricular pacing
        example_biventricular_pacing()

        # Example 3: Reentry
        example_reentry_simulation()

        print("\n" + "=" * 60)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\nGenerated files:")
        print("  - heart_mesh_lv.png")
        print("  - heart_propagation_lv.png")
        print("  - heart_activation_map.png")
        print("  - heart_mesh_biventricular.png")
        print("  - heart_biventricular_pacing.png")
        print("  - heart_reentry_spiral.png")
        print("=" * 60)

    except Exception as e:
        print(f"\nError during simulation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')

    if len(sys.argv) > 1:
        if sys.argv[1] == "lv":
            example_lv_propagation()
        elif sys.argv[1] == "biv":
            example_biventricular_pacing()
        elif sys.argv[1] == "reentry":
            example_reentry_simulation()
        else:
            print("Usage: python example_heart_fem.py [lv|biv|reentry]")
            print("Or run without arguments to run all examples")
    else:
        run_all_examples()
