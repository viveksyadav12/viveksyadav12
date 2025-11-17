"""
Example: PDE Solving on Curved Meshes

Demonstrates solving reaction-diffusion PDEs on unstructured and curved meshes:
- Heat equation on circular domain
- Reaction-diffusion on Delaunay mesh
- Gray-Scott patterns on unstructured mesh
- Adaptive mesh refinement

Author: Vivek Singh Yadav
Date: 2025-11-17
"""

import numpy as np
import matplotlib.pyplot as plt
from curved_mesh_generator import CurvedMeshGenerator, ElementType
from mesh_visualization import MeshVisualizer
from unstructured_pde_solver import (
    UnstructuredPDESystem, UnstructuredPDESolver,
    interpolate_solution_to_mesh
)


def example_heat_equation_circular():
    """Solve heat equation on circular domain"""
    print("=" * 70)
    print("Example 1: Heat Equation on Circular Domain")
    print("=" * 70)

    # Generate circular mesh
    mesh_gen = CurvedMeshGenerator()
    mesh_gen.generate_circular_domain(
        center=(0.0, 0.0),
        radius=1.0,
        n_radial=10,
        n_angular=24
    )

    print(f"Mesh: {len(mesh_gen.nodes)} nodes, {len(mesh_gen.elements)} elements")

    # Setup PDE system
    system = UnstructuredPDESystem(
        mesh=mesh_gen,
        T=0.5,
        dt=0.001,
        Du=0.1,
        boundary_condition="dirichlet",
        boundary_value_u=0.0
    )

    # Initial condition: Gaussian in center
    u0 = np.zeros(len(mesh_gen.nodes))
    for nid, node in mesh_gen.nodes.items():
        r = np.sqrt(node.x**2 + node.y**2)
        u0[nid] = np.exp(-20 * r**2)

    # Solve
    solver = UnstructuredPDESolver(system)
    u_history, _ = solver.solve(u0, store_interval=50)

    print(f"Solution computed at {u_history.shape[0]} time steps")

    # Visualize solution evolution
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    time_indices = np.linspace(0, u_history.shape[0]-1, 6, dtype=int)

    for idx, ax in enumerate(axes):
        t_idx = time_indices[idx]
        t = system.t[t_idx * 50]  # Adjust for store_interval

        # Create triangulation for plotting
        triangles = []
        for elem in mesh_gen.elements.values():
            if elem.element_type == ElementType.TRIANGLE_LINEAR:
                triangles.append(elem.node_ids[:3])

        triangles = np.array(triangles)

        # Get node coordinates
        x_coords = np.array([node.x for node in mesh_gen.nodes.values()])
        y_coords = np.array([node.y for node in mesh_gen.nodes.values()])

        # Plot solution
        tcf = ax.tricontourf(x_coords, y_coords, triangles, u_history[t_idx, :],
                            levels=15, cmap='hot')
        ax.triplot(x_coords, y_coords, triangles, 'k-', linewidth=0.3, alpha=0.3)
        ax.set_aspect('equal')
        ax.set_title(f't = {t:.3f}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        plt.colorbar(tcf, ax=ax)

    plt.tight_layout()
    plt.savefig('pde_heat_circular.png', dpi=300, bbox_inches='tight')
    print("Saved: pde_heat_circular.png")
    plt.close()


def example_reaction_diffusion_unstructured():
    """Solve Fisher's equation on unstructured mesh"""
    print("\n" + "=" * 70)
    print("Example 2: Fisher's Equation on Unstructured Mesh")
    print("=" * 70)

    # Generate unstructured mesh using Delaunay
    np.random.seed(42)
    n_interior = 100
    points_interior = np.random.rand(n_interior, 2) * 2.0

    # Boundary points
    n_boundary = 40
    boundary_points = []
    for i in range(n_boundary):
        theta = 2 * np.pi * i / n_boundary
        x = 1.0 + 0.9 * np.cos(theta)
        y = 1.0 + 0.9 * np.sin(theta)
        boundary_points.append([x, y])

    boundary_points = np.array(boundary_points)
    all_points = np.vstack([points_interior, boundary_points])

    mesh_gen = CurvedMeshGenerator()
    mesh_gen.bowyer_watson_triangulation(all_points)

    # Mark boundary nodes (circular)
    for nid, node in mesh_gen.nodes.items():
        r = np.sqrt((node.x - 1.0)**2 + (node.y - 1.0)**2)
        if r > 0.85:
            node.on_boundary = True

    print(f"Mesh: {len(mesh_gen.nodes)} nodes, {len(mesh_gen.elements)} elements")

    # Smooth mesh for better quality
    mesh_gen.smooth_mesh_laplacian(iterations=5)

    stats = mesh_gen.get_mesh_statistics()
    print(f"Mesh quality: Min={stats['quality_min']:.3f}, Mean={stats['quality_mean']:.3f}")

    # Setup PDE: Fisher's equation du/dt = D*Δu + r*u*(1 - u/K)
    system = UnstructuredPDESystem(
        mesh=mesh_gen,
        T=5.0,
        dt=0.01,
        Du=0.01,
        boundary_condition="neumann"
    )

    # Reaction term: Fisher's equation
    r = 1.0
    K = 1.0

    def reaction_fisher(u, v, x, y):
        return r * u * (1 - u / K)

    # Initial condition: localized perturbation
    u0 = np.zeros(len(mesh_gen.nodes))
    for nid, node in mesh_gen.nodes.items():
        dist = np.sqrt((node.x - 1.0)**2 + (node.y - 1.0)**2)
        if dist < 0.3:
            u0[nid] = 0.5 * (1 + np.cos(np.pi * dist / 0.3))

    # Solve
    solver = UnstructuredPDESolver(system)
    u_history, _ = solver.solve(u0, reaction_u=reaction_fisher, store_interval=50)

    print(f"Solution computed")

    # Visualize
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    time_indices = np.linspace(0, u_history.shape[0]-1, 6, dtype=int)

    # Create triangulation
    triangles = []
    for elem in mesh_gen.elements.values():
        if elem.element_type == ElementType.TRIANGLE_LINEAR:
            triangles.append(elem.node_ids[:3])
    triangles = np.array(triangles)

    x_coords = np.array([node.x for node in mesh_gen.nodes.values()])
    y_coords = np.array([node.y for node in mesh_gen.nodes.values()])

    for idx, ax in enumerate(axes):
        t_idx = time_indices[idx]
        t = system.t[t_idx * 50]

        tcf = ax.tricontourf(x_coords, y_coords, triangles, u_history[t_idx, :],
                            levels=15, cmap='viridis')
        ax.triplot(x_coords, y_coords, triangles, 'k-', linewidth=0.2, alpha=0.2)
        ax.set_aspect('equal')
        ax.set_title(f"t = {t:.2f}")
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        plt.colorbar(tcf, ax=ax, label='u')

    plt.suptitle("Fisher's Equation on Unstructured Mesh", fontsize=14)
    plt.tight_layout()
    plt.savefig('pde_fisher_unstructured.png', dpi=300, bbox_inches='tight')
    print("Saved: pde_fisher_unstructured.png")
    plt.close()


def example_gray_scott_unstructured():
    """Solve Gray-Scott model on unstructured mesh"""
    print("\n" + "=" * 70)
    print("Example 3: Gray-Scott Pattern Formation on Unstructured Mesh")
    print("=" * 70)

    # Generate rectangular mesh
    mesh_gen = CurvedMeshGenerator()
    mesh_gen.generate_rectangular_domain(
        x_min=0.0, x_max=2.5,
        y_min=0.0, y_max=2.5,
        nx=40, ny=40,
        element_type=ElementType.TRIANGLE_LINEAR
    )

    print(f"Mesh: {len(mesh_gen.nodes)} nodes, {len(mesh_gen.elements)} elements")

    # Gray-Scott parameters (spot pattern)
    Du = 0.16
    Dv = 0.08
    F = 0.060
    k = 0.062

    system = UnstructuredPDESystem(
        mesh=mesh_gen,
        T=10000,  # Long time for pattern formation
        dt=1.0,
        Du=Du,
        Dv=Dv,
        boundary_condition="neumann"
    )

    # Reaction terms
    def reaction_u(u, v, x, y):
        return -u * v**2 + F * (1 - u)

    def reaction_v(u, v, x, y):
        return u * v**2 - (F + k) * v

    # Initial conditions
    u0 = np.ones(len(mesh_gen.nodes))
    v0 = np.zeros(len(mesh_gen.nodes))

    # Add random perturbations
    np.random.seed(42)
    for nid, node in mesh_gen.nodes.items():
        # Central square perturbation
        if 1.0 < node.x < 1.5 and 1.0 < node.y < 1.5:
            u0[nid] = 0.5 + 0.01 * np.random.randn()
            v0[nid] = 0.25 + 0.01 * np.random.randn()

    # Solve
    print("Solving Gray-Scott (this may take a minute)...")
    solver = UnstructuredPDESolver(system)
    u_history, v_history = solver.solve(
        u0, v0,
        reaction_u=reaction_u,
        reaction_v=reaction_v,
        store_interval=1000
    )

    # Visualize patterns
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Create triangulation
    triangles = []
    for elem in mesh_gen.elements.values():
        if elem.element_type == ElementType.TRIANGLE_LINEAR:
            triangles.append(elem.node_ids[:3])
    triangles = np.array(triangles)

    x_coords = np.array([node.x for node in mesh_gen.nodes.values()])
    y_coords = np.array([node.y for node in mesh_gen.nodes.values()])

    time_indices = np.linspace(0, u_history.shape[0]-1, 6, dtype=int)

    for idx in range(6):
        ax = axes[idx // 3, idx % 3]
        t_idx = time_indices[idx]
        t = system.t[t_idx * 1000]

        # Plot v component (shows patterns better)
        tcf = ax.tricontourf(x_coords, y_coords, triangles, v_history[t_idx, :],
                            levels=15, cmap='RdYlBu_r')
        ax.set_aspect('equal')
        ax.set_title(f't = {t:.0f}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        plt.colorbar(tcf, ax=ax, label='v')

    plt.suptitle("Gray-Scott Pattern Formation (v component)", fontsize=14)
    plt.tight_layout()
    plt.savefig('pde_grayscott_unstructured.png', dpi=300, bbox_inches='tight')
    print("Saved: pde_grayscott_unstructured.png")
    plt.close()


def example_adaptive_refinement():
    """Demonstrate adaptive mesh refinement"""
    print("\n" + "=" * 70)
    print("Example 4: Adaptive Mesh Refinement")
    print("=" * 70)

    # Start with coarse mesh
    mesh_coarse = CurvedMeshGenerator()
    mesh_coarse.generate_rectangular_domain(
        x_min=0.0, x_max=1.0,
        y_min=0.0, y_max=1.0,
        nx=8, ny=8,
        element_type=ElementType.TRIANGLE_LINEAR
    )

    print(f"Coarse mesh: {len(mesh_coarse.nodes)} nodes, {len(mesh_coarse.elements)} elements")

    # Solve heat equation with localized source
    system_coarse = UnstructuredPDESystem(
        mesh=mesh_coarse,
        T=0.1,
        dt=0.001,
        Du=0.01,
        boundary_condition="dirichlet",
        boundary_value_u=0.0
    )

    # Initial condition: steep gradient at center
    u0_coarse = np.zeros(len(mesh_coarse.nodes))
    for nid, node in mesh_coarse.nodes.items():
        r = np.sqrt((node.x - 0.5)**2 + (node.y - 0.5)**2)
        u0_coarse[nid] = np.exp(-100 * r**2)

    solver_coarse = UnstructuredPDESolver(system_coarse)
    u_history_coarse, _ = solver_coarse.solve(u0_coarse, store_interval=100)
    u_final_coarse = u_history_coarse[-1, :]

    print("Coarse solution computed")

    # Adaptive refinement
    mesh_refined = solver_coarse.adaptive_refine(u_final_coarse, threshold=0.3)

    print(f"Refined mesh: {len(mesh_refined.nodes)} nodes, {len(mesh_refined.elements)} elements")

    # Interpolate solution to refined mesh
    u0_refined = interpolate_solution_to_mesh(u_final_coarse, mesh_coarse, mesh_refined)

    # Solve on refined mesh
    system_refined = UnstructuredPDESystem(
        mesh=mesh_refined,
        T=0.1,
        dt=0.001,
        Du=0.01,
        boundary_condition="dirichlet",
        boundary_value_u=0.0
    )

    solver_refined = UnstructuredPDESolver(system_refined)
    u_history_refined, _ = solver_refined.solve(u0_refined, store_interval=100)
    u_final_refined = u_history_refined[-1, :]

    print("Refined solution computed")

    # Visualize comparison
    fig = plt.figure(figsize=(16, 6))

    # Coarse mesh
    ax1 = fig.add_subplot(131)
    viz_coarse = MeshVisualizer(mesh_coarse)
    viz_coarse.plot_mesh(ax=ax1, show_nodes=True, color_by_quality=True)
    ax1.set_title('Coarse Mesh')

    # Refined mesh
    ax2 = fig.add_subplot(132)
    viz_refined = MeshVisualizer(mesh_refined)
    viz_refined.plot_mesh(ax=ax2, show_nodes=True, color_by_quality=True)
    ax2.set_title('Refined Mesh')

    # Solution comparison
    ax3 = fig.add_subplot(133)

    # Create triangulations
    tri_coarse = []
    for elem in mesh_coarse.elements.values():
        if elem.element_type == ElementType.TRIANGLE_LINEAR:
            tri_coarse.append(elem.node_ids[:3])
    tri_coarse = np.array(tri_coarse)

    x_coarse = np.array([node.x for node in mesh_coarse.nodes.values()])
    y_coarse = np.array([node.y for node in mesh_coarse.nodes.values()])

    tcf = ax3.tricontourf(x_coarse, y_coarse, tri_coarse, u_final_coarse,
                         levels=15, cmap='hot')
    ax3.triplot(x_coarse, y_coarse, tri_coarse, 'k-', linewidth=0.3, alpha=0.3)
    ax3.set_aspect('equal')
    ax3.set_title('Solution on Coarse Mesh')
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    plt.colorbar(tcf, ax=ax3)

    plt.tight_layout()
    plt.savefig('pde_adaptive_refinement.png', dpi=300, bbox_inches='tight')
    print("Saved: pde_adaptive_refinement.png")
    plt.close()


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("PDE SOLVING ON CURVED/UNSTRUCTURED MESHES")
    print("=" * 70)

    example_heat_equation_circular()
    example_reaction_diffusion_unstructured()
    example_gray_scott_unstructured()
    example_adaptive_refinement()

    print("\n" + "=" * 70)
    print("ALL PDE EXAMPLES COMPLETED!")
    print("=" * 70)
    print("\nGenerated files:")
    print("  - pde_heat_circular.png")
    print("  - pde_fisher_unstructured.png")
    print("  - pde_grayscott_unstructured.png")
    print("  - pde_adaptive_refinement.png")
