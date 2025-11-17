"""
Example: Basic Curved Mesh Generation

Demonstrates fundamental mesh generation capabilities:
- Structured meshes on rectangular and circular domains
- Delaunay triangulation
- Mesh quality analysis
- Mesh visualization

Author: Vivek Singh Yadav
Date: 2025-11-17
"""

import numpy as np
import matplotlib.pyplot as plt
from curved_mesh_generator import CurvedMeshGenerator, ElementType
from mesh_visualization import MeshVisualizer, plot_refinement_progression


def example_rectangular_mesh():
    """Generate structured triangular mesh on rectangular domain"""
    print("=" * 70)
    print("Example 1: Rectangular Domain with Triangular Mesh")
    print("=" * 70)

    mesh_gen = CurvedMeshGenerator()
    mesh_gen.generate_rectangular_domain(
        x_min=0.0, x_max=2.0,
        y_min=0.0, y_max=1.0,
        nx=11, ny=6,
        element_type=ElementType.TRIANGLE_LINEAR
    )

    # Get statistics
    stats = mesh_gen.get_mesh_statistics()
    print(f"\nMesh Statistics:")
    print(f"  Nodes: {stats['num_nodes']}")
    print(f"  Elements: {stats['num_elements']}")
    print(f"  Boundary nodes: {stats['num_boundary_nodes']}")
    print(f"  Quality - Min: {stats['quality_min']:.4f}, "
          f"Mean: {stats['quality_mean']:.4f}, Max: {stats['quality_max']:.4f}")

    # Visualize
    viz = MeshVisualizer(mesh_gen)
    fig = viz.plot_mesh(show_nodes=True, show_boundary=True,
                       color_by_quality=True, figsize=(12, 6))
    plt.savefig('mesh_rectangular.png', dpi=300, bbox_inches='tight')
    print("\nSaved: mesh_rectangular.png")
    plt.close()

    return mesh_gen


def example_circular_mesh():
    """Generate structured mesh on circular domain"""
    print("\n" + "=" * 70)
    print("Example 2: Circular Domain with Radial Mesh")
    print("=" * 70)

    mesh_gen = CurvedMeshGenerator()
    mesh_gen.generate_circular_domain(
        center=(0.0, 0.0),
        radius=1.0,
        n_radial=5,
        n_angular=16,
        element_type=ElementType.TRIANGLE_LINEAR
    )

    stats = mesh_gen.get_mesh_statistics()
    print(f"\nMesh Statistics:")
    print(f"  Nodes: {stats['num_nodes']}")
    print(f"  Elements: {stats['num_elements']}")
    print(f"  Quality - Min: {stats['quality_min']:.4f}, "
          f"Mean: {stats['quality_mean']:.4f}, Max: {stats['quality_max']:.4f}")

    viz = MeshVisualizer(mesh_gen)
    fig = viz.plot_mesh(show_nodes=True, show_boundary=True,
                       color_by_quality=True, figsize=(10, 10))
    plt.savefig('mesh_circular.png', dpi=300, bbox_inches='tight')
    print("\nSaved: mesh_circular.png")
    plt.close()

    return mesh_gen


def example_delaunay_triangulation():
    """Generate unstructured mesh using Delaunay triangulation"""
    print("\n" + "=" * 70)
    print("Example 3: Delaunay Triangulation (Unstructured Mesh)")
    print("=" * 70)

    # Generate random points in unit square
    np.random.seed(42)
    n_points = 50
    points = np.random.rand(n_points, 2)

    # Add boundary points for better quality
    n_boundary = 20
    boundary_x = np.linspace(0, 1, n_boundary)
    boundary_points = np.vstack([
        np.column_stack([boundary_x, np.zeros(n_boundary)]),  # Bottom
        np.column_stack([boundary_x, np.ones(n_boundary)]),   # Top
        np.column_stack([np.zeros(n_boundary), np.linspace(0, 1, n_boundary)]),  # Left
        np.column_stack([np.ones(n_boundary), np.linspace(0, 1, n_boundary)])    # Right
    ])

    all_points = np.vstack([points, boundary_points])

    mesh_gen = CurvedMeshGenerator()
    mesh_gen.bowyer_watson_triangulation(all_points)

    # Mark boundary nodes
    for nid, node in mesh_gen.nodes.items():
        if (abs(node.x) < 1e-6 or abs(node.x - 1) < 1e-6 or
            abs(node.y) < 1e-6 or abs(node.y - 1) < 1e-6):
            node.on_boundary = True

    stats = mesh_gen.get_mesh_statistics()
    print(f"\nMesh Statistics:")
    print(f"  Nodes: {stats['num_nodes']}")
    print(f"  Elements: {stats['num_elements']}")
    print(f"  Quality - Min: {stats['quality_min']:.4f}, "
          f"Mean: {stats['quality_mean']:.4f}, Max: {stats['quality_max']:.4f}")

    viz = MeshVisualizer(mesh_gen)
    fig = viz.plot_mesh(show_nodes=True, show_boundary=True,
                       color_by_quality=True, figsize=(10, 10))
    plt.savefig('mesh_delaunay.png', dpi=300, bbox_inches='tight')
    print("\nSaved: mesh_delaunay.png")
    plt.close()

    # Quality histogram
    fig_hist = viz.plot_quality_histogram()
    plt.savefig('mesh_delaunay_quality.png', dpi=300, bbox_inches='tight')
    print("Saved: mesh_delaunay_quality.png")
    plt.close()

    return mesh_gen


def example_mesh_smoothing():
    """Demonstrate mesh smoothing"""
    print("\n" + "=" * 70)
    print("Example 4: Mesh Smoothing with Laplacian Smoothing")
    print("=" * 70)

    # Start with Delaunay mesh
    np.random.seed(42)
    n_points = 40
    points = np.random.rand(n_points, 2) * 2 - 0.5  # Random points in [-0.5, 1.5]

    mesh_gen = CurvedMeshGenerator()
    mesh_gen.bowyer_watson_triangulation(points)

    # Mark boundary
    x_coords = np.array([node.x for node in mesh_gen.nodes.values()])
    y_coords = np.array([node.y for node in mesh_gen.nodes.values()])
    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()

    for nid, node in mesh_gen.nodes.items():
        if (abs(node.x - x_min) < 1e-6 or abs(node.x - x_max) < 1e-6 or
            abs(node.y - y_min) < 1e-6 or abs(node.y - y_max) < 1e-6):
            node.on_boundary = True

    stats_before = mesh_gen.get_mesh_statistics()
    print(f"\nBefore smoothing:")
    print(f"  Quality - Min: {stats_before['quality_min']:.4f}, "
          f"Mean: {stats_before['quality_mean']:.4f}")

    # Create copy for smoothing
    import copy
    mesh_smoothed = copy.deepcopy(mesh_gen)

    # Apply smoothing
    mesh_smoothed.smooth_mesh_laplacian(iterations=10, relaxation=0.5)

    stats_after = mesh_smoothed.get_mesh_statistics()
    print(f"\nAfter smoothing:")
    print(f"  Quality - Min: {stats_after['quality_min']:.4f}, "
          f"Mean: {stats_after['quality_mean']:.4f}")
    print(f"  Improvement: {100*(stats_after['quality_mean']/stats_before['quality_mean']-1):.1f}%")

    # Compare
    viz_before = MeshVisualizer(mesh_gen)
    viz_after = MeshVisualizer(mesh_smoothed)

    fig = viz_before.plot_mesh_comparison(mesh_smoothed, figsize=(16, 7))
    fig.suptitle('Mesh Smoothing Comparison', fontsize=14, y=1.02)
    plt.savefig('mesh_smoothing_comparison.png', dpi=300, bbox_inches='tight')
    print("\nSaved: mesh_smoothing_comparison.png")
    plt.close()

    return mesh_gen, mesh_smoothed


def example_mesh_refinement():
    """Demonstrate uniform mesh refinement"""
    print("\n" + "=" * 70)
    print("Example 5: Uniform Mesh Refinement")
    print("=" * 70)

    # Start with coarse rectangular mesh
    mesh_level0 = CurvedMeshGenerator()
    mesh_level0.generate_rectangular_domain(
        x_min=0.0, x_max=1.0,
        y_min=0.0, y_max=1.0,
        nx=5, ny=5,
        element_type=ElementType.TRIANGLE_LINEAR
    )

    # Refine once
    import copy
    mesh_level1 = copy.deepcopy(mesh_level0)
    mesh_level1.refine_mesh_uniform()

    # Refine again
    mesh_level2 = copy.deepcopy(mesh_level1)
    mesh_level2.refine_mesh_uniform()

    # Statistics
    meshes = [mesh_level0, mesh_level1, mesh_level2]
    for i, mesh in enumerate(meshes):
        stats = mesh.get_mesh_statistics()
        print(f"\nLevel {i}:")
        print(f"  Nodes: {stats['num_nodes']}, Elements: {stats['num_elements']}")
        print(f"  Quality - Mean: {stats['quality_mean']:.4f}")

    # Visualize progression
    titles = [f"Level {i}" for i in range(3)]
    fig = plot_refinement_progression(meshes, titles=titles)
    plt.savefig('mesh_refinement_progression.png', dpi=300, bbox_inches='tight')
    print("\nSaved: mesh_refinement_progression.png")
    plt.close()

    return meshes


def example_quadratic_elements():
    """Demonstrate quadratic (curved) elements"""
    print("\n" + "=" * 70)
    print("Example 6: Quadratic Elements (Higher-Order)")
    print("=" * 70)

    # Generate base mesh
    mesh_linear = CurvedMeshGenerator()
    mesh_linear.generate_circular_domain(
        center=(0.0, 0.0),
        radius=1.0,
        n_radial=3,
        n_angular=12,
        element_type=ElementType.TRIANGLE_LINEAR
    )

    stats_linear = mesh_linear.get_mesh_statistics()
    print(f"\nLinear mesh:")
    print(f"  Nodes: {stats_linear['num_nodes']}")
    print(f"  Elements: {stats_linear['num_elements']}")

    # Elevate to quadratic
    import copy
    mesh_quadratic = copy.deepcopy(mesh_linear)
    mesh_quadratic.elevate_to_quadratic()

    stats_quadratic = mesh_quadratic.get_mesh_statistics()
    print(f"\nQuadratic mesh:")
    print(f"  Nodes: {stats_quadratic['num_nodes']} "
          f"(+{stats_quadratic['num_nodes'] - stats_linear['num_nodes']} mid-side nodes)")
    print(f"  Elements: {stats_quadratic['num_elements']}")
    print(f"  Element types: {stats_quadratic['element_types']}")

    # Visualize
    viz = MeshVisualizer(mesh_quadratic)
    fig = viz.plot_curved_elements(n_subdivisions=20, figsize=(10, 10))
    plt.savefig('mesh_quadratic_elements.png', dpi=300, bbox_inches='tight')
    print("\nSaved: mesh_quadratic_elements.png")
    plt.close()

    return mesh_linear, mesh_quadratic


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("CURVED MESH GENERATION EXAMPLES")
    print("Demonstrating GRUMMP-like mesh generation capabilities")
    print("=" * 70)

    # Run all examples
    mesh1 = example_rectangular_mesh()
    mesh2 = example_circular_mesh()
    mesh3 = example_delaunay_triangulation()
    mesh4_before, mesh4_after = example_mesh_smoothing()
    mesh5_levels = example_mesh_refinement()
    mesh6_linear, mesh6_quadratic = example_quadratic_elements()

    print("\n" + "=" * 70)
    print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print("\nGenerated files:")
    print("  - mesh_rectangular.png")
    print("  - mesh_circular.png")
    print("  - mesh_delaunay.png")
    print("  - mesh_delaunay_quality.png")
    print("  - mesh_smoothing_comparison.png")
    print("  - mesh_refinement_progression.png")
    print("  - mesh_quadratic_elements.png")
    print("\nThese demonstrate the full range of curved mesh generation capabilities!")
