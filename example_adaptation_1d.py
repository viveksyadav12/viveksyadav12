"""
1D HOMES Examples and Demonstrations
=====================================

Comprehensive examples showing the capabilities of the 1D HOMES
(High-Order MOESS) mesh adaptation framework.

Examples:
1. Basic mesh creation and visualization
2. Uniform metric adaptation
3. Non-uniform metric adaptation (refinement in specific regions)
4. Adaptation to analytic solution
5. Comparison of different polynomial orders
6. r-adaptation vs q-adaptation comparison
"""

import numpy as np
import matplotlib.pyplot as plt
from mesh_adaptation_1d import (
    HighOrderMesh1D,
    RiemannianMetricField1D,
    ErrorModel1D,
    MeshOptimizer1D,
    visualize_adaptation
)


def example_1_basic_mesh():
    """Example 1: Basic mesh creation and visualization."""
    print("\n" + "="*70)
    print("Example 1: Basic Mesh Creation and Visualization")
    print("="*70)

    fig, axes = plt.subplots(3, 1, figsize=(14, 8))

    # Linear mesh
    vertices = np.linspace(0, 1, 11)
    mesh_p1 = HighOrderMesh1D(vertices, order=1)
    mesh_p1.plot(ax=axes[0])
    axes[0].set_title('Linear Mesh (P1)', fontsize=12, fontweight='bold')

    # Quadratic mesh
    mesh_p2 = HighOrderMesh1D(vertices, order=2)
    mesh_p2.plot(ax=axes[1])
    axes[1].set_title('Quadratic Mesh (P2)', fontsize=12, fontweight='bold')

    # Cubic mesh
    mesh_p3 = HighOrderMesh1D(vertices, order=3)
    mesh_p3.plot(ax=axes[2])
    axes[2].set_title('Cubic Mesh (P3)', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig('example_1_basic_meshes.png', dpi=150, bbox_inches='tight')
    print("✓ Created meshes of orders 1, 2, and 3")
    print("✓ Saved visualization to 'example_1_basic_meshes.png'")

    return fig


def example_2_uniform_metric():
    """Example 2: Adaptation to uniform metric field."""
    print("\n" + "="*70)
    print("Example 2: Uniform Metric Adaptation")
    print("="*70)

    # Create non-uniform initial mesh
    vertices = np.array([0.0, 0.1, 0.15, 0.3, 0.5, 0.6, 0.85, 0.9, 1.0])
    mesh_initial = HighOrderMesh1D(vertices, order=2)

    print(f"Initial mesh: {mesh_initial.n_elements} elements")

    # Uniform metric (constant spacing)
    metric = RiemannianMetricField1D(lambda x: 0.1)

    # Create error model and optimizer
    error_model = ErrorModel1D(n_samples=15)
    optimizer = MeshOptimizer1D(mesh_initial, metric, error_model)

    # Compute initial error
    error_initial = error_model.compute_global_error(mesh_initial, metric)
    print(f"Initial error: {error_initial['total']:.6f}")

    # Adapt mesh
    print("\nAdapting mesh...")
    history = optimizer.adapt(n_iterations=20, alternate=True)

    # Compute final error
    error_final = error_model.compute_global_error(optimizer.mesh, metric)
    print(f"Final error: {error_final['total']:.6f}")
    print(f"Error reduction: {(1 - error_final['total']/error_initial['total'])*100:.2f}%")

    # Visualize results
    fig, axes = visualize_adaptation(
        mesh_initial, optimizer.mesh, metric,
        title="Example 2: Uniform Metric Adaptation"
    )
    plt.savefig('example_2_uniform_metric.png', dpi=150, bbox_inches='tight')
    print("✓ Saved adaptation results to 'example_2_uniform_metric.png'")

    # Plot convergence
    fig_conv, ax_conv = plt.subplots(figsize=(10, 6))
    optimizer.plot_convergence(ax=ax_conv)
    plt.savefig('example_2_convergence.png', dpi=150, bbox_inches='tight')
    print("✓ Saved convergence plot to 'example_2_convergence.png'")

    return fig, fig_conv


def example_3_nonuniform_metric():
    """Example 3: Adaptation to non-uniform metric (local refinement)."""
    print("\n" + "="*70)
    print("Example 3: Non-Uniform Metric Adaptation")
    print("="*70)

    # Create uniform initial mesh
    vertices = np.linspace(0, 1, 11)
    mesh_initial = HighOrderMesh1D(vertices, order=2)

    # Non-uniform metric: refine near x=0.3 and x=0.7
    def size_function(x):
        h_min = 0.02  # Fine spacing
        h_max = 0.15  # Coarse spacing

        # Distance to refinement centers
        d1 = abs(x - 0.3)
        d2 = abs(x - 0.7)
        d = min(d1, d2)

        # Smooth transition
        h = h_min + (h_max - h_min) * (1 / (1 + np.exp(-20*(d - 0.1))))
        return h

    metric = RiemannianMetricField1D(size_function)

    # Adapt
    error_model = ErrorModel1D(n_samples=20)
    optimizer = MeshOptimizer1D(mesh_initial, metric, error_model)

    error_initial = error_model.compute_global_error(mesh_initial, metric)
    print(f"Initial error: {error_initial['total']:.6f}")

    print("\nAdapting mesh to non-uniform metric...")
    history = optimizer.adapt(n_iterations=25, alternate=False)

    error_final = error_model.compute_global_error(optimizer.mesh, metric)
    print(f"Final error: {error_final['total']:.6f}")
    print(f"Error reduction: {(1 - error_final['total']/error_initial['total'])*100:.2f}%")

    # Visualize
    fig, axes = visualize_adaptation(
        mesh_initial, optimizer.mesh, metric,
        title="Example 3: Non-Uniform Metric (Local Refinement at x=0.3, 0.7)"
    )
    plt.savefig('example_3_nonuniform_metric.png', dpi=150, bbox_inches='tight')
    print("✓ Saved adaptation results to 'example_3_nonuniform_metric.png'")

    # Show mesh spacing
    fig_spacing, ax = plt.subplots(figsize=(12, 6))

    # Plot target metric
    x_plot = np.linspace(0, 1, 500)
    h_plot = np.array([size_function(x) for x in x_plot])
    ax.plot(x_plot, h_plot, 'b-', linewidth=2, label='Target metric h(x)')

    # Plot actual mesh spacing
    vertices_final = optimizer.mesh.vertices
    h_actual = np.diff(vertices_final)
    x_actual = 0.5 * (vertices_final[:-1] + vertices_final[1:])
    ax.plot(x_actual, h_actual, 'ro-', markersize=8, linewidth=2,
            label='Actual mesh spacing', alpha=0.7)

    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('Mesh spacing h', fontsize=12)
    ax.set_title('Metric Field vs Actual Mesh Spacing', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.savefig('example_3_spacing_comparison.png', dpi=150, bbox_inches='tight')
    print("✓ Saved spacing comparison to 'example_3_spacing_comparison.png'")

    return fig, fig_spacing


def example_4_analytic_solution():
    """Example 4: Adaptation based on solution Hessian."""
    print("\n" + "="*70)
    print("Example 4: Hessian-Based Adaptation")
    print("="*70)

    # Analytic solution with varying curvature
    def solution(x):
        return np.sin(2*np.pi*x) + 0.5*np.sin(10*np.pi*x) + 0.1*np.sin(50*np.pi*x)

    # Create initial mesh
    vertices = np.linspace(0, 1, 15)
    mesh_initial = HighOrderMesh1D(vertices, order=3)

    # Construct metric from Hessian
    complexity = 2.0  # Controls overall mesh resolution
    metric = RiemannianMetricField1D.from_solution_hessian(
        solution, complexity=complexity, domain=(0, 1), n_samples=1000
    )

    print("Constructed metric from solution Hessian")

    # Adapt
    error_model = ErrorModel1D(n_samples=25)
    optimizer = MeshOptimizer1D(mesh_initial, metric, error_model)

    error_initial = error_model.compute_global_error(mesh_initial, metric)
    print(f"Initial error: {error_initial['total']:.6f}")

    print("\nAdapting mesh...")
    history = optimizer.adapt(n_iterations=30, alternate=False)

    error_final = error_model.compute_global_error(optimizer.mesh, metric)
    print(f"Final error: {error_final['total']:.6f}")
    print(f"Error reduction: {(1 - error_final['total']/error_initial['total'])*100:.2f}%")

    # Visualize solution and mesh
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    # Plot solution
    x_plot = np.linspace(0, 1, 1000)
    u_plot = solution(x_plot)
    axes[0].plot(x_plot, u_plot, 'b-', linewidth=2)
    axes[0].set_ylabel('u(x)', fontsize=11)
    axes[0].set_title('Analytic Solution', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    # Plot metric field
    h_plot = np.array([metric.eval_metric(x) for x in x_plot])
    axes[1].plot(x_plot, h_plot, 'g-', linewidth=2)
    axes[1].set_ylabel('h(x)', fontsize=11)
    axes[1].set_title('Metric Field (from Hessian)', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    # Plot adapted mesh
    optimizer.mesh.plot(ax=axes[2])
    axes[2].set_title('Adapted Mesh', fontsize=12, fontweight='bold')

    plt.suptitle('Example 4: Hessian-Based Adaptation', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('example_4_hessian_adaptation.png', dpi=150, bbox_inches='tight')
    print("✓ Saved results to 'example_4_hessian_adaptation.png'")

    return fig


def example_5_polynomial_order_comparison():
    """Example 5: Compare adaptation for different polynomial orders."""
    print("\n" + "="*70)
    print("Example 5: Polynomial Order Comparison")
    print("="*70)

    # Non-uniform metric
    def size_function(x):
        return 0.05 + 0.1 * np.sin(4*np.pi*x)**2

    metric = RiemannianMetricField1D(size_function)

    orders = [1, 2, 3, 4]
    results = {}

    fig, axes = plt.subplots(len(orders), 1, figsize=(14, 3*len(orders)))

    for idx, order in enumerate(orders):
        print(f"\n--- Testing order {order} ---")

        # Create mesh
        vertices = np.linspace(0, 1, 12)
        mesh = HighOrderMesh1D(vertices, order=order)

        # Adapt
        error_model = ErrorModel1D(n_samples=15)
        optimizer = MeshOptimizer1D(mesh, metric, error_model)

        error_initial = error_model.compute_global_error(mesh, metric)
        print(f"Initial error: {error_initial['total']:.6f}")

        history = optimizer.adapt(n_iterations=15, alternate=True)

        error_final = error_model.compute_global_error(optimizer.mesh, metric)
        print(f"Final error: {error_final['total']:.6f}")

        results[order] = {
            'initial_error': error_initial['total'],
            'final_error': error_final['total'],
            'history': history,
            'mesh': optimizer.mesh
        }

        # Plot
        optimizer.mesh.plot(ax=axes[idx])
        axes[idx].set_title(f'Adapted Mesh (Order {order})', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig('example_5_order_comparison.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved comparison to 'example_5_order_comparison.png'")

    # Plot error comparison
    fig_error, ax = plt.subplots(figsize=(10, 6))

    for order in orders:
        iterations = [h['iteration'] for h in results[order]['history']]
        errors = [h['error_after'] for h in results[order]['history']]
        ax.semilogy(iterations, errors, 'o-', linewidth=2, markersize=7,
                   label=f'Order {order}')

    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Total Error', fontsize=12)
    ax.set_title('Convergence Comparison: Different Polynomial Orders',
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.savefig('example_5_error_comparison.png', dpi=150, bbox_inches='tight')
    print("✓ Saved error comparison to 'example_5_error_comparison.png'")

    return fig, fig_error


def example_6_r_vs_q_adaptation():
    """Example 6: Compare r-adaptation vs q-adaptation."""
    print("\n" + "="*70)
    print("Example 6: r-Adaptation vs q-Adaptation Comparison")
    print("="*70)

    # Create metric
    def size_function(x):
        return 0.03 + 0.12 * (x - 0.5)**2

    metric = RiemannianMetricField1D(size_function)

    # Test with quadratic elements
    vertices = np.linspace(0, 1, 11)

    # r-adaptation only
    print("\n--- r-Adaptation only ---")
    mesh_r = HighOrderMesh1D(vertices.copy(), order=2)
    error_model = ErrorModel1D(n_samples=20)
    optimizer_r = MeshOptimizer1D(mesh_r, metric, error_model)

    error_r_initial = error_model.compute_global_error(mesh_r, metric)
    print(f"Initial error: {error_r_initial['total']:.6f}")

    # Only r-adaptation
    for i in range(10):
        optimizer_r.optimize_r_nodes()

    error_r_final = error_model.compute_global_error(optimizer_r.mesh, metric)
    print(f"Final error (r-only): {error_r_final['total']:.6f}")

    # q-adaptation only
    print("\n--- q-Adaptation only ---")
    mesh_q = HighOrderMesh1D(vertices.copy(), order=2)
    optimizer_q = MeshOptimizer1D(mesh_q, metric, error_model)

    error_q_initial = error_model.compute_global_error(mesh_q, metric)

    # Only q-adaptation
    for i in range(10):
        optimizer_q.optimize_q_nodes()

    error_q_final = error_model.compute_global_error(optimizer_q.mesh, metric)
    print(f"Final error (q-only): {error_q_final['total']:.6f}")

    # Combined r+q adaptation
    print("\n--- Combined r+q Adaptation ---")
    mesh_rq = HighOrderMesh1D(vertices.copy(), order=2)
    optimizer_rq = MeshOptimizer1D(mesh_rq, metric, error_model)

    history_rq = optimizer_rq.adapt(n_iterations=20, alternate=True)

    error_rq_final = error_model.compute_global_error(optimizer_rq.mesh, metric)
    print(f"Final error (r+q): {error_rq_final['total']:.6f}")

    # Visualize comparison
    fig, axes = plt.subplots(4, 1, figsize=(14, 12))

    # Metric field
    x_plot = np.linspace(0, 1, 500)
    h_plot = np.array([size_function(x) for x in x_plot])
    axes[0].plot(x_plot, h_plot, 'b-', linewidth=2)
    axes[0].set_ylabel('h(x)', fontsize=11)
    axes[0].set_title('Target Metric Field', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    # r-only
    optimizer_r.mesh.plot(ax=axes[1])
    axes[1].set_title(f'r-Adaptation Only (Error: {error_r_final["total"]:.6f})',
                     fontsize=12, fontweight='bold')

    # q-only
    optimizer_q.mesh.plot(ax=axes[2])
    axes[2].set_title(f'q-Adaptation Only (Error: {error_q_final["total"]:.6f})',
                     fontsize=12, fontweight='bold')

    # Combined
    optimizer_rq.mesh.plot(ax=axes[3])
    axes[3].set_title(f'Combined r+q Adaptation (Error: {error_rq_final["total"]:.6f})',
                     fontsize=12, fontweight='bold')

    plt.suptitle('Example 6: r vs q vs Combined Adaptation',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('example_6_r_vs_q.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved comparison to 'example_6_r_vs_q.png'")

    print("\n" + "-"*70)
    print("Summary:")
    print(f"  r-only error:     {error_r_final['total']:.6f}")
    print(f"  q-only error:     {error_q_final['total']:.6f}")
    print(f"  Combined error:   {error_rq_final['total']:.6f}")
    print(f"  Best approach:    {'r+q combined' if error_rq_final['total'] < min(error_r_final['total'], error_q_final['total']) else 'r-only' if error_r_final['total'] < error_q_final['total'] else 'q-only'}")
    print("-"*70)

    return fig


def run_all_examples():
    """Run all examples."""
    print("\n")
    print("="*70)
    print(" "*15 + "1D HOMES DEMONSTRATION SUITE")
    print("="*70)
    print("\nRunning comprehensive examples of 1D mesh adaptation...")
    print("This will generate visualizations for each example.\n")

    try:
        example_1_basic_mesh()
        example_2_uniform_metric()
        example_3_nonuniform_metric()
        example_4_analytic_solution()
        example_5_polynomial_order_comparison()
        example_6_r_vs_q_adaptation()

        print("\n" + "="*70)
        print(" "*20 + "ALL EXAMPLES COMPLETED!")
        print("="*70)
        print("\nGenerated files:")
        print("  - example_1_basic_meshes.png")
        print("  - example_2_uniform_metric.png")
        print("  - example_2_convergence.png")
        print("  - example_3_nonuniform_metric.png")
        print("  - example_3_spacing_comparison.png")
        print("  - example_4_hessian_adaptation.png")
        print("  - example_5_order_comparison.png")
        print("  - example_5_error_comparison.png")
        print("  - example_6_r_vs_q.png")
        print("\nAll visualizations saved successfully!")
        print("="*70 + "\n")

    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)

    # Run all examples
    run_all_examples()

    # Show plots (comment out if running in batch mode)
    # plt.show()
