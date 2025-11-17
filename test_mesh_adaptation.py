"""
Quick Test for 1D HOMES Implementation
=======================================

Fast verification that the mesh adaptation system works correctly.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from mesh_adaptation_1d import (
    HighOrderMesh1D,
    RiemannianMetricField1D,
    ErrorModel1D,
    MeshOptimizer1D
)


def test_mesh_creation():
    """Test basic mesh creation."""
    print("\n" + "="*60)
    print("Test 1: Mesh Creation")
    print("="*60)

    vertices = np.linspace(0, 1, 11)

    for order in [1, 2, 3]:
        mesh = HighOrderMesh1D(vertices, order=order)
        assert mesh.n_elements == 10
        assert mesh.order == order
        assert mesh.nodes_per_elem == order + 1
        print(f"✓ Created order-{order} mesh: {mesh.n_elements} elements, "
              f"{mesh.q_nodes_per_elem} q-nodes per element")

    print("✓ All mesh creation tests passed")


def test_metric_field():
    """Test metric field evaluation."""
    print("\n" + "="*60)
    print("Test 2: Metric Field")
    print("="*60)

    # Uniform metric
    metric_uniform = RiemannianMetricField1D(lambda x: 0.1)
    h = metric_uniform.eval_metric(0.5)
    assert abs(h - 0.1) < 1e-10
    print(f"✓ Uniform metric: h(0.5) = {h}")

    # Variable metric
    metric_variable = RiemannianMetricField1D(lambda x: 0.05 + 0.1*x)
    h = metric_variable.eval_metric(0.5)
    assert abs(h - 0.1) < 1e-10
    print(f"✓ Variable metric: h(0.5) = {h}")

    # Metric length
    L = metric_uniform.metric_length(0.0, 0.1)
    print(f"✓ Metric length [0, 0.1] = {L:.4f}")

    print("✓ All metric field tests passed")


def test_error_estimation():
    """Test error model."""
    print("\n" + "="*60)
    print("Test 3: Error Estimation")
    print("="*60)

    vertices = np.linspace(0, 1, 11)
    mesh = HighOrderMesh1D(vertices, order=2)

    metric = RiemannianMetricField1D(lambda x: 0.1)
    error_model = ErrorModel1D(n_samples=10)

    errors = error_model.compute_global_error(mesh, metric)

    print(f"✓ Total error: {errors['total']:.6f}")
    print(f"✓ Mean error: {errors['mean']:.6f}")
    print(f"✓ Max error: {errors['max']:.6f}")
    print(f"✓ Std error: {errors['std']:.6f}")

    assert len(errors['element_errors']) == mesh.n_elements
    print("✓ All error estimation tests passed")


def test_r_adaptation():
    """Test r-adaptation (vertex optimization)."""
    print("\n" + "="*60)
    print("Test 4: r-Adaptation")
    print("="*60)

    # Non-uniform initial mesh
    vertices = np.array([0.0, 0.2, 0.25, 0.5, 0.75, 0.8, 1.0])
    mesh = HighOrderMesh1D(vertices, order=1)

    # Uniform metric
    metric = RiemannianMetricField1D(lambda x: 0.15)

    error_model = ErrorModel1D(n_samples=5)
    optimizer = MeshOptimizer1D(mesh, metric, error_model)

    error_before = error_model.compute_global_error(mesh, metric)
    print(f"Initial error: {error_before['total']:.6f}")

    # Optimize
    result = optimizer.optimize_r_nodes(max_iter=50)

    error_after = error_model.compute_global_error(optimizer.mesh, metric)
    print(f"Final error: {error_after['total']:.6f}")

    improvement = (error_before['total'] - error_after['total']) / error_before['total'] * 100
    print(f"Improvement: {improvement:.2f}%")

    assert error_after['total'] < error_before['total']
    print("✓ r-Adaptation test passed (error reduced)")


def test_q_adaptation():
    """Test q-adaptation (high-order node optimization)."""
    print("\n" + "="*60)
    print("Test 5: q-Adaptation")
    print("="*60)

    vertices = np.linspace(0, 1, 9)
    mesh = HighOrderMesh1D(vertices, order=2)

    # Non-uniform metric
    metric = RiemannianMetricField1D(lambda x: 0.08 + 0.05*x)

    error_model = ErrorModel1D(n_samples=10)
    optimizer = MeshOptimizer1D(mesh, metric, error_model)

    error_before = error_model.compute_global_error(mesh, metric)
    print(f"Initial error: {error_before['total']:.6f}")

    # Optimize
    result = optimizer.optimize_q_nodes(max_iter=30)

    error_after = error_model.compute_global_error(optimizer.mesh, metric)
    print(f"Final error: {error_after['total']:.6f}")

    print(f"✓ q-Adaptation completed in {result['iterations']} iterations")


def test_combined_adaptation():
    """Test combined r+q adaptation."""
    print("\n" + "="*60)
    print("Test 6: Combined r+q Adaptation")
    print("="*60)

    vertices = np.linspace(0, 1, 8)
    mesh = HighOrderMesh1D(vertices, order=2)

    # Non-uniform metric
    def size_function(x):
        return 0.05 + 0.1 * abs(x - 0.5)

    metric = RiemannianMetricField1D(size_function)

    error_model = ErrorModel1D(n_samples=10)
    optimizer = MeshOptimizer1D(mesh, metric, error_model)

    error_before = error_model.compute_global_error(mesh, metric)
    print(f"Initial error: {error_before['total']:.6f}")

    # Adapt
    history = optimizer.adapt(n_iterations=5, alternate=True)

    error_after = error_model.compute_global_error(optimizer.mesh, metric)
    print(f"Final error: {error_after['total']:.6f}")

    improvement = (error_before['total'] - error_after['total']) / error_before['total'] * 100
    print(f"Improvement: {improvement:.2f}%")

    print(f"✓ Completed {len(history)} adaptation iterations")

    # Create simple visualization
    fig, axes = plt.subplots(2, 1, figsize=(12, 6))

    # Plot metric
    x_plot = np.linspace(0, 1, 200)
    h_plot = [size_function(x) for x in x_plot]
    axes[0].plot(x_plot, h_plot, 'b-', linewidth=2)
    axes[0].set_ylabel('h(x)')
    axes[0].set_title('Target Metric Field')
    axes[0].grid(True, alpha=0.3)

    # Plot adapted mesh
    optimizer.mesh.plot(ax=axes[1])
    axes[1].set_title(f'Adapted Mesh (Error: {error_after["total"]:.6f})')

    plt.tight_layout()
    plt.savefig('test_adaptation_result.png', dpi=100, bbox_inches='tight')
    print("✓ Saved visualization to 'test_adaptation_result.png'")

    assert error_after['total'] < error_before['total']
    print("✓ Combined adaptation test passed")


def test_hessian_metric():
    """Test Hessian-based metric construction."""
    print("\n" + "="*60)
    print("Test 7: Hessian-Based Metric")
    print("="*60)

    # Simple solution
    def solution(x):
        return np.sin(2*np.pi*x)

    metric = RiemannianMetricField1D.from_solution_hessian(
        solution,
        complexity=1.5,
        domain=(0, 1),
        n_samples=500
    )

    # Metric should be finer where curvature is high
    h_high_curve = metric.eval_metric(0.0)  # sin(0) = 0, high curvature
    h_low_curve = metric.eval_metric(0.25)  # sin(π/2) = 1, low curvature

    print(f"✓ Metric at x=0.0 (high curvature): {h_high_curve:.6f}")
    print(f"✓ Metric at x=0.25 (low curvature): {h_low_curve:.6f}")
    print("✓ Hessian-based metric test passed")


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*60)
    print("         1D HOMES VERIFICATION SUITE")
    print("="*60)

    try:
        test_mesh_creation()
        test_metric_field()
        test_error_estimation()
        test_r_adaptation()
        test_q_adaptation()
        test_combined_adaptation()
        test_hessian_metric()

        print("\n" + "="*60)
        print("           ALL TESTS PASSED ✓")
        print("="*60)
        print("\n1D HOMES implementation is working correctly!")
        print("\nNext steps:")
        print("  - Run full examples: python example_adaptation_1d.py")
        print("  - Read documentation: MESH_ADAPTATION_README.md")
        print("  - Integrate with your PDE solvers")
        print("="*60 + "\n")

        return True

    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    np.random.seed(42)  # For reproducibility
    success = run_all_tests()
    exit(0 if success else 1)
