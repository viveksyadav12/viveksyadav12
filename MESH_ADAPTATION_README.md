# 1D HOMES: High-Order Mesh Optimization via Error Sampling and Synthesis

A Python implementation of the HOMES algorithm for 1D high-order mesh adaptation, based on the paper "Error Sampling and Synthesis for High-Order Node Movement" by Sanjaya, Rangarajan, and Ollivier-Gooch (AIAA 2025-0780).

## Overview

This module provides a complete framework for adaptive mesh optimization in 1D, supporting:

- **Arbitrary polynomial order**: Linear (P1), quadratic (P2), cubic (P3), and higher-order elements
- **Metric-based adaptation**: Riemannian metric fields guide mesh optimization
- **r-adaptation**: Vertex position optimization
- **q-adaptation**: High-order geometry node optimization
- **Error-driven refinement**: A posteriori error estimation via sampling and synthesis

## Key Concepts

### MOESS vs HOMES

- **MOESS** (Mesh Optimization via Error Sampling and Synthesis): Original algorithm for standard (linear) meshes
- **HOMES** (High-Order MOESS): Extension to high-order curved meshes with arbitrary polynomial degree

### Node Types

- **r-nodes** (vertices): Element endpoints that define mesh topology
- **q-nodes** (high-order geometry nodes): Interior nodes that define element curvature

### Adaptation Strategies

1. **r-adaptation**: Optimize vertex positions to match the metric field
   - Changes mesh topology
   - Affects element size distribution
   - Global coupling between all vertices

2. **q-adaptation**: Optimize high-order node positions
   - Preserves mesh topology
   - Affects element shape and curvature
   - Local optimization within each element

3. **Combined r+q**: Alternating or simultaneous optimization
   - Best overall error reduction
   - Captures both sizing and curvature requirements

### Riemannian Metric Field

In 1D, the metric field is simply a scalar size function h(x) that specifies the desired mesh spacing at each location.

The metric-based length of an interval [x₀, x₁] is:

```
L = ∫(x₀ to x₁) (1/h(x)) dx
```

An ideal mesh has all elements with unit metric length (L ≈ 1.0).

### Error Sampling and Synthesis

The error model estimates how well the current mesh conforms to the target metric:

1. **Element splitting**: Analyze element in parametric space
2. **Random sampling**: Perturb q-nodes randomly to sample error landscape
3. **Error kernel estimation**: Measure deviation from ideal metric length
4. **Global synthesis**: Sum element errors to guide optimization

## Installation

### Dependencies

```bash
pip install numpy scipy matplotlib
```

Or use the existing requirements:

```bash
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

```python
import numpy as np
from mesh_adaptation_1d import (
    HighOrderMesh1D,
    RiemannianMetricField1D,
    ErrorModel1D,
    MeshOptimizer1D
)

# Create initial mesh
vertices = np.linspace(0, 1, 11)
mesh = HighOrderMesh1D(vertices, order=2)  # Quadratic elements

# Define target metric (uniform spacing of 0.05)
metric = RiemannianMetricField1D(lambda x: 0.05)

# Create error model and optimizer
error_model = ErrorModel1D(n_samples=20)
optimizer = MeshOptimizer1D(mesh, metric, error_model)

# Adapt mesh
history = optimizer.adapt(n_iterations=20, alternate=True)

# Access optimized mesh
adapted_mesh = optimizer.mesh
```

### Hessian-Based Adaptation

Adapt mesh to capture solution features:

```python
# Define solution with varying curvature
def solution(x):
    return np.sin(2*np.pi*x) + 0.1*np.sin(20*np.pi*x)

# Construct metric from Hessian (second derivative)
metric = RiemannianMetricField1D.from_solution_hessian(
    solution,
    complexity=2.0,  # Controls resolution
    domain=(0, 1)
)

# Adapt as before
mesh = HighOrderMesh1D(np.linspace(0, 1, 11), order=3)
error_model = ErrorModel1D(n_samples=25)
optimizer = MeshOptimizer1D(mesh, metric, error_model)
optimizer.adapt(n_iterations=30)
```

### Non-Uniform Metric

Refine locally in specific regions:

```python
def size_function(x):
    """Fine spacing near x=0.3 and x=0.7"""
    h_min = 0.02  # Fine
    h_max = 0.15  # Coarse

    d1 = abs(x - 0.3)
    d2 = abs(x - 0.7)
    d = min(d1, d2)

    # Smooth transition
    return h_min + (h_max - h_min) / (1 + np.exp(-20*(d - 0.1)))

metric = RiemannianMetricField1D(size_function)
```

## API Reference

### HighOrderMesh1D

**Constructor:**
```python
HighOrderMesh1D(vertices, order=1)
```
- `vertices`: Array of vertex positions (r-nodes)
- `order`: Polynomial order (1=linear, 2=quadratic, 3=cubic, etc.)

**Key Methods:**
- `get_element_nodes(elem)`: Get all nodes for an element
- `eval_element_position(elem, xi)`: Evaluate position at parametric coordinate ξ ∈ [0,1]
- `eval_element_jacobian(elem, xi)`: Evaluate Jacobian dx/dξ
- `plot(ax, show_q_nodes)`: Visualize mesh
- `refine_uniform()`: Create uniformly refined mesh

**Attributes:**
- `vertices`: Vertex positions (r-nodes)
- `q_nodes`: High-order node positions, shape (n_elements, q_nodes_per_elem)
- `connectivity`: Element-to-vertex connectivity
- `n_elements`: Number of elements
- `nodes_per_elem`: Total nodes per element (including vertices)
- `q_nodes_per_elem`: Interior nodes per element (order - 1)

### RiemannianMetricField1D

**Constructor:**
```python
RiemannianMetricField1D(size_function)
```
- `size_function`: Callable h(x) returning desired mesh size

**Static Methods:**
```python
from_solution_hessian(solution, complexity, domain, n_samples=1000)
```
Construct metric from solution Hessian (second derivative).
- `solution`: Callable u(x)
- `complexity`: Mesh complexity parameter (controls resolution)
- `domain`: Tuple (x_min, x_max)

**Methods:**
- `eval_metric(x)`: Evaluate metric h(x) at position x
- `metric_length(x0, x1)`: Compute metric-based length of interval
- `element_metric_length(mesh, elem)`: Metric length of mesh element

### ErrorModel1D

**Constructor:**
```python
ErrorModel1D(pde_residual=None, n_samples=20)
```
- `pde_residual`: Optional function for PDE-based error estimation
- `n_samples`: Number of random samples for q-node perturbation

**Methods:**
- `estimate_error_kernel(mesh, metric, elem)`: Error kernel for element
- `compute_global_error(mesh, metric)`: Global error statistics

**Returns (compute_global_error):**
```python
{
    'total': float,           # Sum of element errors
    'mean': float,            # Mean element error
    'max': float,             # Maximum element error
    'std': float,             # Standard deviation
    'element_errors': array   # Per-element errors
}
```

### MeshOptimizer1D

**Constructor:**
```python
MeshOptimizer1D(mesh, metric, error_model)
```
- `mesh`: Initial HighOrderMesh1D
- `metric`: RiemannianMetricField1D
- `error_model`: ErrorModel1D

**Methods:**

`optimize_r_nodes(max_iter=100, tol=1e-6)`
- Optimize vertex (r-node) positions
- Returns dict with optimization results

`optimize_q_nodes(max_iter=100, tol=1e-6)`
- Optimize high-order geometry (q-node) positions
- Returns dict with optimization results

`adapt(n_iterations=10, alternate=True)`
- Perform mesh adaptation
- `alternate=True`: Alternate between r and q adaptation
- `alternate=False`: Optimize both simultaneously each iteration
- Returns history of iteration statistics

`plot_convergence(ax=None)`
- Visualize error convergence

## Examples

The `example_adaptation_1d.py` file contains 6 comprehensive examples:

1. **Basic Mesh Creation**: Visualize meshes of different polynomial orders
2. **Uniform Metric Adaptation**: Adapt non-uniform mesh to uniform spacing
3. **Non-Uniform Metric**: Local refinement at specific locations
4. **Hessian-Based Adaptation**: Adapt to analytic solution features
5. **Polynomial Order Comparison**: Compare adaptation for P1, P2, P3, P4
6. **r vs q Adaptation**: Compare r-only, q-only, and combined strategies

### Running Examples

```bash
# Run all examples
python example_adaptation_1d.py

# Run individual examples in Python:
from example_adaptation_1d import example_3_nonuniform_metric
example_3_nonuniform_metric()
```

Generated visualizations:
- `example_1_basic_meshes.png`
- `example_2_uniform_metric.png`
- `example_2_convergence.png`
- `example_3_nonuniform_metric.png`
- `example_3_spacing_comparison.png`
- `example_4_hessian_adaptation.png`
- `example_5_order_comparison.png`
- `example_5_error_comparison.png`
- `example_6_r_vs_q.png`

## Theoretical Background

### Error Model

The error kernel measures element quality in metric space:

```
E_elem = (L_metric - 1)² + 0.5 * E_geometry
```

where:
- `L_metric`: Metric-based element length (should be ≈ 1.0)
- `E_geometry`: Error from q-node perturbation sampling

### Optimization Objective

Minimize global error:

```
E_total = Σ E_elem
```

subject to:
- Boundary vertices fixed
- q-nodes stay within their element
- Element ordering preserved

### Shape Functions

High-order elements use Lagrange shape functions:

```
N_i(ξ) = Π_{j≠i} (ξ - ξ_j) / (ξ_i - ξ_j)
```

Position within element:

```
x(ξ) = Σ N_i(ξ) * x_i
```

where x_i are the element node positions (vertices + q-nodes).

## Performance Considerations

### Computational Complexity

- **Error evaluation**: O(n_elem × n_samples × n_quad)
  - `n_elem`: Number of elements
  - `n_samples`: Error sampling points
  - `n_quad`: Quadrature points

- **r-optimization**: O(n_iter × n_vertices × n_elem)
- **q-optimization**: O(n_iter × n_elem × q_per_elem)

### Recommended Settings

| Mesh Size | Order | n_samples | n_iterations |
|-----------|-------|-----------|--------------|
| Small (<20 elem) | 1-2 | 10-15 | 10-20 |
| Medium (20-50 elem) | 2-3 | 15-20 | 15-25 |
| Large (>50 elem) | 2-4 | 20-30 | 20-30 |

### Tips for Faster Convergence

1. **Start with coarse mesh**: Refine adaptively rather than starting fine
2. **Alternate r/q**: Often faster than simultaneous optimization
3. **Reduce n_samples**: Start with 10-15 samples, increase if needed
4. **Use appropriate order**: Higher order isn't always better
5. **Smooth metrics**: Discontinuous metrics may slow convergence

## Integration with PDE Solvers

The adapted meshes can be used with finite element solvers:

```python
# Generate adapted mesh
optimizer = MeshOptimizer1D(mesh, metric, error_model)
optimizer.adapt(n_iterations=20)
adapted_mesh = optimizer.mesh

# Extract nodes for FEM
for elem in range(adapted_mesh.n_elements):
    nodes = adapted_mesh.get_element_nodes(elem)
    # Use nodes for element assembly

    # Quadrature
    for xi, w in gauss_points:
        x = adapted_mesh.eval_element_position(elem, xi)
        J = adapted_mesh.eval_element_jacobian(elem, xi)
        # Compute integrals...
```

## Future Enhancements

Phase 2 and beyond will include:

- **2D extension**: Triangular and quadrilateral elements
- **Anisotropic metrics**: Full metric tensor support
- **Topology changes**: Edge swapping, element insertion/deletion
- **Parallel optimization**: Distributed mesh adaptation
- **Curved boundary support**: Integration with curved geometry
- **Error indicators**: Integration with PDE solution error estimates

## References

1. Sanjaya, D.P., Rangarajan, A., Ollivier-Gooch, C.F., "Error Sampling and Synthesis for High-Order Node Movement", AIAA 2025-0780

2. Loseille, A., Löhner, R., "Adaptive Anisotropic Simulations in Aerodynamics", AIAA Paper 2010-169

3. Persson, P.-O., Peraire, J., "Curved Mesh Generation and Mesh Refinement using Lagrangian Solid Mechanics", AIAA Paper 2009-949

## License

This implementation is part of the viveksyadav12 repository.

## Contact

For questions or issues, please open an issue on GitHub.

---

**Note**: This is a research implementation of the 1D HOMES algorithm. For production use, additional validation and testing is recommended.
