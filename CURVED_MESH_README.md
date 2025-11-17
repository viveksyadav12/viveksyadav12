# Curved Mesh Generation - GRUMMP-like Capabilities

This module provides comprehensive unstructured and curved mesh generation capabilities similar to the GRUMMP (Generation and Refinement of Unstructured, Mixed-element Meshes in Parallel) software library.

## Overview

The curved mesh generator enables solving PDEs on complex geometries with curved boundaries and unstructured meshes. It integrates seamlessly with the existing reaction-diffusion PDE solver to enable finite element analysis on arbitrary domains.

## Features

### 1. **Mesh Generation Algorithms**
- **Structured Meshes**: Rectangular and circular domains with regular grids
- **Delaunay Triangulation**: Bowyer-Watson algorithm for unstructured mesh generation
- **Boundary-Conforming**: Meshes that respect curved and complex boundaries

### 2. **Element Types**
- **Linear Elements** (P1):
  - 3-node triangles
  - 4-node quadrilaterals
- **Quadratic Elements** (P2):
  - 6-node triangles with mid-side nodes
  - 9-node quadrilaterals
- **Cubic Elements** (P3):
  - 10-node triangles (planned)

### 3. **Curved Boundaries**
- Straight line segments
- Circular arcs
- Cubic splines
- Parametric curves (arbitrary functions)
- Ellipses and other conic sections

### 4. **Mesh Quality & Optimization**
- **Quality Metrics**: Aspect ratio, inscribed/circumscribed circle ratio
- **Laplacian Smoothing**: Iterative mesh optimization
- **Quality Visualization**: Histograms and color-coded quality maps

### 5. **Mesh Refinement**
- **Uniform Refinement**: Split each triangle into 4 sub-triangles
- **Adaptive Refinement**: Refine based on solution gradients (planned)
- **Multi-level Refinement**: Hierarchical mesh structures

### 6. **PDE Solving on Unstructured Meshes**
- **Finite Element Method**: P1 (linear) finite elements
- **Time Integration**: Forward Euler (explicit)
- **Boundary Conditions**: Dirichlet and Neumann
- **Coupled Systems**: Multi-species reaction-diffusion

## Module Structure

```
curved_mesh_generator.py       # Core mesh generation classes and algorithms
mesh_visualization.py          # Comprehensive visualization tools
unstructured_pde_solver.py     # FEM solver for unstructured meshes
example_curved_mesh_basic.py   # Basic mesh generation examples
example_pde_on_curved_mesh.py  # PDE solving examples
```

## Quick Start

### 1. Generate a Simple Rectangular Mesh

```python
from curved_mesh_generator import CurvedMeshGenerator, ElementType
from mesh_visualization import MeshVisualizer
import matplotlib.pyplot as plt

# Create mesh generator
mesh_gen = CurvedMeshGenerator()

# Generate rectangular mesh
mesh_gen.generate_rectangular_domain(
    x_min=0.0, x_max=2.0,
    y_min=0.0, y_max=1.0,
    nx=20, ny=10,
    element_type=ElementType.TRIANGLE_LINEAR
)

# Visualize
viz = MeshVisualizer(mesh_gen)
viz.plot_mesh(show_nodes=True, show_boundary=True, color_by_quality=True)
plt.savefig('my_mesh.png')
```

### 2. Generate Circular Domain with Radial Mesh

```python
# Create circular mesh
mesh_gen = CurvedMeshGenerator()
mesh_gen.generate_circular_domain(
    center=(0.0, 0.0),
    radius=1.0,
    n_radial=10,
    n_angular=24
)

# Get statistics
stats = mesh_gen.get_mesh_statistics()
print(f"Nodes: {stats['num_nodes']}")
print(f"Elements: {stats['num_elements']}")
print(f"Mean quality: {stats['quality_mean']:.3f}")
```

### 3. Delaunay Triangulation (Unstructured)

```python
import numpy as np

# Generate random points
np.random.seed(42)
points = np.random.rand(100, 2)

# Create Delaunay triangulation
mesh_gen = CurvedMeshGenerator()
mesh_gen.bowyer_watson_triangulation(points)

# Smooth for better quality
mesh_gen.smooth_mesh_laplacian(iterations=10, relaxation=0.5)
```

### 4. Create Quadratic (Curved) Elements

```python
# Start with linear mesh
mesh_gen = CurvedMeshGenerator()
mesh_gen.generate_circular_domain(
    center=(0.0, 0.0),
    radius=1.0,
    n_radial=5,
    n_angular=16
)

# Elevate to quadratic
mesh_gen.elevate_to_quadratic()

# Visualize curved elements
viz = MeshVisualizer(mesh_gen)
viz.plot_curved_elements(n_subdivisions=20)
plt.savefig('curved_elements.png')
```

### 5. Mesh Refinement

```python
import copy

# Create coarse mesh
mesh_coarse = CurvedMeshGenerator()
mesh_coarse.generate_rectangular_domain(0, 1, 0, 1, 5, 5)

# Refine uniformly
mesh_fine = copy.deepcopy(mesh_coarse)
mesh_fine.refine_mesh_uniform()

print(f"Coarse: {len(mesh_coarse.elements)} elements")
print(f"Fine: {len(mesh_fine.elements)} elements")
```

### 6. Solve PDE on Unstructured Mesh

```python
from unstructured_pde_solver import UnstructuredPDESystem, UnstructuredPDESolver

# Generate mesh
mesh_gen = CurvedMeshGenerator()
mesh_gen.generate_circular_domain(
    center=(0.0, 0.0),
    radius=1.0,
    n_radial=10,
    n_angular=24
)

# Setup PDE system (heat equation)
system = UnstructuredPDESystem(
    mesh=mesh_gen,
    T=1.0,          # Final time
    dt=0.01,        # Time step
    Du=0.1,         # Diffusion coefficient
    boundary_condition="dirichlet",
    boundary_value_u=0.0
)

# Initial condition: Gaussian
u0 = np.zeros(len(mesh_gen.nodes))
for nid, node in mesh_gen.nodes.items():
    r = np.sqrt(node.x**2 + node.y**2)
    u0[nid] = np.exp(-10 * r**2)

# Solve
solver = UnstructuredPDESolver(system)
u_history, _ = solver.solve(u0, store_interval=10)

print(f"Solution shape: {u_history.shape}")
```

## Advanced Examples

### Curved Boundary Definition

```python
from curved_mesh_generator import (
    BoundaryCurve, BoundaryType,
    create_curved_boundary_circle,
    create_curved_boundary_ellipse
)
import numpy as np

# Circular boundary
circle = create_curved_boundary_circle(
    center=(0.0, 0.0),
    radius=1.0,
    n_points=100
)

# Elliptical boundary
ellipse = create_curved_boundary_ellipse(
    center=(0.0, 0.0),
    a=2.0,  # Semi-major axis
    b=1.0,  # Semi-minor axis
    n_points=100
)

# Custom parametric boundary
def heart_curve(t):
    """Heart-shaped boundary"""
    theta = 2 * np.pi * t
    x = 16 * np.sin(theta)**3
    y = 13*np.cos(theta) - 5*np.cos(2*theta) - 2*np.cos(3*theta) - np.cos(4*theta)
    return x/20, y/20

heart = BoundaryCurve(
    id=0,
    curve_type=BoundaryType.PARAMETRIC,
    control_points=np.array([[0, 0]]),  # Dummy
    parametric_func=heart_curve
)

# Evaluate boundary at parameter t
x, y = heart.evaluate(0.5)
```

### Reaction-Diffusion on Unstructured Mesh

```python
from unstructured_pde_solver import UnstructuredPDESystem, UnstructuredPDESolver

# Generate mesh
mesh_gen = CurvedMeshGenerator()
mesh_gen.generate_rectangular_domain(0, 2.5, 0, 2.5, 40, 40)

# Gray-Scott parameters
Du, Dv = 0.16, 0.08
F, k = 0.060, 0.062

system = UnstructuredPDESystem(
    mesh=mesh_gen,
    T=10000,
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

# Add perturbation
for nid, node in mesh_gen.nodes.items():
    if 1.0 < node.x < 1.5 and 1.0 < node.y < 1.5:
        u0[nid] = 0.5
        v0[nid] = 0.25

# Solve
solver = UnstructuredPDESolver(system)
u_history, v_history = solver.solve(
    u0, v0,
    reaction_u=reaction_u,
    reaction_v=reaction_v,
    store_interval=1000
)
```

### Adaptive Mesh Refinement

```python
from unstructured_pde_solver import (
    UnstructuredPDESolver,
    interpolate_solution_to_mesh
)

# Solve on coarse mesh
solver_coarse = UnstructuredPDESolver(system_coarse)
u_history, _ = solver_coarse.solve(u0)
u_final = u_history[-1, :]

# Refine based on solution gradient
mesh_refined = solver_coarse.adaptive_refine(u_final, threshold=0.5)

# Interpolate solution to refined mesh
u_refined = interpolate_solution_to_mesh(u_final, mesh_coarse, mesh_refined)

# Continue solving on refined mesh
system_refined = UnstructuredPDESystem(
    mesh=mesh_refined,
    T=1.0, dt=0.01, Du=0.1
)
solver_refined = UnstructuredPDESolver(system_refined)
u_history_refined, _ = solver_refined.solve(u_refined)
```

## Visualization Tools

### Basic Mesh Visualization

```python
from mesh_visualization import MeshVisualizer

viz = MeshVisualizer(mesh_gen)

# Standard mesh plot
viz.plot_mesh(
    show_nodes=True,
    show_node_ids=False,
    show_element_ids=False,
    show_boundary=True,
    color_by_quality=True
)

# Quality histogram
viz.plot_quality_histogram(bins=20)

# Node connectivity graph
viz.plot_node_connectivity()

# Save to file
viz.save_mesh_figure('my_mesh.png', dpi=300)
```

### Curved Elements Visualization

```python
viz.plot_curved_elements(n_subdivisions=20)
```

### Refinement Progression

```python
from mesh_visualization import plot_refinement_progression

meshes = [mesh_level0, mesh_level1, mesh_level2]
titles = ['Coarse', 'Medium', 'Fine']
plot_refinement_progression(meshes, titles=titles)
```

## Mesh Quality Metrics

The mesh generator computes element quality using the ratio of inscribed to circumscribed circle radii:

$$Q = 2 \frac{r_{in}}{r_{out}}$$

where:
- $Q = 1$ for equilateral triangles (perfect quality)
- $Q \to 0$ for degenerate/sliver elements

Access quality metrics:

```python
# Element-wise quality
quality = mesh_gen.compute_element_quality(element_id)

# Global statistics
stats = mesh_gen.get_mesh_statistics()
print(f"Min quality: {stats['quality_min']}")
print(f"Mean quality: {stats['quality_mean']}")
print(f"Max quality: {stats['quality_max']}")
```

## Mesh Data Export

Export mesh to numpy arrays for custom solvers:

```python
# Export to arrays
nodes, elements = mesh_gen.export_to_arrays()

# nodes: (num_nodes, 2) - node coordinates
# elements: (num_elements, max_nodes_per_element) - connectivity

print(f"Nodes shape: {nodes.shape}")
print(f"Elements shape: {elements.shape}")
```

## Performance Considerations

### Memory Usage
- Linear elements: ~3 integers per element
- Quadratic elements: ~6 integers per element
- FEM matrices: O(N²) for dense, O(N) for sparse (future)

### Computational Complexity
- Delaunay triangulation: O(N log N) average case
- Mesh smoothing: O(N × iterations)
- FEM assembly: O(N_elements)
- FEM solve: O(N_nodes × N_timesteps)

### Optimization Tips
1. **Use coarser meshes** when exploring parameters
2. **Refine adaptively** only where needed
3. **Store solutions sparingly** (large store_interval)
4. **Use Neumann BCs** when possible (no constraint equations)

## Comparison with GRUMMP

| Feature | This Implementation | GRUMMP |
|---------|-------------------|---------|
| Delaunay triangulation | ✅ Bowyer-Watson | ✅ Multiple algorithms |
| Curved elements | ✅ Quadratic (P2) | ✅ Up to P3 |
| Mesh refinement | ✅ Uniform | ✅ Adaptive + Uniform |
| 3D meshes | ❌ 2D only | ✅ Full 3D |
| Parallel | ❌ Serial | ✅ MPI parallel |
| Mixed elements | ⚠️ Triangles + Quads | ✅ Full support |
| Boundary curves | ✅ Parametric curves | ✅ CAD integration |

## Future Enhancements

1. **Advanced Refinement**
   - Error-based adaptive refinement
   - Anisotropic refinement
   - Mesh coarsening

2. **3D Mesh Generation**
   - Tetrahedral meshes
   - Hexahedral meshes
   - Curved 3D boundaries

3. **Performance**
   - Sparse matrix storage
   - Iterative solvers (CG, GMRES)
   - Parallel assembly

4. **Advanced Elements**
   - Cubic elements (P3)
   - Mixed formulations
   - Discontinuous Galerkin

5. **Geometry**
   - CAD file import (.step, .iges)
   - Automatic boundary detection
   - Mesh healing

## Examples

Run the included examples:

```bash
# Basic mesh generation examples
python example_curved_mesh_basic.py

# PDE solving on curved meshes
python example_pde_on_curved_mesh.py
```

This will generate visualizations demonstrating:
- Rectangular and circular meshes
- Delaunay triangulation
- Mesh smoothing
- Mesh refinement
- Quadratic elements
- Heat equation on circular domain
- Fisher's equation on unstructured mesh
- Gray-Scott patterns
- Adaptive refinement

## References

1. **GRUMMP**: C.L. Stimpson et al., "The GRUMMP mesh generation toolkit"
2. **Delaunay Triangulation**: P. Cignoni et al., "DeWall: A fast divide and conquer Delaunay triangulation algorithm"
3. **Mesh Quality**: J. Shewchuk, "What Is a Good Linear Finite Element?"
4. **FEM**: O.C. Zienkiewicz, R.L. Taylor, "The Finite Element Method"

## License

This code is part of the reaction-diffusion PDE solver project by Vivek Singh Yadav.

## Author

**Vivek Singh Yadav**
Postdoctoral Research Associate
Date: 2025-11-17

---

For questions or issues, please refer to the main repository README.
