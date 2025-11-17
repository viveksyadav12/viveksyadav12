- Hi, I am Vivek Singh Yadav, currently working as a Postdoctoral Research Associate
- I'm interested in numerical methods and scientific computing
- I'm currently learning cuda programming and machine learning to simulate PDEs
- I'm looking to collaborate on numerical methods
- You can reach me at my Email
- Pronouns: He

## Featured Projects

### Time-Dependent Reaction-Diffusion PDE Solver

A comprehensive Python implementation of numerical solvers for reaction-diffusion systems. This project demonstrates advanced finite difference methods for solving time-dependent PDEs that arise in:

- **Pattern Formation**: Gray-Scott model, Turing patterns
- **Population Dynamics**: Fisher's equation, invasion waves
- **Excitable Media**: FitzHugh-Nagumo model, neuronal activity
- **Chemical Oscillators**: Brusselator, Schnakenberg models

**Key Features:**
- Multiple boundary conditions (periodic, Neumann, Dirichlet)
- Comprehensive visualization tools (animations, phase space, power spectra)
- 4 classic models with detailed examples
- Stability analysis and parameter exploration

See [REACTION_DIFFUSION_README.md](REACTION_DIFFUSION_README.md) for full documentation.

**Quick Start:**
```bash
pip install -r requirements.txt
python quick_test.py           # Verify installation
python example_gray_scott.py   # Run Gray-Scott simulation
```

### Curved Mesh Generation (GRUMMP-like)

A powerful unstructured mesh generation library with capabilities similar to GRUMMP (Generation and Refinement of Unstructured, Mixed-element Meshes). Enables solving PDEs on complex geometries with curved boundaries.

**Key Features:**
- **Delaunay Triangulation**: Bowyer-Watson algorithm for automatic unstructured mesh generation
- **Curved Elements**: Linear, quadratic, and cubic elements with mid-side nodes
- **Curved Boundaries**: Parametric curves, circular arcs, splines
- **Mesh Quality**: Quality metrics, Laplacian smoothing, optimization
- **Mesh Refinement**: Uniform and adaptive refinement strategies
- **FEM Integration**: Finite element solver for PDEs on unstructured meshes
- **Comprehensive Visualization**: Quality maps, connectivity graphs, curved element rendering

**Mesh Types:**
- Structured meshes (rectangular, circular domains)
- Unstructured Delaunay triangulation
- Quadratic triangles (P2 elements)
- Mixed triangular/quadrilateral meshes

See [CURVED_MESH_README.md](CURVED_MESH_README.md) for full documentation.

**Quick Start:**
```bash
# Generate basic meshes
python example_curved_mesh_basic.py

# Solve PDEs on unstructured meshes
python example_pde_on_curved_mesh.py
```

**Example - Delaunay Mesh:**
```python
from curved_mesh_generator import CurvedMeshGenerator
import numpy as np

# Random points
points = np.random.rand(100, 2)

# Generate mesh
mesh = CurvedMeshGenerator()
mesh.bowyer_watson_triangulation(points)
mesh.smooth_mesh_laplacian(iterations=10)

# Get statistics
stats = mesh.get_mesh_statistics()
print(f"Quality: {stats['quality_mean']:.3f}")
```
- 

<!---
viveksyadav12/viveksyadav12 is a ✨ special ✨ repository because its `README.md` (this file) appears on your GitHub profile.
You can click the Preview link to take a look at your changes.
--->
