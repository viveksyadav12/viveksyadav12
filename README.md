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

### 1D HOMES: High-Order Mesh Adaptation

A complete implementation of the HOMES (High-Order MOESS) algorithm for adaptive mesh optimization based on error sampling and synthesis. This implements the algorithm from "Error Sampling and Synthesis for High-Order Node Movement" (Sanjaya et al., AIAA 2025-0780).

**Key Features:**
- **Arbitrary Polynomial Order**: Support for P1, P2, P3, P4+ elements
- **Metric-Based Adaptation**: Riemannian metric fields guide optimization
- **r-Adaptation**: Vertex position optimization for optimal mesh spacing
- **q-Adaptation**: High-order geometry node optimization for element curvature
- **Error Sampling & Synthesis**: A posteriori error estimation via random sampling
- **Hessian-Based Metrics**: Automatic metric construction from solution features

**Adaptation Strategies:**
- r-only: Optimize vertex positions (changes topology)
- q-only: Optimize high-order nodes (preserves topology)
- Combined r+q: Alternating or simultaneous optimization

See [MESH_ADAPTATION_README.md](MESH_ADAPTATION_README.md) for full documentation.

**Quick Start:**
```bash
# Run quick verification tests
python test_mesh_adaptation.py

# Run comprehensive examples
python example_adaptation_1d.py
```

**Example - Basic Adaptation:**
```python
from mesh_adaptation_1d import (
    HighOrderMesh1D, RiemannianMetricField1D,
    ErrorModel1D, MeshOptimizer1D
)
import numpy as np

# Create initial mesh (quadratic elements)
vertices = np.linspace(0, 1, 11)
mesh = HighOrderMesh1D(vertices, order=2)

# Define target metric (fine spacing near x=0.5)
metric = RiemannianMetricField1D(
    lambda x: 0.02 + 0.1 * abs(x - 0.5)
)

# Adapt mesh
error_model = ErrorModel1D(n_samples=20)
optimizer = MeshOptimizer1D(mesh, metric, error_model)
optimizer.adapt(n_iterations=20, alternate=True)

# Access optimized mesh
adapted_mesh = optimizer.mesh
```

**Example - Hessian-Based Adaptation:**
```python
# Adapt to solution features
def solution(x):
    return np.sin(2*np.pi*x) + 0.1*np.sin(20*np.pi*x)

# Construct metric from Hessian
metric = RiemannianMetricField1D.from_solution_hessian(
    solution, complexity=2.0, domain=(0, 1)
)

# Adapt as before...
```

### Heart Mesh FEM - Cardiac Electrophysiology

A comprehensive implementation for generating anatomically-inspired heart meshes and simulating cardiac electrophysiology using Finite Element Methods. This project enables realistic simulations of electrical wave propagation in the heart.

**Key Features:**
- **Anatomical Heart Geometries**: Left ventricle, right ventricle, biventricular meshes
- **Cardiac FEM Solver**: Monodomain equation with anisotropic diffusion
- **Ionic Models**: Aliev-Panfilov, FitzHugh-Nagumo cardiac action potential models
- **Fiber Orientations**: Transmural fiber rotation for realistic conduction
- **Stimulus Protocols**: Point stimulation, regional pacing, periodic pacing
- **Advanced Simulations**: Action potential propagation, activation maps, reentrant waves

**Applications:**
- Cardiac arrhythmia research
- Pacing strategy optimization
- Wave propagation studies
- Educational demonstrations

See [HEART_MESH_FEM_README.md](HEART_MESH_FEM_README.md) for full documentation.

**Quick Start:**
```bash
# Test heart mesh generator
python heart_mesh_generator.py

# Run cardiac electrophysiology examples
python example_heart_fem.py lv        # LV propagation
python example_heart_fem.py biv       # Biventricular pacing
python example_heart_fem.py reentry   # Spiral waves
python example_heart_fem.py           # All examples
```

**Example - LV Action Potential:**
```python
from heart_mesh_generator import create_simple_lv_mesh
from cardiac_fem_solver import (
    CardiacFEMSolver, AlievPanfilovModel, StimulusProtocol
)

# Create heart mesh
heart_mesh = create_simple_lv_mesh(n_radial=10, n_angular=30)
mesh_data = heart_mesh.export_for_fem()

# Set up solver
solver = CardiacFEMSolver(
    mesh_data['nodes'],
    mesh_data['elements'],
    fiber_field=mesh_data['fibers'],
    ionic_model=AlievPanfilovModel()
)

# Define apex stimulus
stimulus = StimulusProtocol.point_stimulus(
    center=(0.0, -1.0), radius=0.5,
    amplitude=50.0, duration=2.0
)

# Simulate electrical propagation
times, solutions = solver.solve(
    T=50.0, dt=0.05,
    stimulus_func=stimulus,
    save_interval=10
)
```

**Example - Biventricular Pacing:**
```python
from heart_mesh_generator import create_biventricular_mesh

# Create biventricular mesh
heart_mesh = create_biventricular_mesh()
mesh_data = heart_mesh.export_for_fem()

# Dual-site pacing simulation
# ... (see example_heart_fem.py for complete code)
```

-

<!---
viveksyadav12/viveksyadav12 is a ✨ special ✨ repository because its `README.md` (this file) appears on your GitHub profile.
You can click the Preview link to take a look at your changes.
--->
