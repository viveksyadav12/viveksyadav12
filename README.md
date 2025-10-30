- Hi, I am Vivek Singh Yadav, currently working as a Postdoctoral Research Associate
- I'm interested in numerical methods and scientific computing
- I'm currently learning cuda programming and machine learning to simulate PDEs
- I'm looking to collaborate on numerical methods
- You can reach me at my Email
- Pronouns: He

---

# 2D Laplace Equation Solver using Finite Element Method (FEM)

This repository contains a MATLAB implementation of a 2D Laplace/Poisson equation solver using the Finite Element Method with linear triangular elements.

## Overview

The solver can handle:
- **Homogeneous Laplace equation**: ∇²u = 0
- **Poisson equation**: -∇²u = f(x,y)
- **Dirichlet boundary conditions**: u = g(x,y) on ∂Ω

## Features

- Structured triangular mesh generation
- Linear triangular finite elements
- Sparse matrix assembly for efficiency
- Dirichlet boundary condition enforcement
- Visualization tools (surface and contour plots)
- Multiple example cases included

## Files

### Core Functions
- `laplace2d_fem_solver.m` - Main FEM solver function
- `generate_mesh_2d.m` - Structured triangular mesh generator
- `assemble_system.m` - Global stiffness matrix and load vector assembly
- `apply_dirichlet_bc.m` - Boundary condition application
- `plot_solution.m` - Solution visualization

### Examples
- `quickstart.m` - Simple example to get started
- `example_laplace2d.m` - Comprehensive examples with different BCs and source terms

## Quick Start

Run the quickstart example:
```matlab
quickstart
```

This solves ∇²u = 0 on [0,1]×[0,1] with boundary condition u = sin(πx)sin(πy).

## Usage

### Basic Usage

```matlab
% Define domain
domain = [0, 1, 0, 1];  % [xmin, xmax, ymin, ymax]

% Define boundary condition
bc = @(x,y) sin(pi*x).*sin(pi*y);

% Set mesh resolution
nx = 30;  % elements in x-direction
ny = 30;  % elements in y-direction

% Solve Laplace equation
[u, mesh] = laplace2d_fem_solver(domain, bc, nx, ny);

% Plot solution
plot_solution(mesh, u);
```

### With Source Term (Poisson Equation)

```matlab
% Define source term
source = @(x,y) 2*pi^2*sin(pi*x).*sin(pi*y);

% Solve with source term
[u, mesh] = laplace2d_fem_solver(domain, bc, nx, ny, source);
```

## Examples Included

1. **Homogeneous Laplace** with sinusoidal BC
2. **Polynomial boundary conditions** (u = x² + y²)
3. **Poisson equation** with analytical solution verification
4. **Heat distribution** problem with mixed BCs
5. **Radial boundary condition** demonstration

Run all examples:
```matlab
example_laplace2d
```

## Mathematical Formulation

### Weak Form
Find u ∈ H¹(Ω) such that:
```
∫_Ω ∇u · ∇v dΩ = ∫_Ω f v dΩ  ∀v ∈ H¹₀(Ω)
```

### Discretization
- **Elements**: Linear triangular elements (3 nodes)
- **Shape functions**: Linear basis functions
- **Integration**: Analytical integration for linear elements

### Element Stiffness Matrix
For a triangular element with constant shape function gradients:
```
K_ij^e = ∫_Ωe ∇N_i · ∇N_j dΩ = (b_i*b_j + c_i*c_j)/(4*Area)
```

## Requirements

- MATLAB (tested on R2018b and later)
- No additional toolboxes required

## Algorithm

1. **Mesh Generation**: Create structured triangular mesh
2. **Assembly**: Build global stiffness matrix K and load vector F
3. **Boundary Conditions**: Apply Dirichlet BCs using direct elimination
4. **Solution**: Solve linear system Ku = F
5. **Visualization**: Plot results as surface and contour plots

## Performance

- Uses sparse matrices for memory efficiency
- Typical solve time for 30×30 mesh: < 1 second
- Scales well for moderate mesh sizes (up to ~100×100)

## License

MIT License - Feel free to use and modify

## Contributing

Contributions are welcome! Feel free to submit issues or pull requests.

## References

- Zienkiewicz, O. C., & Taylor, R. L. (2000). The Finite Element Method
- Hughes, T. J. R. (2000). The Finite Element Method: Linear Static and Dynamic Finite Element Analysis

---

<!---
viveksyadav12/viveksyadav12 is a ✨ special ✨ repository because its `README.md` (this file) appears on your GitHub profile.
You can click the Preview link to take a look at your changes.
--->
