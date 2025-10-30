# Time-Dependent Reaction-Diffusion PDE Solver

A comprehensive Python implementation of numerical solvers for time-dependent reaction-diffusion partial differential equations (PDEs). This project demonstrates advanced numerical methods for solving PDEs that arise in biology, chemistry, physics, and pattern formation.

## Overview

Reaction-diffusion systems describe how the concentration of one or more substances distributed in space changes under the influence of two processes:
- **Diffusion**: Random motion spreading substances uniformly
- **Reaction**: Chemical or biological processes that transform substances

The general form is:
```
∂u/∂t = D_u∇²u + R(u, v, ...)
∂v/∂t = D_v∇²v + R'(u, v, ...)
```

where:
- `u, v`: Concentration fields
- `D_u, D_v`: Diffusion coefficients
- `R, R'`: Reaction terms (nonlinear functions)
- `∇²`: Laplacian operator (spatial diffusion)

## Features

### Numerical Methods
- **Finite Difference Method**: Second-order central differences for spatial derivatives
- **Explicit Euler Integration**: Forward Euler time stepping
- **Multiple Boundary Conditions**: Periodic, Neumann (zero-flux), Dirichlet (fixed value)
- **Stability Analysis**: Automatic stability condition checking

### Implemented Models

1. **Gray-Scott Model**: Pattern formation system producing spots, stripes, spirals
   - Applications: Chemical patterns, morphogenesis
   - Parameters: Feed rate (F), kill rate (k)

2. **Turing Patterns** (Schnakenberg, Brusselator): Spontaneous pattern formation
   - Applications: Animal coat patterns, developmental biology
   - Key: Different diffusion rates create instability

3. **Fisher's Equation**: Population dynamics and invasion waves
   - Applications: Ecology, epidemiology, genetics
   - Shows traveling wave solutions

4. **FitzHugh-Nagumo Model**: Neuronal excitation and inhibition
   - Applications: Cardiac electrophysiology, neuroscience
   - Exhibits excitable media, spiral waves

### Visualization Tools
- Concentration field snapshots
- Time evolution plots
- Animated simulations (GIF export)
- 1D concentration profiles
- Phase space diagrams
- Power spectrum analysis (for pattern wavelengths)

## Installation

```bash
pip install -r requirements.txt
```

Requirements:
- Python 3.7+
- NumPy ≥ 1.20.0
- Matplotlib ≥ 3.3.0
- SciPy ≥ 1.6.0 (optional)

## Quick Start

### Example 1: Gray-Scott Pattern Formation

```python
import numpy as np
from reaction_diffusion_solver import (
    ReactionDiffusionSystem,
    ReactionDiffusionSolver,
    gray_scott_reaction,
    random_perturbation
)
from visualization import plot_evolution

# Configure system
system = ReactionDiffusionSystem(
    Lx=2.5, Ly=2.5,          # Domain size
    Nx=256, Ny=256,          # Grid resolution
    T=10000, dt=1.0,         # Time parameters
    Du=2e-5, Dv=1e-5,        # Diffusion coefficients
    bc_type='periodic'
)

# Create solver
solver = ReactionDiffusionSolver(system)

# Set initial conditions
u0, v0 = random_perturbation(system, u_base=1.0, v_base=0.0, noise_level=0.01)
solver.set_initial_conditions(u0, v0)

# Solve with Gray-Scott reaction
u_history, v_history, times = solver.solve(
    gray_scott_reaction,
    F=0.04,  # Feed rate
    k=0.06,  # Kill rate
    save_every=100
)

# Visualize
plot_evolution(u_history, v_history, times, n_snapshots=6)
```

### Example 2: Turing Patterns

```python
from reaction_diffusion_solver import schnakenberg_reaction

system = ReactionDiffusionSystem(
    Lx=2.0, Ly=2.0, Nx=256, Ny=256,
    T=5000, dt=0.1,
    Du=1e-4,   # Slow activator
    Dv=1e-3,   # Fast inhibitor (10x)
    bc_type='neumann'
)

solver = ReactionDiffusionSolver(system)

# Start near homogeneous steady state
u_star, v_star = 1.0, 0.81
u0, v0 = random_perturbation(system, u_base=u_star, v_base=v_star, noise_level=0.05)
solver.set_initial_conditions(u0, v0)

# Solve
u_history, v_history, times = solver.solve(
    schnakenberg_reaction,
    a=0.1, b=0.9,
    save_every=50
)
```

### Example 3: Fisher's Equation (1D Wave)

```python
from reaction_diffusion_solver import fisher_reaction

# Quasi-1D domain
system = ReactionDiffusionSystem(
    Lx=10.0, Ly=1.0, Nx=512, Ny=32,
    T=200, dt=0.01,
    Du=1e-3, bc_type='neumann'
)

solver = ReactionDiffusionSolver(system)

# Localized initial population
u0 = np.zeros((system.Ny, system.Nx))
u0[:, :int(0.1 * system.Nx)] = 1.0  # Population on left side

solver.set_initial_conditions(u0)

# Solve Fisher equation
u_history, _, times = solver.solve(
    fisher_reaction,
    r=1.0,  # Growth rate
    K=1.0,  # Carrying capacity
    save_every=50
)
```

## Running Examples

The repository includes detailed example scripts:

```bash
# Gray-Scott mitosis pattern
python example_gray_scott.py

# Turing pattern formation
python example_turing_patterns.py

# Fisher invasion wave
python example_fisher.py

# FitzHugh-Nagumo excitable media
python example_fitzhugh_nagumo.py
```

## Mathematical Background

### Finite Difference Discretization

Spatial derivatives (Laplacian):
```
∇²u ≈ (u[i+1,j] - 2u[i,j] + u[i-1,j])/dx²
    + (u[i,j+1] - 2u[i,j] + u[i,j-1])/dy²
```

Time stepping (Explicit Euler):
```
u^(n+1) = u^n + dt * (D∇²u^n + R(u^n, v^n))
```

### Stability Condition

For explicit schemes, stability requires:
```
r_x + r_y ≤ 0.5
where r_x = D*dt/dx², r_y = D*dt/dy²
```

The solver automatically checks this condition and warns if violated.

### Turing Instability Conditions

Turing patterns require:
1. Homogeneous steady state exists
2. Linearization around steady state is stable (no oscillations)
3. Different diffusion rates: typically D_v >> D_u
4. Specific parameter relationships for instability

## Customization

### Defining New Reaction Functions

```python
def my_reaction(u, v, param1=1.0, param2=0.5):
    """
    Custom reaction function.

    Returns:
        Tuple of (R_u, R_v) reaction terms
    """
    R_u = param1 * u - u * v
    R_v = u * v - param2 * v
    return R_u, R_v

# Use in solver
u_history, v_history, times = solver.solve(
    my_reaction,
    param1=1.5,
    param2=0.3,
    save_every=100
)
```

### Custom Initial Conditions

```python
# Gaussian bump
sigma = 0.2
u0 = np.exp(-((system.X - system.Lx/2)**2 + (system.Y - system.Ly/2)**2) / (2*sigma**2))
v0 = np.zeros_like(u0)

# Multiple spots
u0 = np.zeros((system.Ny, system.Nx))
for x, y in [(0.3, 0.3), (0.7, 0.7), (0.3, 0.7)]:
    r2 = (system.X - x*system.Lx)**2 + (system.Y - y*system.Ly)**2
    u0 += np.exp(-r2 / 0.05)
```

## Parameter Exploration

### Gray-Scott Parameters

Different (F, k) values produce different patterns:

| F | k | Pattern |
|---|---|---------|
| 0.04 | 0.06 | Mitosis (dividing spots) |
| 0.03 | 0.06 | Stripes |
| 0.05 | 0.065 | Spirals |
| 0.014 | 0.054 | Maze patterns |
| 0.062 | 0.061 | Moving spots |

### Turing Pattern Conditions

For Schnakenberg with a=0.1, b=0.9:
- Requires D_v/D_u > 2 for patterns
- Wavelength scales with √(D_v/D_u)
- Larger diffusion ratio → larger patterns

## Performance Tips

1. **Grid Resolution**: Start with coarse grid (128×128), increase for details
2. **Time Step**: Respect stability condition (solver warns automatically)
3. **Saving Frequency**: `save_every` parameter controls memory usage
4. **Boundary Conditions**: Periodic boundaries are faster than Neumann/Dirichlet

## Applications

### Biology
- Morphogenesis (how organisms develop patterns)
- Animal coat patterns (zebra stripes, leopard spots)
- Bacterial colonies
- Vegetation patterns in semi-arid regions

### Chemistry
- Belousov-Zhabotinsky reaction
- Chemical oscillators
- Autocatalytic reactions

### Medicine
- Cardiac arrhythmias (spiral waves)
- Neuronal activity (excitable media)
- Tumor growth models
- Wound healing

### Physics
- Phase transitions
- Crystal growth
- Phase separation

## References

### Classic Papers
1. Turing, A.M. (1952). "The Chemical Basis of Morphogenesis." *Philosophical Transactions of the Royal Society B*, 237(641), 37-72.

2. Gray, P., & Scott, S.K. (1985). "Autocatalytic reactions in the isothermal, continuous stirred tank reactor: Oscillations and instabilities in the system A + 2B → 3B; B → C." *Chemical Engineering Science*, 39(6), 1087-1097.

3. Fisher, R.A. (1937). "The wave of advance of advantageous genes." *Annals of Eugenics*, 7(4), 355-369.

4. FitzHugh, R. (1961). "Impulses and Physiological States in Theoretical Models of Nerve Membrane." *Biophysical Journal*, 1(6), 445-466.

### Books
- Murray, J.D. (2002). *Mathematical Biology I: An Introduction* (3rd ed.). Springer.
- Strogatz, S.H. (2015). *Nonlinear Dynamics and Chaos* (2nd ed.). Westview Press.

## Author

**Vivek Singh Yadav**
Postdoctoral Research Associate
Research Interests: Numerical Methods, Scientific Computing, PDE Simulation

## License

This project is open source and available for educational and research purposes.

## Acknowledgments

This implementation is inspired by classic works in mathematical biology and pattern formation theory. Special thanks to the reaction-diffusion community for decades of fascinating research.
