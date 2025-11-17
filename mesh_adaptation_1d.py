"""
1D HOMES (High-Order MOESS) Implementation
==========================================

Mesh Optimization via Error Sampling and Synthesis for 1D high-order meshes.

This module implements the HOMES algorithm for 1D mesh adaptation, including:
- High-order mesh representation with arbitrary polynomial order
- Riemannian metric field construction
- Error sampling and synthesis
- r-adaptation (vertex movement)
- q-adaptation (high-order geometry node movement)

References:
-----------
Sanjaya, D.P., Rangarajan, A., Ollivier-Gooch, C.F., "Error Sampling and Synthesis
for High-Order Node Movement", AIAA 2025-0780
"""

import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import lagrange, BarycentricInterpolator
from scipy.integrate import quad
from typing import Callable, Tuple, List, Optional, Dict
import matplotlib.pyplot as plt


class HighOrderMesh1D:
    """
    1D High-order mesh with arbitrary polynomial order.

    Stores:
    - Vertex positions (r-nodes): endpoints of elements
    - High-order geometry nodes (q-nodes): interior nodes
    - Element connectivity and order
    """

    def __init__(self, vertices: np.ndarray, order: int = 1):
        """
        Initialize 1D high-order mesh.

        Parameters:
        -----------
        vertices : np.ndarray
            Array of vertex positions (r-nodes), shape (n_vertices,)
        order : int
            Polynomial order (1=linear, 2=quadratic, 3=cubic, etc.)
        """
        self.vertices = np.array(vertices, dtype=float)
        self.order = order
        self.n_vertices = len(vertices)
        self.n_elements = self.n_vertices - 1

        # Number of nodes per element (including vertices)
        self.nodes_per_elem = order + 1

        # Number of interior (high-order) nodes per element
        self.q_nodes_per_elem = order - 1

        # Initialize connectivity (each element connects two consecutive vertices)
        self.connectivity = np.array([[i, i+1] for i in range(self.n_elements)])

        # Initialize high-order nodes (q-nodes) using uniform spacing
        self._initialize_q_nodes()

    def _initialize_q_nodes(self):
        """Initialize high-order geometry nodes uniformly within each element."""
        if self.order == 1:
            # Linear elements have no interior nodes
            self.q_nodes = np.array([]).reshape(self.n_elements, 0)
        else:
            # Uniform spacing in parametric space [0, 1]
            xi = np.linspace(0, 1, self.order + 1)[1:-1]  # Exclude endpoints

            self.q_nodes = np.zeros((self.n_elements, self.q_nodes_per_elem))

            for elem in range(self.n_elements):
                v0, v1 = self.vertices[self.connectivity[elem]]
                # Map from parametric to physical space
                self.q_nodes[elem] = v0 + xi * (v1 - v0)

    def get_element_nodes(self, elem: int) -> np.ndarray:
        """
        Get all nodes (vertices + q-nodes) for an element.

        Parameters:
        -----------
        elem : int
            Element index

        Returns:
        --------
        nodes : np.ndarray
            All node positions [v0, q1, q2, ..., q_{p-1}, v1]
        """
        v_indices = self.connectivity[elem]
        v_nodes = self.vertices[v_indices]

        if self.order == 1:
            return v_nodes
        else:
            q = self.q_nodes[elem]
            return np.concatenate([[v_nodes[0]], q, [v_nodes[1]]])

    def eval_element_position(self, elem: int, xi: float) -> float:
        """
        Evaluate position within element using high-order shape functions.

        Parameters:
        -----------
        elem : int
            Element index
        xi : float
            Parametric coordinate in [0, 1]

        Returns:
        --------
        x : float
            Physical position
        """
        nodes = self.get_element_nodes(elem)

        # Lagrange shape functions on [0, 1]
        xi_nodes = np.linspace(0, 1, self.nodes_per_elem)

        # Evaluate Lagrange basis functions
        N = self._lagrange_basis(xi, xi_nodes)

        return np.dot(N, nodes)

    def eval_element_jacobian(self, elem: int, xi: float) -> float:
        """
        Evaluate Jacobian dx/dxi at parametric coordinate xi.

        Parameters:
        -----------
        elem : int
            Element index
        xi : float
            Parametric coordinate in [0, 1]

        Returns:
        --------
        J : float
            Jacobian dx/dxi
        """
        nodes = self.get_element_nodes(elem)
        xi_nodes = np.linspace(0, 1, self.nodes_per_elem)

        # Derivative of Lagrange basis functions
        dN_dxi = self._lagrange_basis_derivative(xi, xi_nodes)

        return np.dot(dN_dxi, nodes)

    @staticmethod
    def _lagrange_basis(xi: float, xi_nodes: np.ndarray) -> np.ndarray:
        """Evaluate Lagrange basis functions at xi."""
        n = len(xi_nodes)
        N = np.zeros(n)

        for i in range(n):
            N[i] = 1.0
            for j in range(n):
                if i != j:
                    N[i] *= (xi - xi_nodes[j]) / (xi_nodes[i] - xi_nodes[j])

        return N

    @staticmethod
    def _lagrange_basis_derivative(xi: float, xi_nodes: np.ndarray) -> np.ndarray:
        """Evaluate derivatives of Lagrange basis functions at xi."""
        n = len(xi_nodes)
        dN = np.zeros(n)

        for i in range(n):
            for k in range(n):
                if k != i:
                    prod = 1.0
                    for j in range(n):
                        if j != i and j != k:
                            prod *= (xi - xi_nodes[j]) / (xi_nodes[i] - xi_nodes[j])
                    dN[i] += prod / (xi_nodes[i] - xi_nodes[k])

        return dN

    def get_mesh_bounds(self) -> Tuple[float, float]:
        """Get spatial bounds of the mesh."""
        return self.vertices[0], self.vertices[-1]

    def refine_uniform(self) -> 'HighOrderMesh1D':
        """
        Uniformly refine mesh by splitting each element into two.

        Returns:
        --------
        refined_mesh : HighOrderMesh1D
            Refined mesh with same order
        """
        new_vertices = np.zeros(2 * self.n_elements + 1)
        new_vertices[::2] = self.vertices

        # Insert midpoints
        for i in range(self.n_elements):
            v0, v1 = self.vertices[i:i+2]
            new_vertices[2*i + 1] = 0.5 * (v0 + v1)

        return HighOrderMesh1D(new_vertices, order=self.order)

    def plot(self, ax=None, show_q_nodes=True):
        """Visualize the mesh."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 3))

        # Plot vertices
        ax.plot(self.vertices, np.zeros_like(self.vertices), 'ro',
                markersize=10, label='Vertices (r-nodes)', zorder=3)

        # Plot q-nodes
        if show_q_nodes and self.order > 1:
            q_flat = self.q_nodes.flatten()
            ax.plot(q_flat, np.zeros_like(q_flat), 'bs',
                    markersize=8, label='High-order nodes (q-nodes)', zorder=3)

        # Plot elements
        for elem in range(self.n_elements):
            xi = np.linspace(0, 1, 50)
            x = np.array([self.eval_element_position(elem, xi_i) for xi_i in xi])
            ax.plot(x, np.zeros_like(x), 'k-', alpha=0.3, linewidth=2)

        ax.set_ylim(-0.5, 0.5)
        ax.set_xlabel('x')
        ax.set_title(f'1D High-Order Mesh (order {self.order})')
        ax.legend()
        ax.grid(True, alpha=0.3)

        return ax


class RiemannianMetricField1D:
    """
    1D Riemannian metric field for mesh adaptation.

    In 1D, the metric is simply a scalar size field h(x) that specifies
    the desired mesh spacing at each point.
    """

    def __init__(self, size_function: Callable[[float], float]):
        """
        Initialize metric field.

        Parameters:
        -----------
        size_function : callable
            Function h(x) that returns desired mesh size at position x
        """
        self.size_function = size_function

    def eval_metric(self, x: float) -> float:
        """
        Evaluate metric (desired mesh size) at position x.

        Parameters:
        -----------
        x : float
            Position

        Returns:
        --------
        h : float
            Desired mesh size
        """
        return self.size_function(x)

    def metric_length(self, x0: float, x1: float, n_quad: int = 50) -> float:
        """
        Compute metric-based length of interval [x0, x1].

        In Riemannian metric, length = integral of (1/h(x)) dx

        Parameters:
        -----------
        x0, x1 : float
            Interval endpoints
        n_quad : int
            Number of quadrature points

        Returns:
        --------
        length : float
            Metric-based length
        """
        def integrand(x):
            return 1.0 / self.size_function(x)

        length, _ = quad(integrand, x0, x1, limit=n_quad)
        return length

    def element_metric_length(self, mesh: HighOrderMesh1D, elem: int,
                             n_quad: int = 50) -> float:
        """
        Compute metric-based length of a mesh element.

        Parameters:
        -----------
        mesh : HighOrderMesh1D
            Mesh containing the element
        elem : int
            Element index
        n_quad : int
            Number of quadrature points

        Returns:
        --------
        length : float
            Metric-based length
        """
        # Quadrature in parametric space
        xi_quad, w_quad = np.polynomial.legendre.leggauss(n_quad)
        xi_quad = 0.5 * (xi_quad + 1.0)  # Map from [-1, 1] to [0, 1]
        w_quad = 0.5 * w_quad

        length = 0.0
        for xi, w in zip(xi_quad, w_quad):
            x = mesh.eval_element_position(elem, xi)
            J = mesh.eval_element_jacobian(elem, xi)
            h = self.size_function(x)
            length += w * J / h

        return length

    @staticmethod
    def from_solution_hessian(solution: Callable[[float], float],
                            complexity: float,
                            domain: Tuple[float, float],
                            n_samples: int = 1000) -> 'RiemannianMetricField1D':
        """
        Construct metric field from solution Hessian (second derivative).

        This is a common approach in a posteriori error estimation.

        Parameters:
        -----------
        solution : callable
            Solution function u(x)
        complexity : float
            Mesh complexity (controls overall resolution)
        domain : tuple
            Domain bounds (x_min, x_max)
        n_samples : int
            Number of samples for normalization

        Returns:
        --------
        metric : RiemannianMetricField1D
            Metric field based on solution Hessian
        """
        # Numerical second derivative
        eps = 1e-6

        def hessian(x):
            u_plus = solution(x + eps)
            u_center = solution(x)
            u_minus = solution(x - eps)
            return (u_plus - 2*u_center + u_minus) / eps**2

        # Sample Hessian over domain
        x_samples = np.linspace(domain[0], domain[1], n_samples)
        H_samples = np.array([abs(hessian(x)) for x in x_samples])

        # Avoid division by zero
        H_samples = np.maximum(H_samples, 1e-12)

        # Normalize to achieve desired complexity
        # In 1D: h(x) = C * |H(x)|^{-1/3} where C is chosen for complexity
        H_mean = np.mean(H_samples)
        C = complexity / (domain[1] - domain[0]) * H_mean**(-1/3)

        def size_function(x):
            H = abs(hessian(x))
            H = max(H, 1e-12)
            return C * H**(-1/3)

        return RiemannianMetricField1D(size_function)


class ErrorModel1D:
    """
    Error sampling and synthesis for 1D HOMES.

    Implements the error kernel estimation via:
    1. Element splitting
    2. Random sampling of q-node movements
    3. Metric-based error estimation
    """

    def __init__(self, pde_residual: Optional[Callable] = None,
                 n_samples: int = 20):
        """
        Initialize error model.

        Parameters:
        -----------
        pde_residual : callable, optional
            Function to compute PDE residual for error estimation
        n_samples : int
            Number of random samples for q-node perturbation
        """
        self.pde_residual = pde_residual
        self.n_samples = n_samples

    def estimate_error_kernel(self, mesh: HighOrderMesh1D,
                             metric: RiemannianMetricField1D,
                             elem: int) -> float:
        """
        Estimate metric-based error kernel for an element.

        The error kernel measures how much the element deviates from
        the ideal metric-based configuration.

        Parameters:
        -----------
        mesh : HighOrderMesh1D
            Current mesh
        metric : RiemannianMetricField1D
            Target metric field
        elem : int
            Element index

        Returns:
        --------
        error : float
            Error kernel estimate
        """
        # Current metric length
        L_metric = metric.element_metric_length(mesh, elem)

        # Ideal metric length is 1.0 (unit edge in metric space)
        L_ideal = 1.0

        # Basic error: deviation from unit metric length
        error_length = abs(L_metric - L_ideal)**2

        if mesh.order == 1:
            # Linear elements: only length error
            return error_length

        # High-order elements: also check q-node placement
        # Sample random q-node perturbations
        error_geometry = 0.0

        for _ in range(self.n_samples):
            # Perturb q-nodes randomly
            perturbed_mesh = self._perturb_q_nodes(mesh, elem)

            # Compute metric length with perturbed q-nodes
            L_perturbed = metric.element_metric_length(perturbed_mesh, elem)

            # Error increases if perturbation changes metric length significantly
            error_geometry += abs(L_perturbed - L_ideal)**2

        error_geometry /= self.n_samples

        # Combined error: length + geometry
        return error_length + 0.5 * error_geometry

    def _perturb_q_nodes(self, mesh: HighOrderMesh1D, elem: int,
                        perturbation_scale: float = 0.1) -> HighOrderMesh1D:
        """
        Create a perturbed copy of mesh with random q-node movement.

        Parameters:
        -----------
        mesh : HighOrderMesh1D
            Original mesh
        elem : int
            Element to perturb
        perturbation_scale : float
            Scale of random perturbation relative to element size

        Returns:
        --------
        perturbed_mesh : HighOrderMesh1D
            Mesh with perturbed q-nodes
        """
        # Create copy
        perturbed = HighOrderMesh1D(mesh.vertices.copy(), order=mesh.order)
        perturbed.q_nodes = mesh.q_nodes.copy()

        if mesh.order > 1:
            # Element size
            v0, v1 = mesh.vertices[mesh.connectivity[elem]]
            h = abs(v1 - v0)

            # Random perturbation
            perturbation = perturbation_scale * h * (np.random.rand(mesh.q_nodes_per_elem) - 0.5)
            perturbed.q_nodes[elem] += perturbation

        return perturbed

    def compute_global_error(self, mesh: HighOrderMesh1D,
                           metric: RiemannianMetricField1D) -> Dict[str, float]:
        """
        Compute global error metrics.

        Parameters:
        -----------
        mesh : HighOrderMesh1D
            Current mesh
        metric : RiemannianMetricField1D
            Target metric field

        Returns:
        --------
        errors : dict
            Dictionary with error statistics
        """
        element_errors = np.zeros(mesh.n_elements)

        for elem in range(mesh.n_elements):
            element_errors[elem] = self.estimate_error_kernel(mesh, metric, elem)

        return {
            'total': np.sum(element_errors),
            'mean': np.mean(element_errors),
            'max': np.max(element_errors),
            'std': np.std(element_errors),
            'element_errors': element_errors
        }


class MeshOptimizer1D:
    """
    Mesh optimization via r-adaptation and q-adaptation.

    r-adaptation: Optimize vertex (r-node) positions
    q-adaptation: Optimize high-order geometry (q-node) positions
    """

    def __init__(self, mesh: HighOrderMesh1D, metric: RiemannianMetricField1D,
                 error_model: ErrorModel1D):
        """
        Initialize optimizer.

        Parameters:
        -----------
        mesh : HighOrderMesh1D
            Initial mesh
        metric : RiemannianMetricField1D
            Target metric field
        error_model : ErrorModel1D
            Error model for optimization objective
        """
        self.mesh = mesh
        self.metric = metric
        self.error_model = error_model
        self.history = []

    def optimize_r_nodes(self, max_iter: int = 100, tol: float = 1e-6) -> Dict:
        """
        Optimize vertex (r-node) positions to match metric field.

        Parameters:
        -----------
        max_iter : int
            Maximum optimization iterations
        tol : float
            Convergence tolerance

        Returns:
        --------
        result : dict
            Optimization results
        """
        # Fix boundary vertices
        x0 = self.mesh.vertices[1:-1].copy()
        bounds = [(self.mesh.vertices[0], self.mesh.vertices[-1])
                 for _ in range(len(x0))]

        def objective(r_nodes):
            # Update mesh vertices
            temp_mesh = HighOrderMesh1D(
                np.concatenate([[self.mesh.vertices[0]], r_nodes, [self.mesh.vertices[-1]]]),
                order=self.mesh.order
            )
            temp_mesh.q_nodes = self.mesh.q_nodes.copy()

            # Compute total error
            errors = self.error_model.compute_global_error(temp_mesh, self.metric)
            return errors['total']

        result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                         options={'maxiter': max_iter, 'ftol': tol})

        # Update mesh
        self.mesh.vertices[1:-1] = result.x

        return {
            'success': result.success,
            'iterations': result.nit,
            'final_error': result.fun,
            'message': result.message
        }

    def optimize_q_nodes(self, max_iter: int = 100, tol: float = 1e-6) -> Dict:
        """
        Optimize high-order geometry (q-node) positions.

        Parameters:
        -----------
        max_iter : int
            Maximum optimization iterations
        tol : float
            Convergence tolerance

        Returns:
        --------
        result : dict
            Optimization results
        """
        if self.mesh.order == 1:
            return {'success': True, 'iterations': 0,
                   'final_error': 0.0, 'message': 'No q-nodes for linear mesh'}

        x0 = self.mesh.q_nodes.flatten()

        # Bounds: q-nodes must stay within their element
        bounds = []
        for elem in range(self.mesh.n_elements):
            v0, v1 = self.mesh.vertices[self.mesh.connectivity[elem]]
            for _ in range(self.mesh.q_nodes_per_elem):
                bounds.append((v0, v1))

        def objective(q_nodes_flat):
            # Update mesh q-nodes
            temp_mesh = HighOrderMesh1D(self.mesh.vertices.copy(), order=self.mesh.order)
            temp_mesh.q_nodes = q_nodes_flat.reshape(self.mesh.n_elements,
                                                     self.mesh.q_nodes_per_elem)

            # Compute total error
            errors = self.error_model.compute_global_error(temp_mesh, self.metric)
            return errors['total']

        result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                         options={'maxiter': max_iter, 'ftol': tol})

        # Update mesh
        self.mesh.q_nodes = result.x.reshape(self.mesh.n_elements,
                                             self.mesh.q_nodes_per_elem)

        return {
            'success': result.success,
            'iterations': result.nit,
            'final_error': result.fun,
            'message': result.message
        }

    def adapt(self, n_iterations: int = 10, alternate: bool = True) -> List[Dict]:
        """
        Perform mesh adaptation via alternating r and q optimization.

        Parameters:
        -----------
        n_iterations : int
            Number of adaptation iterations
        alternate : bool
            If True, alternate between r and q adaptation
            If False, optimize both simultaneously

        Returns:
        --------
        history : list
            List of dictionaries with iteration statistics
        """
        self.history = []

        for it in range(n_iterations):
            # Compute error before optimization
            error_before = self.error_model.compute_global_error(self.mesh, self.metric)

            if alternate:
                # Alternate r and q adaptation
                if it % 2 == 0:
                    result_r = self.optimize_r_nodes()
                    result_q = {'success': True, 'iterations': 0}
                else:
                    result_r = {'success': True, 'iterations': 0}
                    result_q = self.optimize_q_nodes()
            else:
                # Optimize both
                result_r = self.optimize_r_nodes()
                result_q = self.optimize_q_nodes()

            # Compute error after optimization
            error_after = self.error_model.compute_global_error(self.mesh, self.metric)

            history_entry = {
                'iteration': it,
                'error_before': error_before['total'],
                'error_after': error_after['total'],
                'r_iterations': result_r['iterations'],
                'q_iterations': result_q['iterations'],
                'r_success': result_r['success'],
                'q_success': result_q['success']
            }

            self.history.append(history_entry)

            # Check convergence
            error_reduction = error_before['total'] - error_after['total']
            if error_reduction < 1e-8:
                print(f"Converged at iteration {it}")
                break

        return self.history

    def plot_convergence(self, ax=None):
        """Plot convergence history."""
        if not self.history:
            print("No optimization history available")
            return

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        iterations = [h['iteration'] for h in self.history]
        errors = [h['error_after'] for h in self.history]

        ax.semilogy(iterations, errors, 'o-', linewidth=2, markersize=8)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Total Error')
        ax.set_title('Mesh Adaptation Convergence')
        ax.grid(True, alpha=0.3)

        return ax


def visualize_adaptation(mesh_initial: HighOrderMesh1D,
                        mesh_final: HighOrderMesh1D,
                        metric: RiemannianMetricField1D,
                        title: str = "Mesh Adaptation Results"):
    """
    Visualize mesh adaptation results.

    Parameters:
    -----------
    mesh_initial : HighOrderMesh1D
        Initial mesh
    mesh_final : HighOrderMesh1D
        Adapted mesh
    metric : RiemannianMetricField1D
        Target metric field
    title : str
        Plot title
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    # Plot metric field
    x_plot = np.linspace(*mesh_initial.get_mesh_bounds(), 500)
    h_plot = np.array([metric.eval_metric(x) for x in x_plot])

    axes[0].plot(x_plot, h_plot, 'b-', linewidth=2)
    axes[0].set_ylabel('Metric h(x)')
    axes[0].set_title('Target Metric Field')
    axes[0].grid(True, alpha=0.3)

    # Plot initial mesh
    mesh_initial.plot(ax=axes[1])
    axes[1].set_title('Initial Mesh')

    # Plot final mesh
    mesh_final.plot(ax=axes[2])
    axes[2].set_title('Adapted Mesh')

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    return fig, axes


if __name__ == "__main__":
    # Quick test
    print("1D HOMES Implementation")
    print("=" * 50)

    # Create simple mesh
    vertices = np.linspace(0, 1, 11)
    mesh = HighOrderMesh1D(vertices, order=2)

    print(f"Created mesh with {mesh.n_elements} elements, order {mesh.order}")
    print(f"Vertices: {mesh.n_vertices}")
    print(f"Q-nodes per element: {mesh.q_nodes_per_elem}")

    # Create uniform metric
    metric = RiemannianMetricField1D(lambda x: 0.1)

    # Test error model
    error_model = ErrorModel1D(n_samples=10)
    errors = error_model.compute_global_error(mesh, metric)

    print(f"\nGlobal error: {errors['total']:.6f}")
    print(f"Mean element error: {errors['mean']:.6f}")

    print("\n1D HOMES core implementation complete!")
