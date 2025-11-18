"""
Heart Mesh Generator for Cardiac FEM Simulations

This module provides anatomically-inspired heart mesh generation for cardiac
electrophysiology and mechanics simulations.

Features:
- Realistic heart geometry with multiple chambers
- Left and right ventricles
- Adjustable mesh density
- Region markers for different cardiac tissues
- Fiber orientation fields for anisotropic conduction
- Integration with FEM solvers

Author: Vivek Singh Yadav
Date: 2025-11-18
"""

import numpy as np
from typing import Tuple, Dict, Optional, List, Callable
from dataclasses import dataclass
import warnings

try:
    from curved_mesh_generator import CurvedMeshGenerator, ElementType
except ImportError:
    warnings.warn("curved_mesh_generator not available, using standalone mode")


@dataclass
class HeartRegion:
    """Represents a region of the heart"""
    name: str
    element_ids: List[int]
    conductivity: float  # Electrical conductivity (S/m)
    fiber_direction: np.ndarray  # Primary fiber orientation


class HeartGeometry:
    """
    Generates anatomically-inspired heart geometry
    """

    def __init__(self, scale: float = 1.0):
        """
        Initialize heart geometry generator

        Args:
            scale: Overall size scale factor
        """
        self.scale = scale

    def left_ventricle_boundary(self, theta: float) -> Tuple[float, float]:
        """
        Parametric function for left ventricle outer boundary

        Uses a truncated ellipsoid with rounded apex

        Args:
            theta: Parameter from 0 to 2π

        Returns:
            (x, y) coordinates
        """
        # Elliptical shape with apex modification
        a = 3.0 * self.scale  # Major axis
        b = 2.0 * self.scale  # Minor axis

        # Map theta to shape
        if theta <= np.pi:  # Upper half (base to apex)
            # Elliptical
            t = theta
            x = a * np.cos(t)
            y = b * np.sin(t)
        else:  # Lower half (apex region)
            # Narrower, pointed apex
            t = theta - np.pi
            x = a * 0.3 * np.cos(t + np.pi)
            y = -b * 0.8 + b * 0.2 * (1 - np.cos(t))

        return x, y

    def left_ventricle_inner_boundary(self, theta: float) -> Tuple[float, float]:
        """
        Parametric function for left ventricle cavity (inner boundary)

        Args:
            theta: Parameter from 0 to 2π

        Returns:
            (x, y) coordinates
        """
        # Smaller ellipse for cavity
        scale_factor = 0.6
        x_outer, y_outer = self.left_ventricle_boundary(theta)
        return x_outer * scale_factor, y_outer * scale_factor

    def right_ventricle_boundary(self, theta: float) -> Tuple[float, float]:
        """
        Parametric function for right ventricle (crescent-shaped)

        Args:
            theta: Parameter from 0 to 2π

        Returns:
            (x, y) coordinates
        """
        # Crescent wrapping around left ventricle
        a = 2.5 * self.scale
        b = 1.8 * self.scale

        # Offset to wrap around LV
        offset_x = 2.0 * self.scale

        if theta <= np.pi:
            t = theta
            x = offset_x + a * 0.7 * np.cos(t)
            y = b * np.sin(t)
        else:
            t = theta - np.pi
            x = offset_x + a * 0.4 * np.cos(t + np.pi)
            y = -b * 0.7 + b * 0.3 * (1 - np.cos(t))

        return x, y

    def complete_heart_boundary(self, theta: float) -> Tuple[float, float]:
        """
        Combined boundary for simplified biventricular geometry

        Args:
            theta: Parameter from 0 to 2π

        Returns:
            (x, y) coordinates
        """
        # Heart-like shape using cardioid-based curve
        a = 2.5 * self.scale

        # Modified cardioid for realistic heart shape
        x = a * (2 * np.cos(theta) - np.cos(2 * theta))
        y = a * (2 * np.sin(theta) - np.sin(2 * theta))

        # Rotate and shift to standard orientation
        angle = -np.pi / 4
        x_rot = x * np.cos(angle) - y * np.sin(angle)
        y_rot = x * np.sin(angle) + y * np.cos(angle)

        return x_rot, y_rot + 2.0 * self.scale


class HeartMeshGenerator:
    """
    Main class for generating heart meshes for FEM simulations
    """

    def __init__(self, geometry: Optional[HeartGeometry] = None):
        """
        Initialize heart mesh generator

        Args:
            geometry: HeartGeometry object (creates default if None)
        """
        self.geometry = geometry if geometry is not None else HeartGeometry()
        self.mesh_generator = CurvedMeshGenerator()
        self.regions: Dict[str, HeartRegion] = {}
        self.fiber_field: Optional[np.ndarray] = None

    def generate_left_ventricle_mesh(self, n_radial: int = 15, n_angular: int = 40,
                                     wall_thickness: float = 0.4) -> None:
        """
        Generate mesh for left ventricle with myocardial wall

        Args:
            n_radial: Number of radial divisions through wall thickness
            n_angular: Number of angular divisions around circumference
            wall_thickness: Relative wall thickness (0-1)
        """
        # Clear existing mesh
        self.mesh_generator.nodes.clear()
        self.mesh_generator.elements.clear()
        self.mesh_generator.next_node_id = 0
        self.mesh_generator.next_element_id = 0

        # Sample boundary curves
        theta_samples = np.linspace(0, 2*np.pi, n_angular, endpoint=False)

        # Create inner and outer boundaries
        outer_points = np.array([self.geometry.left_ventricle_boundary(t)
                                for t in theta_samples])
        inner_points = np.array([self.geometry.left_ventricle_inner_boundary(t)
                                for t in theta_samples])

        # Generate nodes in radial layers
        node_grid = np.zeros((n_radial + 1, n_angular), dtype=int)

        for i_r in range(n_radial + 1):
            # Interpolate between inner and outer boundaries
            alpha = i_r / n_radial

            for i_theta in range(n_angular):
                x = (1 - alpha) * inner_points[i_theta, 0] + alpha * outer_points[i_theta, 0]
                y = (1 - alpha) * inner_points[i_theta, 1] + alpha * outer_points[i_theta, 1]

                on_boundary = (i_r == 0) or (i_r == n_radial)
                node_id = self.mesh_generator.add_node(x, y, on_boundary)
                node_grid[i_r, i_theta] = node_id

        # Generate triangular elements
        myocardium_elements = []

        for i_r in range(n_radial):
            for i_theta in range(n_angular):
                i_theta_next = (i_theta + 1) % n_angular

                # Inner layer nodes
                n_inner_0 = node_grid[i_r, i_theta]
                n_inner_1 = node_grid[i_r, i_theta_next]

                # Outer layer nodes
                n_outer_0 = node_grid[i_r + 1, i_theta]
                n_outer_1 = node_grid[i_r + 1, i_theta_next]

                # Two triangles per quadrilateral
                elem1 = self.mesh_generator.add_element(
                    ElementType.TRIANGLE_LINEAR,
                    [n_inner_0, n_outer_0, n_inner_1]
                )
                elem2 = self.mesh_generator.add_element(
                    ElementType.TRIANGLE_LINEAR,
                    [n_inner_1, n_outer_0, n_outer_1]
                )

                myocardium_elements.extend([elem1, elem2])

        # Store region information
        self.regions['myocardium'] = HeartRegion(
            name='left_ventricle_myocardium',
            element_ids=myocardium_elements,
            conductivity=0.3,  # S/m (typical myocardium)
            fiber_direction=np.array([1.0, 0.0])  # Will be computed later
        )

    def generate_biventricular_mesh(self, n_points_lv: int = 50,
                                   n_points_rv: int = 40,
                                   n_septum: int = 30) -> None:
        """
        Generate simplified biventricular mesh

        Args:
            n_points_lv: Number of points for LV boundary
            n_points_rv: Number of points for RV boundary
            n_septum: Number of points for septum (shared wall)
        """
        # Sample complete heart boundary
        n_boundary = 60
        theta = np.linspace(0, 2*np.pi, n_boundary, endpoint=False)
        boundary_points = np.array([self.geometry.complete_heart_boundary(t)
                                   for t in theta])

        # Generate interior points for better triangulation
        # Use a grid-based approach with random perturbation
        x_min, x_max = boundary_points[:, 0].min(), boundary_points[:, 0].max()
        y_min, y_max = boundary_points[:, 1].min(), boundary_points[:, 1].max()

        n_interior = 150
        interior_points = []

        # Random sampling with rejection
        while len(interior_points) < n_interior:
            x = np.random.uniform(x_min, x_max)
            y = np.random.uniform(y_min, y_max)

            # Check if point is inside boundary (simplified)
            if self._point_in_heart(x, y):
                interior_points.append([x, y])

        interior_points = np.array(interior_points)

        # Combine boundary and interior points
        all_points = np.vstack([boundary_points, interior_points])

        # Generate Delaunay triangulation
        self.mesh_generator.bowyer_watson_triangulation(all_points)

        # Smooth the mesh
        self.mesh_generator.smooth_mesh_laplacian(iterations=10, relaxation=0.5)

        # Identify regions (simplified: all myocardium)
        all_elements = list(self.mesh_generator.elements.keys())
        self.regions['myocardium'] = HeartRegion(
            name='biventricular_myocardium',
            element_ids=all_elements,
            conductivity=0.3,
            fiber_direction=np.array([1.0, 0.0])
        )

    def _point_in_heart(self, x: float, y: float, margin: float = 0.1) -> bool:
        """
        Check if point is inside heart boundary (simplified)

        Args:
            x, y: Point coordinates
            margin: Safety margin from boundary

        Returns:
            True if inside heart
        """
        # Sample boundary and check distance
        n_samples = 100
        theta = np.linspace(0, 2*np.pi, n_samples)
        boundary = np.array([self.geometry.complete_heart_boundary(t) for t in theta])

        # Compute winding number (simple point-in-polygon test)
        point = np.array([x, y])
        winding = 0

        for i in range(n_samples):
            p1 = boundary[i]
            p2 = boundary[(i + 1) % n_samples]

            if p1[1] <= y:
                if p2[1] > y:
                    if self._is_left(p1, p2, point) > 0:
                        winding += 1
            else:
                if p2[1] <= y:
                    if self._is_left(p1, p2, point) < 0:
                        winding -= 1

        return winding != 0

    def _is_left(self, p0: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> float:
        """Test if point p2 is left/on/right of line p0-p1"""
        return ((p1[0] - p0[0]) * (p2[1] - p0[1]) -
                (p2[0] - p0[0]) * (p1[1] - p0[1]))

    def compute_fiber_orientations(self, fiber_angle_endo: float = -60.0,
                                  fiber_angle_epi: float = 60.0) -> None:
        """
        Compute cardiac fiber orientations through myocardial wall

        Fibers rotate from endocardium to epicardium (typically -60° to +60°)

        Args:
            fiber_angle_endo: Fiber angle at endocardium (degrees)
            fiber_angle_epi: Fiber angle at epicardium (degrees)
        """
        n_nodes = len(self.mesh_generator.nodes)
        self.fiber_field = np.zeros((n_nodes, 2))

        # For each node, compute fiber direction based on position
        for node_id, node in self.mesh_generator.nodes.items():
            # Compute radial position (distance from center)
            center = np.array([0.0, 2.0 * self.geometry.scale])
            pos = np.array([node.x, node.y])
            radial_vec = pos - center
            radius = np.linalg.norm(radial_vec)

            # Normalize radial vector
            if radius > 1e-10:
                radial_unit = radial_vec / radius
            else:
                radial_unit = np.array([1.0, 0.0])

            # Circumferential direction (perpendicular to radial)
            circum_unit = np.array([-radial_unit[1], radial_unit[0]])

            # Determine transmural position (0 = endo, 1 = epi)
            # Simplified: use y-coordinate
            y_min = min(n.y for n in self.mesh_generator.nodes.values())
            y_max = max(n.y for n in self.mesh_generator.nodes.values())
            transmural = (node.y - y_min) / (y_max - y_min + 1e-10)

            # Linearly interpolate fiber angle
            angle_deg = fiber_angle_endo + transmural * (fiber_angle_epi - fiber_angle_endo)
            angle_rad = np.radians(angle_deg)

            # Fiber is rotated from circumferential direction
            fiber = np.cos(angle_rad) * circum_unit + np.sin(angle_rad) * radial_unit
            fiber = fiber / (np.linalg.norm(fiber) + 1e-10)

            self.fiber_field[node_id] = fiber

    def refine_apex_region(self, apex_center: Tuple[float, float],
                          apex_radius: float, refinement_level: int = 1) -> None:
        """
        Refine mesh in apex region for better resolution

        Args:
            apex_center: Center of apex region
            apex_radius: Radius of refinement zone
            refinement_level: Number of refinement passes
        """
        for _ in range(refinement_level):
            # Identify elements in apex region
            elements_to_refine = []

            for elem_id, elem in self.mesh_generator.elements.items():
                # Check if element centroid is in apex region
                nodes = [self.mesh_generator.nodes[nid] for nid in elem.node_ids[:3]]
                centroid = np.mean([[n.x, n.y] for n in nodes], axis=0)
                dist = np.linalg.norm(centroid - np.array(apex_center))

                if dist < apex_radius:
                    elements_to_refine.append(elem_id)

            # Refine selected elements
            # This is simplified - would need selective refinement
            if elements_to_refine:
                self.mesh_generator.refine_mesh_uniform()
                break

    def export_for_fem(self) -> Dict:
        """
        Export mesh data for FEM simulation

        Returns:
            Dictionary containing:
                - nodes: Node coordinates (N x 2)
                - elements: Element connectivity (M x 3)
                - regions: Region information
                - fibers: Fiber orientations (N x 2)
        """
        nodes, elements = self.mesh_generator.export_to_arrays()

        return {
            'nodes': nodes,
            'elements': elements,
            'regions': self.regions,
            'fibers': self.fiber_field,
            'mesh_generator': self.mesh_generator
        }

    def get_mesh_statistics(self) -> Dict:
        """Get comprehensive mesh statistics"""
        stats = self.mesh_generator.get_mesh_statistics()

        # Add heart-specific statistics
        stats['num_regions'] = len(self.regions)
        stats['has_fiber_field'] = self.fiber_field is not None

        for region_name, region in self.regions.items():
            stats[f'{region_name}_elements'] = len(region.element_ids)

        return stats


def create_simple_lv_mesh(n_radial: int = 10, n_angular: int = 30) -> HeartMeshGenerator:
    """
    Helper function to create a simple left ventricle mesh

    Args:
        n_radial: Radial divisions through wall
        n_angular: Angular divisions around circumference

    Returns:
        HeartMeshGenerator with LV mesh
    """
    geometry = HeartGeometry(scale=1.0)
    heart_mesh = HeartMeshGenerator(geometry)
    heart_mesh.generate_left_ventricle_mesh(n_radial, n_angular)
    heart_mesh.compute_fiber_orientations()

    return heart_mesh


def create_biventricular_mesh(n_boundary: int = 80, n_interior: int = 200) -> HeartMeshGenerator:
    """
    Helper function to create a biventricular heart mesh

    Args:
        n_boundary: Number of boundary points
        n_interior: Number of interior points

    Returns:
        HeartMeshGenerator with biventricular mesh
    """
    geometry = HeartGeometry(scale=1.0)
    heart_mesh = HeartMeshGenerator(geometry)
    heart_mesh.generate_biventricular_mesh(n_boundary, n_interior)
    heart_mesh.compute_fiber_orientations()

    return heart_mesh


if __name__ == "__main__":
    # Quick test
    print("Testing Heart Mesh Generator...")

    print("\n1. Creating simple LV mesh...")
    lv_mesh = create_simple_lv_mesh(n_radial=8, n_angular=24)
    stats = lv_mesh.get_mesh_statistics()
    print(f"   Nodes: {stats['num_nodes']}")
    print(f"   Elements: {stats['num_elements']}")
    print(f"   Quality: {stats['quality_mean']:.3f}")

    print("\n2. Creating biventricular mesh...")
    bv_mesh = create_biventricular_mesh()
    stats = bv_mesh.get_mesh_statistics()
    print(f"   Nodes: {stats['num_nodes']}")
    print(f"   Elements: {stats['num_elements']}")
    print(f"   Quality: {stats['quality_mean']:.3f}")

    print("\nHeart mesh generator ready!")
