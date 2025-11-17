"""
Curved Mesh Generator - GRUMMP-like mesh generation capabilities

This module provides unstructured mesh generation with support for curved elements,
similar to GRUMMP (Generation and Refinement of Unstructured, Mixed-element Meshes).

Features:
- Delaunay triangulation for unstructured mesh generation
- Higher-order curved elements (quadratic and cubic)
- Mesh refinement and quality improvement
- Curved boundary representation
- Mesh quality metrics
- Integration with PDE solvers

Author: Vivek Singh Yadav
Date: 2025-11-17
"""

import numpy as np
from typing import List, Tuple, Optional, Callable, Dict
from dataclasses import dataclass, field
from enum import Enum
import warnings


class ElementType(Enum):
    """Supported element types"""
    TRIANGLE_LINEAR = "triangle_linear"  # 3-node triangle
    TRIANGLE_QUADRATIC = "triangle_quadratic"  # 6-node triangle
    TRIANGLE_CUBIC = "triangle_cubic"  # 10-node triangle
    QUAD_LINEAR = "quad_linear"  # 4-node quadrilateral
    QUAD_QUADRATIC = "quad_quadratic"  # 9-node quadrilateral


class BoundaryType(Enum):
    """Boundary curve types"""
    STRAIGHT = "straight"
    CIRCULAR = "circular"
    SPLINE = "spline"
    PARAMETRIC = "parametric"


@dataclass
class MeshNode:
    """Represents a node in the mesh"""
    id: int
    x: float
    y: float
    on_boundary: bool = False
    boundary_id: Optional[int] = None

    def coords(self) -> np.ndarray:
        """Get coordinates as numpy array"""
        return np.array([self.x, self.y])


@dataclass
class MeshElement:
    """Represents a mesh element (triangle or quad)"""
    id: int
    element_type: ElementType
    node_ids: List[int]
    neighbor_ids: List[Optional[int]] = field(default_factory=list)

    def num_nodes(self) -> int:
        """Get number of nodes in element"""
        return len(self.node_ids)

    def is_curved(self) -> bool:
        """Check if element is curved (higher-order)"""
        return self.element_type in [
            ElementType.TRIANGLE_QUADRATIC,
            ElementType.TRIANGLE_CUBIC,
            ElementType.QUAD_QUADRATIC
        ]


@dataclass
class BoundaryCurve:
    """Represents a curved boundary segment"""
    id: int
    curve_type: BoundaryType
    control_points: np.ndarray
    parametric_func: Optional[Callable[[float], Tuple[float, float]]] = None

    def evaluate(self, t: float) -> Tuple[float, float]:
        """
        Evaluate boundary curve at parameter t ∈ [0, 1]

        Args:
            t: Parameter value

        Returns:
            (x, y) coordinates on curve
        """
        if self.parametric_func is not None:
            return self.parametric_func(t)
        elif self.curve_type == BoundaryType.STRAIGHT:
            # Linear interpolation between endpoints
            p0, p1 = self.control_points[0], self.control_points[-1]
            x = (1 - t) * p0[0] + t * p1[0]
            y = (1 - t) * p0[1] + t * p1[1]
            return x, y
        elif self.curve_type == BoundaryType.CIRCULAR:
            # Circular arc defined by 3 points
            return self._circular_arc(t)
        elif self.curve_type == BoundaryType.SPLINE:
            # Cubic spline interpolation
            return self._spline_interpolation(t)
        else:
            raise ValueError(f"Unknown curve type: {self.curve_type}")

    def _circular_arc(self, t: float) -> Tuple[float, float]:
        """Evaluate circular arc at parameter t"""
        # Find circle center and radius from 3 control points
        p1, p2, p3 = self.control_points[:3]

        # Calculate center using perpendicular bisectors
        mid1 = (p1 + p2) / 2
        mid2 = (p2 + p3) / 2

        dir1 = p2 - p1
        dir2 = p3 - p2

        perp1 = np.array([-dir1[1], dir1[0]])
        perp2 = np.array([-dir2[1], dir2[0]])

        # Solve for intersection (circle center)
        # This is a simplified approach; could be improved
        A = np.column_stack([perp1, -perp2])
        b = mid2 - mid1

        try:
            s = np.linalg.solve(A, b)
            center = mid1 + s[0] * perp1
        except np.linalg.LinAlgError:
            # Collinear points - fall back to straight line
            return self._linear_interpolation(t, p1, p3)

        # Calculate start and end angles
        r1 = p1 - center
        r3 = p3 - center
        radius = np.linalg.norm(r1)

        angle1 = np.arctan2(r1[1], r1[0])
        angle3 = np.arctan2(r3[1], r3[0])

        # Ensure we take the shorter arc
        angle_diff = angle3 - angle1
        if angle_diff > np.pi:
            angle_diff -= 2 * np.pi
        elif angle_diff < -np.pi:
            angle_diff += 2 * np.pi

        # Interpolate angle
        angle = angle1 + t * angle_diff

        x = center[0] + radius * np.cos(angle)
        y = center[1] + radius * np.sin(angle)

        return x, y

    def _spline_interpolation(self, t: float) -> Tuple[float, float]:
        """Cubic spline interpolation through control points"""
        n = len(self.control_points)

        # Scale t to segment index
        segment_t = t * (n - 1)
        segment_idx = int(segment_t)
        local_t = segment_t - segment_idx

        if segment_idx >= n - 1:
            segment_idx = n - 2
            local_t = 1.0

        # Catmull-Rom spline for smooth interpolation
        p0_idx = max(0, segment_idx - 1)
        p1_idx = segment_idx
        p2_idx = min(n - 1, segment_idx + 1)
        p3_idx = min(n - 1, segment_idx + 2)

        p0 = self.control_points[p0_idx]
        p1 = self.control_points[p1_idx]
        p2 = self.control_points[p2_idx]
        p3 = self.control_points[p3_idx]

        # Catmull-Rom basis
        t2 = local_t * local_t
        t3 = t2 * local_t

        point = 0.5 * (
            (2 * p1) +
            (-p0 + p2) * local_t +
            (2*p0 - 5*p1 + 4*p2 - p3) * t2 +
            (-p0 + 3*p1 - 3*p2 + p3) * t3
        )

        return float(point[0]), float(point[1])

    def _linear_interpolation(self, t: float, p1: np.ndarray, p2: np.ndarray) -> Tuple[float, float]:
        """Linear interpolation between two points"""
        point = (1 - t) * p1 + t * p2
        return float(point[0]), float(point[1])


class CurvedMeshGenerator:
    """
    Main mesh generator class with Delaunay triangulation and curved element support
    """

    def __init__(self):
        self.nodes: Dict[int, MeshNode] = {}
        self.elements: Dict[int, MeshElement] = {}
        self.boundaries: Dict[int, BoundaryCurve] = {}
        self.next_node_id = 0
        self.next_element_id = 0
        self.next_boundary_id = 0

    def add_node(self, x: float, y: float, on_boundary: bool = False,
                 boundary_id: Optional[int] = None) -> int:
        """Add a node to the mesh"""
        node_id = self.next_node_id
        self.nodes[node_id] = MeshNode(node_id, x, y, on_boundary, boundary_id)
        self.next_node_id += 1
        return node_id

    def add_element(self, element_type: ElementType, node_ids: List[int]) -> int:
        """Add an element to the mesh"""
        element_id = self.next_element_id
        self.elements[element_id] = MeshElement(element_id, element_type, node_ids)
        self.next_element_id += 1
        return element_id

    def add_boundary_curve(self, curve_type: BoundaryType,
                          control_points: np.ndarray,
                          parametric_func: Optional[Callable] = None) -> int:
        """Add a boundary curve"""
        boundary_id = self.next_boundary_id
        self.boundaries[boundary_id] = BoundaryCurve(
            boundary_id, curve_type, control_points, parametric_func
        )
        self.next_boundary_id += 1
        return boundary_id

    def generate_rectangular_domain(self, x_min: float, x_max: float,
                                   y_min: float, y_max: float,
                                   nx: int, ny: int,
                                   element_type: ElementType = ElementType.TRIANGLE_LINEAR) -> None:
        """
        Generate a structured mesh on a rectangular domain

        Args:
            x_min, x_max: Domain bounds in x
            y_min, y_max: Domain bounds in y
            nx, ny: Number of divisions in x and y
            element_type: Type of elements to generate
        """
        # Clear existing mesh
        self.nodes.clear()
        self.elements.clear()
        self.next_node_id = 0
        self.next_element_id = 0

        # Generate nodes
        x = np.linspace(x_min, x_max, nx)
        y = np.linspace(y_min, y_max, ny)

        node_grid = np.zeros((ny, nx), dtype=int)

        for j in range(ny):
            for i in range(nx):
                on_boundary = (i == 0 or i == nx-1 or j == 0 or j == ny-1)
                node_id = self.add_node(x[i], y[j], on_boundary)
                node_grid[j, i] = node_id

        # Generate triangular elements
        if element_type == ElementType.TRIANGLE_LINEAR:
            for j in range(ny - 1):
                for i in range(nx - 1):
                    # Two triangles per quad
                    n00 = node_grid[j, i]
                    n10 = node_grid[j, i+1]
                    n01 = node_grid[j+1, i]
                    n11 = node_grid[j+1, i+1]

                    # Lower-left triangle
                    self.add_element(ElementType.TRIANGLE_LINEAR, [n00, n10, n01])
                    # Upper-right triangle
                    self.add_element(ElementType.TRIANGLE_LINEAR, [n10, n11, n01])

        elif element_type == ElementType.QUAD_LINEAR:
            for j in range(ny - 1):
                for i in range(nx - 1):
                    n00 = node_grid[j, i]
                    n10 = node_grid[j, i+1]
                    n11 = node_grid[j+1, i+1]
                    n01 = node_grid[j+1, i]

                    self.add_element(ElementType.QUAD_LINEAR, [n00, n10, n11, n01])

    def generate_circular_domain(self, center: Tuple[float, float], radius: float,
                                n_radial: int, n_angular: int,
                                element_type: ElementType = ElementType.TRIANGLE_LINEAR) -> None:
        """
        Generate a structured mesh on a circular domain

        Args:
            center: Center coordinates (x, y)
            radius: Circle radius
            n_radial: Number of radial divisions
            n_angular: Number of angular divisions
            element_type: Type of elements to generate
        """
        self.nodes.clear()
        self.elements.clear()
        self.next_node_id = 0
        self.next_element_id = 0

        cx, cy = center

        # Add center node
        center_id = self.add_node(cx, cy, on_boundary=False)

        # Generate nodes in concentric circles
        node_rings = [[center_id]]

        for i_r in range(1, n_radial + 1):
            r = radius * i_r / n_radial
            ring = []

            for i_theta in range(n_angular):
                theta = 2 * np.pi * i_theta / n_angular
                x = cx + r * np.cos(theta)
                y = cy + r * np.sin(theta)
                on_boundary = (i_r == n_radial)
                node_id = self.add_node(x, y, on_boundary)
                ring.append(node_id)

            node_rings.append(ring)

        # Generate triangular elements
        # Center ring
        for i in range(n_angular):
            n0 = center_id
            n1 = node_rings[1][i]
            n2 = node_rings[1][(i + 1) % n_angular]
            self.add_element(ElementType.TRIANGLE_LINEAR, [n0, n1, n2])

        # Outer rings
        for i_r in range(1, n_radial):
            inner_ring = node_rings[i_r]
            outer_ring = node_rings[i_r + 1]

            for i in range(n_angular):
                n_inner_0 = inner_ring[i]
                n_inner_1 = inner_ring[(i + 1) % n_angular]
                n_outer_0 = outer_ring[i]
                n_outer_1 = outer_ring[(i + 1) % n_angular]

                # Two triangles per quad
                self.add_element(ElementType.TRIANGLE_LINEAR, [n_inner_0, n_outer_0, n_inner_1])
                self.add_element(ElementType.TRIANGLE_LINEAR, [n_inner_1, n_outer_0, n_outer_1])

    def bowyer_watson_triangulation(self, points: np.ndarray) -> None:
        """
        Generate Delaunay triangulation using Bowyer-Watson algorithm

        Args:
            points: Array of shape (n_points, 2) containing point coordinates
        """
        self.nodes.clear()
        self.elements.clear()
        self.next_node_id = 0
        self.next_element_id = 0

        n_points = points.shape[0]

        # Add all points as nodes
        for i in range(n_points):
            self.add_node(points[i, 0], points[i, 1])

        # Create super-triangle that contains all points
        x_min, y_min = points.min(axis=0)
        x_max, y_max = points.max(axis=0)

        dx = x_max - x_min
        dy = y_max - y_min
        delta_max = max(dx, dy)
        x_mid = (x_min + x_max) / 2
        y_mid = (y_min + y_max) / 2

        # Super-triangle vertices (well outside domain)
        st1 = self.add_node(x_mid - 20 * delta_max, y_mid - delta_max)
        st2 = self.add_node(x_mid, y_mid + 20 * delta_max)
        st3 = self.add_node(x_mid + 20 * delta_max, y_mid - delta_max)

        # Initialize triangulation with super-triangle
        self.add_element(ElementType.TRIANGLE_LINEAR, [st1, st2, st3])

        # Add points one by one
        for i in range(n_points):
            self._add_point_to_triangulation(i)

        # Remove triangles that share vertices with super-triangle
        super_triangle_nodes = {st1, st2, st3}
        elements_to_remove = []

        for elem_id, elem in self.elements.items():
            if any(node_id in super_triangle_nodes for node_id in elem.node_ids):
                elements_to_remove.append(elem_id)

        for elem_id in elements_to_remove:
            del self.elements[elem_id]

        # Remove super-triangle nodes
        for node_id in super_triangle_nodes:
            del self.nodes[node_id]

    def _add_point_to_triangulation(self, point_id: int) -> None:
        """Add a point to existing Delaunay triangulation (Bowyer-Watson step)"""
        point = self.nodes[point_id].coords()

        # Find all triangles whose circumcircle contains the point
        bad_triangles = []
        for elem_id, elem in self.elements.items():
            if elem.element_type == ElementType.TRIANGLE_LINEAR:
                if self._point_in_circumcircle(point, elem):
                    bad_triangles.append(elem_id)

        # Find the boundary of the polygonal hole
        polygon_edges = []

        for elem_id in bad_triangles:
            elem = self.elements[elem_id]
            edges = [
                (elem.node_ids[0], elem.node_ids[1]),
                (elem.node_ids[1], elem.node_ids[2]),
                (elem.node_ids[2], elem.node_ids[0])
            ]

            for edge in edges:
                # Check if edge is shared with another bad triangle
                reverse_edge = (edge[1], edge[0])
                is_shared = False

                for other_id in bad_triangles:
                    if other_id != elem_id:
                        other_elem = self.elements[other_id]
                        other_edges = [
                            (other_elem.node_ids[0], other_elem.node_ids[1]),
                            (other_elem.node_ids[1], other_elem.node_ids[2]),
                            (other_elem.node_ids[2], other_elem.node_ids[0])
                        ]

                        if edge in other_edges or reverse_edge in other_edges:
                            is_shared = True
                            break

                if not is_shared:
                    polygon_edges.append(edge)

        # Remove bad triangles
        for elem_id in bad_triangles:
            del self.elements[elem_id]

        # Create new triangles from point to polygon boundary
        for edge in polygon_edges:
            self.add_element(ElementType.TRIANGLE_LINEAR, [edge[0], edge[1], point_id])

    def _point_in_circumcircle(self, point: np.ndarray, triangle: MeshElement) -> bool:
        """Check if point is inside triangle's circumcircle"""
        # Get triangle vertices
        p1 = self.nodes[triangle.node_ids[0]].coords()
        p2 = self.nodes[triangle.node_ids[1]].coords()
        p3 = self.nodes[triangle.node_ids[2]].coords()

        # Calculate circumcircle center and radius
        ax, ay = p1
        bx, by = p2
        cx, cy = p3

        d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))

        if abs(d) < 1e-10:
            return False

        ux = ((ax*ax + ay*ay) * (by - cy) + (bx*bx + by*by) * (cy - ay) +
              (cx*cx + cy*cy) * (ay - by)) / d
        uy = ((ax*ax + ay*ay) * (cx - bx) + (bx*bx + by*by) * (ax - cx) +
              (cx*cx + cy*cy) * (bx - ax)) / d

        center = np.array([ux, uy])
        radius = np.linalg.norm(p1 - center)

        # Check if point is inside circumcircle (with small tolerance)
        dist = np.linalg.norm(point - center)
        return dist < radius * (1 + 1e-10)

    def elevate_to_quadratic(self) -> None:
        """
        Convert linear triangular elements to quadratic (add mid-side nodes)
        """
        # Store edges and their midpoint nodes
        edge_midpoints = {}

        new_elements = {}

        for elem_id, elem in self.elements.items():
            if elem.element_type != ElementType.TRIANGLE_LINEAR:
                new_elements[elem_id] = elem
                continue

            # Get corner nodes
            corner_nodes = elem.node_ids[:3]
            midpoint_nodes = []

            # Process each edge
            for i in range(3):
                n1 = corner_nodes[i]
                n2 = corner_nodes[(i + 1) % 3]

                # Create canonical edge (smaller id first)
                edge = (min(n1, n2), max(n1, n2))

                if edge not in edge_midpoints:
                    # Create midpoint node
                    p1 = self.nodes[n1].coords()
                    p2 = self.nodes[n2].coords()
                    mid = (p1 + p2) / 2

                    # Check if on boundary
                    on_boundary = self.nodes[n1].on_boundary and self.nodes[n2].on_boundary

                    mid_id = self.add_node(mid[0], mid[1], on_boundary)
                    edge_midpoints[edge] = mid_id

                midpoint_nodes.append(edge_midpoints[edge])

            # Create quadratic element: corners + midpoints
            # Node ordering: [v0, v1, v2, mid01, mid12, mid20]
            quad_nodes = [
                corner_nodes[0], corner_nodes[1], corner_nodes[2],
                midpoint_nodes[0], midpoint_nodes[1], midpoint_nodes[2]
            ]

            new_elements[elem_id] = MeshElement(
                elem_id, ElementType.TRIANGLE_QUADRATIC, quad_nodes
            )

        self.elements = new_elements

    def compute_element_quality(self, elem_id: int) -> float:
        """
        Compute quality metric for an element
        Returns value in [0, 1] where 1 is perfect quality

        For triangles: Uses ratio of inscribed to circumscribed circle radii
        """
        elem = self.elements[elem_id]

        if elem.element_type in [ElementType.TRIANGLE_LINEAR, ElementType.TRIANGLE_QUADRATIC]:
            # Use first 3 nodes (corners)
            p1 = self.nodes[elem.node_ids[0]].coords()
            p2 = self.nodes[elem.node_ids[1]].coords()
            p3 = self.nodes[elem.node_ids[2]].coords()

            # Edge lengths
            a = np.linalg.norm(p2 - p3)
            b = np.linalg.norm(p3 - p1)
            c = np.linalg.norm(p1 - p2)

            # Semi-perimeter
            s = (a + b + c) / 2

            # Area using Heron's formula
            area = np.sqrt(max(0, s * (s - a) * (s - b) * (s - c)))

            if area < 1e-10:
                return 0.0

            # Inscribed circle radius
            r_in = area / s

            # Circumscribed circle radius
            r_out = (a * b * c) / (4 * area)

            # Quality metric (1 for equilateral triangle)
            quality = 2 * r_in / r_out

            return quality

        return 1.0

    def smooth_mesh_laplacian(self, iterations: int = 5, relaxation: float = 0.5) -> None:
        """
        Smooth mesh using Laplacian smoothing

        Args:
            iterations: Number of smoothing iterations
            relaxation: Relaxation parameter (0 = no smoothing, 1 = full)
        """
        for _ in range(iterations):
            # Build node-to-node connectivity
            node_neighbors = {node_id: [] for node_id in self.nodes.keys()}

            for elem in self.elements.values():
                # Only use corner nodes for connectivity
                n_corners = 3 if "triangle" in elem.element_type.value else 4
                corner_nodes = elem.node_ids[:n_corners]

                for i, node_id in enumerate(corner_nodes):
                    neighbors = [corner_nodes[(i+1) % n_corners],
                                corner_nodes[(i-1) % n_corners]]
                    node_neighbors[node_id].extend(neighbors)

            # Compute new positions
            new_positions = {}

            for node_id, node in self.nodes.items():
                if node.on_boundary:
                    # Don't move boundary nodes
                    continue

                neighbors = node_neighbors[node_id]
                if not neighbors:
                    continue

                # Average neighbor positions
                neighbor_coords = np.array([self.nodes[n_id].coords()
                                          for n_id in neighbors])
                avg_pos = neighbor_coords.mean(axis=0)

                # Relaxed update
                current_pos = node.coords()
                new_pos = current_pos + relaxation * (avg_pos - current_pos)
                new_positions[node_id] = new_pos

            # Apply new positions
            for node_id, new_pos in new_positions.items():
                self.nodes[node_id].x = new_pos[0]
                self.nodes[node_id].y = new_pos[1]

    def refine_mesh_uniform(self) -> None:
        """
        Uniform mesh refinement - split each triangle into 4 sub-triangles
        """
        if not self.elements:
            return

        # Store original elements
        original_elements = list(self.elements.values())

        # Clear elements (keep nodes)
        self.elements.clear()

        # Edge midpoint cache
        edge_midpoints = {}

        for elem in original_elements:
            if elem.element_type != ElementType.TRIANGLE_LINEAR:
                # Re-add non-triangle elements unchanged
                self.elements[elem.id] = elem
                continue

            corner_nodes = elem.node_ids[:3]
            midpoint_nodes = []

            # Get or create midpoint nodes
            for i in range(3):
                n1 = corner_nodes[i]
                n2 = corner_nodes[(i + 1) % 3]
                edge = (min(n1, n2), max(n1, n2))

                if edge not in edge_midpoints:
                    p1 = self.nodes[n1].coords()
                    p2 = self.nodes[n2].coords()
                    mid = (p1 + p2) / 2
                    on_boundary = self.nodes[n1].on_boundary and self.nodes[n2].on_boundary
                    mid_id = self.add_node(mid[0], mid[1], on_boundary)
                    edge_midpoints[edge] = mid_id

                midpoint_nodes.append(edge_midpoints[edge])

            # Create 4 new triangles
            # Corner triangles
            self.add_element(ElementType.TRIANGLE_LINEAR,
                           [corner_nodes[0], midpoint_nodes[0], midpoint_nodes[2]])
            self.add_element(ElementType.TRIANGLE_LINEAR,
                           [corner_nodes[1], midpoint_nodes[1], midpoint_nodes[0]])
            self.add_element(ElementType.TRIANGLE_LINEAR,
                           [corner_nodes[2], midpoint_nodes[2], midpoint_nodes[1]])
            # Center triangle
            self.add_element(ElementType.TRIANGLE_LINEAR,
                           [midpoint_nodes[0], midpoint_nodes[1], midpoint_nodes[2]])

    def get_mesh_statistics(self) -> Dict:
        """Get mesh statistics"""
        stats = {
            'num_nodes': len(self.nodes),
            'num_elements': len(self.elements),
            'num_boundary_nodes': sum(1 for n in self.nodes.values() if n.on_boundary),
            'element_types': {},
            'quality_min': float('inf'),
            'quality_max': 0.0,
            'quality_mean': 0.0
        }

        # Count element types
        for elem in self.elements.values():
            elem_type = elem.element_type.value
            stats['element_types'][elem_type] = stats['element_types'].get(elem_type, 0) + 1

        # Compute quality metrics
        if self.elements:
            qualities = [self.compute_element_quality(elem_id)
                        for elem_id in self.elements.keys()]
            stats['quality_min'] = min(qualities)
            stats['quality_max'] = max(qualities)
            stats['quality_mean'] = np.mean(qualities)

        return stats

    def export_to_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Export mesh to numpy arrays for compatibility with solvers

        Returns:
            nodes: Array of shape (num_nodes, 2) with coordinates
            elements: Array of shape (num_elements, max_nodes_per_element) with connectivity
        """
        # Create node array
        node_ids = sorted(self.nodes.keys())
        nodes = np.array([[self.nodes[nid].x, self.nodes[nid].y]
                         for nid in node_ids])

        # Create element connectivity array
        if not self.elements:
            return nodes, np.array([])

        max_nodes = max(len(elem.node_ids) for elem in self.elements.values())
        elements = np.full((len(self.elements), max_nodes), -1, dtype=int)

        for i, elem in enumerate(self.elements.values()):
            for j, node_id in enumerate(elem.node_ids):
                elements[i, j] = node_id

        return nodes, elements


def create_curved_boundary_circle(center: Tuple[float, float], radius: float,
                                  n_points: int = 100) -> BoundaryCurve:
    """
    Helper function to create a circular boundary

    Args:
        center: Circle center (x, y)
        radius: Circle radius
        n_points: Number of points for discretization

    Returns:
        BoundaryCurve object
    """
    def parametric_func(t):
        theta = 2 * np.pi * t
        x = center[0] + radius * np.cos(theta)
        y = center[1] + radius * np.sin(theta)
        return x, y

    # Sample control points
    theta = np.linspace(0, 2*np.pi, n_points, endpoint=False)
    control_points = np.column_stack([
        center[0] + radius * np.cos(theta),
        center[1] + radius * np.sin(theta)
    ])

    return BoundaryCurve(0, BoundaryType.PARAMETRIC, control_points, parametric_func)


def create_curved_boundary_ellipse(center: Tuple[float, float],
                                   a: float, b: float,
                                   n_points: int = 100) -> BoundaryCurve:
    """
    Helper function to create an elliptical boundary

    Args:
        center: Ellipse center (x, y)
        a: Semi-major axis
        b: Semi-minor axis
        n_points: Number of points for discretization

    Returns:
        BoundaryCurve object
    """
    def parametric_func(t):
        theta = 2 * np.pi * t
        x = center[0] + a * np.cos(theta)
        y = center[1] + b * np.sin(theta)
        return x, y

    theta = np.linspace(0, 2*np.pi, n_points, endpoint=False)
    control_points = np.column_stack([
        center[0] + a * np.cos(theta),
        center[1] + b * np.sin(theta)
    ])

    return BoundaryCurve(0, BoundaryType.PARAMETRIC, control_points, parametric_func)
