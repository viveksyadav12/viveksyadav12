"""
Mesh Visualization Module

Provides visualization tools for curved and unstructured meshes generated
by the CurvedMeshGenerator.

Features:
- Plot mesh nodes and elements
- Visualize curved elements
- Show mesh quality metrics
- Display boundary curves
- Adaptive coloring based on quality

Author: Vivek Singh Yadav
Date: 2025-11-17
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle
from matplotlib.collections import PatchCollection, LineCollection
from typing import Optional, Tuple, List
import warnings

from curved_mesh_generator import (
    CurvedMeshGenerator, ElementType, MeshElement, BoundaryCurve
)


class MeshVisualizer:
    """Visualization tools for curved meshes"""

    def __init__(self, mesh_generator: CurvedMeshGenerator):
        """
        Initialize visualizer

        Args:
            mesh_generator: CurvedMeshGenerator instance to visualize
        """
        self.mesh = mesh_generator

    def plot_mesh(self, show_nodes: bool = True, show_node_ids: bool = False,
                  show_element_ids: bool = False, show_boundary: bool = True,
                  color_by_quality: bool = False, ax: Optional[plt.Axes] = None,
                  figsize: Tuple[float, float] = (10, 10)) -> plt.Figure:
        """
        Plot the mesh

        Args:
            show_nodes: Whether to show node markers
            show_node_ids: Whether to label nodes with IDs
            show_element_ids: Whether to label elements with IDs
            show_boundary: Whether to highlight boundary nodes
            color_by_quality: Color elements by quality metric
            ax: Matplotlib axes (creates new if None)
            figsize: Figure size if creating new figure

        Returns:
            Matplotlib figure
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()

        # Collect element patches
        patches = []
        qualities = []

        for elem_id, elem in self.mesh.elements.items():
            # Get corner node coordinates
            if elem.element_type in [ElementType.TRIANGLE_LINEAR,
                                     ElementType.TRIANGLE_QUADRATIC,
                                     ElementType.TRIANGLE_CUBIC]:
                corner_coords = np.array([
                    self.mesh.nodes[elem.node_ids[i]].coords()
                    for i in range(3)
                ])
            elif elem.element_type in [ElementType.QUAD_LINEAR,
                                       ElementType.QUAD_QUADRATIC]:
                n_corners = 4
                corner_coords = np.array([
                    self.mesh.nodes[elem.node_ids[i]].coords()
                    for i in range(n_corners)
                ])
            else:
                continue

            # Create polygon patch
            polygon = Polygon(corner_coords, closed=True, fill=True, edgecolor='black',
                            facecolor='lightblue', linewidth=0.5, alpha=0.6)
            patches.append(polygon)

            if color_by_quality:
                quality = self.mesh.compute_element_quality(elem_id)
                qualities.append(quality)

        # Create patch collection
        if color_by_quality and qualities:
            collection = PatchCollection(patches, cmap='RdYlGn', alpha=0.6,
                                        edgecolors='black', linewidths=0.5)
            collection.set_array(np.array(qualities))
            collection.set_clim(0, 1)
            ax.add_collection(collection)
            cbar = plt.colorbar(collection, ax=ax)
            cbar.set_label('Element Quality', rotation=270, labelpad=20)
        else:
            for patch in patches:
                ax.add_patch(patch)

        # Plot nodes
        if show_nodes or show_boundary:
            regular_nodes = []
            boundary_nodes = []

            for node_id, node in self.mesh.nodes.items():
                if node.on_boundary:
                    boundary_nodes.append(node.coords())
                else:
                    regular_nodes.append(node.coords())

            if regular_nodes and show_nodes:
                regular_nodes = np.array(regular_nodes)
                ax.plot(regular_nodes[:, 0], regular_nodes[:, 1], 'ko',
                       markersize=3, label='Interior nodes')

            if boundary_nodes and show_boundary:
                boundary_nodes = np.array(boundary_nodes)
                ax.plot(boundary_nodes[:, 0], boundary_nodes[:, 1], 'ro',
                       markersize=5, label='Boundary nodes')

        # Show node IDs
        if show_node_ids:
            for node_id, node in self.mesh.nodes.items():
                ax.text(node.x, node.y, str(node_id), fontsize=6,
                       ha='center', va='bottom', color='blue')

        # Show element IDs
        if show_element_ids:
            for elem_id, elem in self.mesh.elements.items():
                # Compute element centroid
                coords = np.array([self.mesh.nodes[nid].coords()
                                 for nid in elem.node_ids[:3]])
                centroid = coords.mean(axis=0)
                ax.text(centroid[0], centroid[1], str(elem_id), fontsize=8,
                       ha='center', va='center', color='red',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

        ax.set_aspect('equal')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('Mesh Visualization')
        ax.grid(True, alpha=0.3)

        if show_nodes and show_boundary:
            ax.legend()

        plt.tight_layout()
        return fig

    def plot_curved_elements(self, n_subdivisions: int = 10,
                           ax: Optional[plt.Axes] = None,
                           figsize: Tuple[float, float] = (10, 10)) -> plt.Figure:
        """
        Plot curved (higher-order) elements with smooth curves

        Args:
            n_subdivisions: Number of subdivisions for plotting curves
            ax: Matplotlib axes
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()

        for elem_id, elem in self.mesh.elements.items():
            if elem.element_type == ElementType.TRIANGLE_QUADRATIC:
                self._plot_quadratic_triangle(elem, n_subdivisions, ax)
            elif elem.element_type == ElementType.TRIANGLE_LINEAR:
                # Plot as regular polygon
                coords = np.array([self.mesh.nodes[nid].coords()
                                 for nid in elem.node_ids])
                polygon = Polygon(coords, closed=True, fill=True,
                                edgecolor='black', facecolor='lightblue',
                                linewidth=0.5, alpha=0.6)
                ax.add_patch(polygon)

        # Plot all nodes
        all_nodes = np.array([[node.x, node.y]
                             for node in self.mesh.nodes.values()])
        ax.plot(all_nodes[:, 0], all_nodes[:, 1], 'ko', markersize=3)

        # Highlight mid-side nodes for curved elements
        midside_nodes = []
        for elem in self.mesh.elements.values():
            if elem.element_type == ElementType.TRIANGLE_QUADRATIC:
                # Nodes 3, 4, 5 are mid-side nodes
                for nid in elem.node_ids[3:6]:
                    node = self.mesh.nodes[nid]
                    midside_nodes.append([node.x, node.y])

        if midside_nodes:
            midside_nodes = np.array(midside_nodes)
            ax.plot(midside_nodes[:, 0], midside_nodes[:, 1], 'rs',
                   markersize=5, label='Mid-side nodes')
            ax.legend()

        ax.set_aspect('equal')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('Curved Elements Visualization')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        return fig

    def _plot_quadratic_triangle(self, elem: MeshElement, n_subdivisions: int,
                                ax: plt.Axes) -> None:
        """Plot a single quadratic triangle element with curved edges"""
        # Node ordering: [v0, v1, v2, mid01, mid12, mid20]
        nodes = [self.mesh.nodes[nid].coords() for nid in elem.node_ids]

        v0, v1, v2 = nodes[0], nodes[1], nodes[2]
        m01, m12, m20 = nodes[3], nodes[4], nodes[5]

        # Plot three curved edges using quadratic interpolation
        edges = [
            (v0, m01, v1),  # Edge 0-1
            (v1, m12, v2),  # Edge 1-2
            (v2, m20, v0),  # Edge 2-0
        ]

        edge_lines = []

        for p0, p_mid, p1 in edges:
            # Quadratic Bezier interpolation
            t = np.linspace(0, 1, n_subdivisions)
            curve = np.zeros((n_subdivisions, 2))

            for i, ti in enumerate(t):
                # Quadratic Bezier formula
                curve[i] = (1-ti)**2 * p0 + 2*(1-ti)*ti * p_mid + ti**2 * p1

            edge_lines.append(curve)

        # Plot edges
        for curve in edge_lines:
            ax.plot(curve[:, 0], curve[:, 1], 'b-', linewidth=1)

        # Fill element (approximate with polygon)
        all_points = np.vstack(edge_lines)
        polygon = Polygon(all_points, closed=True, fill=True,
                         facecolor='lightblue', alpha=0.4)
        ax.add_patch(polygon)

    def plot_quality_histogram(self, bins: int = 20,
                              figsize: Tuple[float, float] = (10, 6)) -> plt.Figure:
        """
        Plot histogram of element quality metrics

        Args:
            bins: Number of histogram bins
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)

        qualities = [self.mesh.compute_element_quality(elem_id)
                    for elem_id in self.mesh.elements.keys()]

        ax.hist(qualities, bins=bins, edgecolor='black', alpha=0.7, color='skyblue')
        ax.axvline(np.mean(qualities), color='red', linestyle='--',
                  linewidth=2, label=f'Mean: {np.mean(qualities):.3f}')
        ax.axvline(np.median(qualities), color='green', linestyle='--',
                  linewidth=2, label=f'Median: {np.median(qualities):.3f}')

        ax.set_xlabel('Element Quality')
        ax.set_ylabel('Count')
        ax.set_title('Element Quality Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add statistics text
        stats_text = f'Min: {np.min(qualities):.3f}\n'
        stats_text += f'Max: {np.max(qualities):.3f}\n'
        stats_text += f'Std: {np.std(qualities):.3f}'

        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
               verticalalignment='top', bbox=dict(boxstyle='round',
               facecolor='wheat', alpha=0.5), fontsize=10)

        plt.tight_layout()
        return fig

    def plot_boundary_curves(self, n_points: int = 100,
                           ax: Optional[plt.Axes] = None,
                           figsize: Tuple[float, float] = (10, 10)) -> plt.Figure:
        """
        Plot boundary curves

        Args:
            n_points: Number of points for curve discretization
            ax: Matplotlib axes
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()

        for boundary_id, boundary in self.mesh.boundaries.items():
            t = np.linspace(0, 1, n_points)
            curve_points = np.array([boundary.evaluate(ti) for ti in t])

            ax.plot(curve_points[:, 0], curve_points[:, 1], 'r-',
                   linewidth=2, label=f'Boundary {boundary_id}')

            # Plot control points
            ax.plot(boundary.control_points[:, 0], boundary.control_points[:, 1],
                   'bo', markersize=6, label=f'Control points {boundary_id}')

        ax.set_aspect('equal')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('Boundary Curves')
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()

        return fig

    def plot_mesh_comparison(self, other_mesh: CurvedMeshGenerator,
                           figsize: Tuple[float, float] = (16, 7)) -> plt.Figure:
        """
        Compare two meshes side-by-side

        Args:
            other_mesh: Another mesh generator to compare
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Plot first mesh
        self.plot_mesh(ax=ax1, show_nodes=True, show_boundary=True,
                      color_by_quality=True)
        ax1.set_title('Original Mesh')

        # Plot second mesh
        other_viz = MeshVisualizer(other_mesh)
        other_viz.plot_mesh(ax=ax2, show_nodes=True, show_boundary=True,
                          color_by_quality=True)
        ax2.set_title('Refined Mesh')

        # Add statistics
        stats1 = self.mesh.get_mesh_statistics()
        stats2 = other_mesh.get_mesh_statistics()

        info_text = f"Mesh 1: {stats1['num_nodes']} nodes, {stats1['num_elements']} elements\n"
        info_text += f"Mesh 2: {stats2['num_nodes']} nodes, {stats2['num_elements']} elements"

        fig.text(0.5, 0.02, info_text, ha='center', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        return fig

    def plot_node_connectivity(self, figsize: Tuple[float, float] = (10, 10)) -> plt.Figure:
        """
        Visualize node connectivity graph

        Args:
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Build connectivity
        edges = set()

        for elem in self.mesh.elements.values():
            # Only use corner nodes
            n_corners = 3 if "triangle" in elem.element_type.value else 4
            corner_nodes = elem.node_ids[:n_corners]

            for i in range(n_corners):
                n1 = corner_nodes[i]
                n2 = corner_nodes[(i + 1) % n_corners]
                edge = (min(n1, n2), max(n1, n2))
                edges.add(edge)

        # Plot edges
        for n1, n2 in edges:
            p1 = self.mesh.nodes[n1].coords()
            p2 = self.mesh.nodes[n2].coords()
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'b-',
                   linewidth=0.5, alpha=0.5)

        # Plot nodes
        all_nodes = np.array([[node.x, node.y]
                             for node in self.mesh.nodes.values()])
        ax.plot(all_nodes[:, 0], all_nodes[:, 1], 'ro', markersize=4)

        ax.set_aspect('equal')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('Node Connectivity Graph')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        return fig

    def save_mesh_figure(self, filename: str, **kwargs) -> None:
        """
        Save mesh visualization to file

        Args:
            filename: Output filename
            **kwargs: Additional arguments passed to plot_mesh
        """
        fig = self.plot_mesh(**kwargs)
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Mesh visualization saved to {filename}")


def plot_refinement_progression(meshes: List[CurvedMeshGenerator],
                                titles: Optional[List[str]] = None,
                                figsize: Tuple[float, float] = (16, 12)) -> plt.Figure:
    """
    Plot multiple meshes showing refinement progression

    Args:
        meshes: List of mesh generators at different refinement levels
        titles: List of titles for each subplot
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    n_meshes = len(meshes)
    n_cols = min(3, n_meshes)
    n_rows = (n_meshes + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1 or n_cols == 1:
        axes = axes.reshape(n_rows, n_cols)

    for i, mesh in enumerate(meshes):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]

        viz = MeshVisualizer(mesh)
        viz.plot_mesh(ax=ax, show_nodes=True, show_boundary=True,
                     color_by_quality=True)

        if titles and i < len(titles):
            ax.set_title(titles[i])
        else:
            stats = mesh.get_mesh_statistics()
            ax.set_title(f"Level {i}: {stats['num_nodes']} nodes, "
                        f"{stats['num_elements']} elements")

    # Hide unused subplots
    for i in range(n_meshes, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].axis('off')

    plt.tight_layout()
    return fig
