"""
Visualization utilities for reaction-diffusion systems.

Author: Vivek Singh Yadav
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib import cm
from typing import List, Optional, Tuple
import os


def plot_snapshot(u: np.ndarray,
                 v: Optional[np.ndarray] = None,
                 time: float = 0.0,
                 title: str = "Reaction-Diffusion System",
                 save_path: Optional[str] = None,
                 cmap: str = 'viridis'):
    """
    Plot a snapshot of the concentration fields.

    Args:
        u: First concentration field
        v: Second concentration field (optional)
        time: Current time
        title: Plot title
        save_path: Path to save the figure (optional)
        cmap: Colormap to use
    """
    if v is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        im1 = ax1.imshow(u, cmap=cmap, interpolation='bilinear', origin='lower')
        ax1.set_title(f'Species u at t={time:.2f}')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

        im2 = ax2.imshow(v, cmap=cmap, interpolation='bilinear', origin='lower')
        ax2.set_title(f'Species v at t={time:.2f}')
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(8, 6))

        im1 = ax1.imshow(u, cmap=cmap, interpolation='bilinear', origin='lower')
        ax1.set_title(f'{title} at t={time:.2f}')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved snapshot to {save_path}")

    plt.show()


def plot_evolution(u_history: List[np.ndarray],
                  v_history: List[np.ndarray],
                  times: List[float],
                  n_snapshots: int = 6,
                  title: str = "Evolution of Reaction-Diffusion System",
                  save_path: Optional[str] = None,
                  cmap: str = 'viridis'):
    """
    Plot the evolution of the system at multiple time points.

    Args:
        u_history: List of u field snapshots
        v_history: List of v field snapshots
        times: List of corresponding times
        n_snapshots: Number of snapshots to display
        title: Overall plot title
        save_path: Path to save the figure (optional)
        cmap: Colormap to use
    """
    # Select evenly spaced snapshots
    indices = np.linspace(0, len(u_history) - 1, n_snapshots, dtype=int)

    if v_history:
        fig, axes = plt.subplots(2, n_snapshots, figsize=(3 * n_snapshots, 6))

        for idx, i in enumerate(indices):
            # Plot u field
            im1 = axes[0, idx].imshow(u_history[i], cmap=cmap,
                                     interpolation='bilinear', origin='lower')
            axes[0, idx].set_title(f't={times[i]:.2f}', fontsize=10)
            axes[0, idx].axis('off')

            # Plot v field
            im2 = axes[1, idx].imshow(v_history[i], cmap=cmap,
                                     interpolation='bilinear', origin='lower')
            axes[1, idx].axis('off')

        axes[0, 0].set_ylabel('Species u', fontsize=12)
        axes[1, 0].set_ylabel('Species v', fontsize=12)

        # Add colorbars
        fig.colorbar(im1, ax=axes[0, :], fraction=0.046, pad=0.04)
        fig.colorbar(im2, ax=axes[1, :], fraction=0.046, pad=0.04)
    else:
        fig, axes = plt.subplots(1, n_snapshots, figsize=(3 * n_snapshots, 3))

        for idx, i in enumerate(indices):
            im = axes[idx].imshow(u_history[i], cmap=cmap,
                                interpolation='bilinear', origin='lower')
            axes[idx].set_title(f't={times[i]:.2f}', fontsize=10)
            axes[idx].axis('off')

        fig.colorbar(im, ax=axes[:], fraction=0.046, pad=0.04)

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved evolution plot to {save_path}")

    plt.show()


def create_animation(u_history: List[np.ndarray],
                    v_history: List[np.ndarray],
                    times: List[float],
                    title: str = "Reaction-Diffusion Animation",
                    save_path: Optional[str] = None,
                    fps: int = 10,
                    cmap: str = 'viridis'):
    """
    Create an animation of the system evolution.

    Args:
        u_history: List of u field snapshots
        v_history: List of v field snapshots
        times: List of corresponding times
        title: Animation title
        save_path: Path to save the animation (optional, as .gif)
        fps: Frames per second
        cmap: Colormap to use
    """
    print(f"Creating animation with {len(u_history)} frames...")

    if v_history:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Initialize plots
        im1 = ax1.imshow(u_history[0], cmap=cmap, interpolation='bilinear',
                        origin='lower', animated=True)
        ax1.set_title(f'Species u')
        ax1.axis('off')
        cb1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

        im2 = ax2.imshow(v_history[0], cmap=cmap, interpolation='bilinear',
                        origin='lower', animated=True)
        ax2.set_title(f'Species v')
        ax2.axis('off')
        cb2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

        time_text = fig.suptitle(f'{title}\nt={times[0]:.2f}', fontsize=14)

        def update(frame):
            im1.set_array(u_history[frame])
            im2.set_array(v_history[frame])
            time_text.set_text(f'{title}\nt={times[frame]:.2f}')
            return [im1, im2, time_text]

    else:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        im = ax.imshow(u_history[0], cmap=cmap, interpolation='bilinear',
                      origin='lower', animated=True)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        time_text = fig.suptitle(f'{title}\nt={times[0]:.2f}', fontsize=14)

        def update(frame):
            im.set_array(u_history[frame])
            time_text.set_text(f'{title}\nt={times[frame]:.2f}')
            return [im, time_text]

    anim = FuncAnimation(fig, update, frames=len(u_history),
                        interval=1000 / fps, blit=True, repeat=True)

    if save_path:
        writer = PillowWriter(fps=fps)
        anim.save(save_path, writer=writer)
        print(f"Animation saved to {save_path}")
    else:
        plt.show()

    return anim


def plot_concentration_profile(u: np.ndarray,
                               v: Optional[np.ndarray] = None,
                               axis: str = 'x',
                               position: Optional[int] = None,
                               time: float = 0.0,
                               title: str = "Concentration Profile",
                               save_path: Optional[str] = None):
    """
    Plot 1D concentration profiles along a specified axis.

    Args:
        u: First concentration field
        v: Second concentration field (optional)
        axis: 'x' or 'y' - axis along which to plot
        position: Index position for the slice (default: middle)
        time: Current time
        title: Plot title
        save_path: Path to save the figure (optional)
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    if axis == 'x':
        if position is None:
            position = u.shape[0] // 2
        x_coords = np.arange(u.shape[1])
        u_profile = u[position, :]
        v_profile = v[position, :] if v is not None else None
        xlabel = 'x index'
        slice_label = f'y={position}'
    else:  # axis == 'y'
        if position is None:
            position = u.shape[1] // 2
        x_coords = np.arange(u.shape[0])
        u_profile = u[:, position]
        v_profile = v[:, position] if v is not None else None
        xlabel = 'y index'
        slice_label = f'x={position}'

    ax.plot(x_coords, u_profile, 'b-', linewidth=2, label='Species u')
    if v_profile is not None:
        ax.plot(x_coords, v_profile, 'r-', linewidth=2, label='Species v')

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel('Concentration', fontsize=12)
    ax.set_title(f'{title} at t={time:.2f}, {slice_label}', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved profile to {save_path}")

    plt.show()


def plot_phase_space(u_history: List[np.ndarray],
                    v_history: List[np.ndarray],
                    sample_points: int = 1000,
                    title: str = "Phase Space",
                    save_path: Optional[str] = None):
    """
    Plot phase space diagram (u vs v).

    Args:
        u_history: List of u field snapshots
        v_history: List of v field snapshots
        sample_points: Number of random points to sample
        title: Plot title
        save_path: Path to save the figure (optional)
    """
    if not v_history:
        print("Phase space plot requires both u and v fields.")
        return

    fig, ax = plt.subplots(figsize=(8, 8))

    # Sample random points from the spatial domain over time
    colors = plt.cm.viridis(np.linspace(0, 1, len(u_history)))

    for idx, (u, v) in enumerate(zip(u_history[::max(1, len(u_history)//10)],
                                     v_history[::max(1, len(v_history)//10)])):
        # Flatten and sample
        u_flat = u.flatten()
        v_flat = v.flatten()

        n_sample = min(sample_points, len(u_flat))
        indices = np.random.choice(len(u_flat), n_sample, replace=False)

        ax.scatter(u_flat[indices], v_flat[indices], c=[colors[idx * (len(u_history)//10)]],
                  alpha=0.3, s=1)

    ax.set_xlabel('u', fontsize=12)
    ax.set_ylabel('v', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved phase space plot to {save_path}")

    plt.show()


def plot_power_spectrum(u: np.ndarray,
                       title: str = "Power Spectrum",
                       save_path: Optional[str] = None):
    """
    Plot the 2D power spectrum of a field (useful for analyzing pattern wavelengths).

    Args:
        u: Concentration field
        title: Plot title
        save_path: Path to save the figure (optional)
    """
    # Compute 2D FFT
    fft = np.fft.fft2(u)
    fft_shifted = np.fft.fftshift(fft)
    power_spectrum = np.abs(fft_shifted) ** 2

    # Use log scale for better visualization
    power_spectrum_log = np.log10(power_spectrum + 1e-10)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot original field
    im1 = ax1.imshow(u, cmap='viridis', interpolation='bilinear', origin='lower')
    ax1.set_title('Concentration Field')
    ax1.axis('off')
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    # Plot power spectrum
    im2 = ax2.imshow(power_spectrum_log, cmap='hot', interpolation='bilinear', origin='lower')
    ax2.set_title('Power Spectrum (log scale)')
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved power spectrum to {save_path}")

    plt.show()
