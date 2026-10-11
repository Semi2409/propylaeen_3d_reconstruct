import os
import numpy as np
import cv2
import matplotlib.pyplot as plt


def export_ply(filename: str, pts3D: np.ndarray, colors: np.ndarray = None) -> None:
    """
    Exports 3D points and optional RGB colors to a standard ASCII .PLY file.
    Can be opened directly in Open3D, MeshLab, or CloudCompare.

    Parameters:
        filename (str): Target output file path (e.g., 'outputs/reconstruction.ply').
        pts3D (np.ndarray): (N, 3) array of Euclidean 3D coordinates.
        colors (np.ndarray, optional): (N, 3) array of RGB color values in [0, 255].
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    pts3D = pts3D.reshape(-1, 3)
    n_points = len(pts3D)

    header = [
        "ply",
        "format ascii 1.0",
        f"element vertex {n_points}",
        "property float x",
        "property float y",
        "property float z",
    ]

    has_color = colors is not None and len(colors) == n_points
    if has_color:
        header.extend([
            "property uchar red",
            "property uchar green",
            "property uchar blue"
        ])

    header.append("end_header")

    with open(filename, "w") as f:
        f.write("\n".join(header) + "\n")
        for i in range(n_points):
            x, y, z = pts3D[i]
            if has_color:
                r, g, b = colors[i].astype(int)
                f.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n")
            else:
                f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")

    print(f"Successfully exported {n_points} 3D points to {filename}")


def plot_point_cloud_matplotlib(
    pts3D: np.ndarray, 
    pts3D_opt: np.ndarray = None, 
    max_points: int = 2000
) -> None:
    """
    Renders an interactive 3D scatter plot of the point cloud inside Matplotlib/Jupyter Notebook.

    Parameters:
        pts3D (np.ndarray): Initial triangulated 3D points (N, 3).
        pts3D_opt (np.ndarray, optional): Optimized 3D points (N, 3) after Bundle Adjustment.
        max_points (int): Max number of points to render for UI performance.
    """
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Subsample if point cloud is huge
    if len(pts3D) > max_points:
        idx = np.random.choice(len(pts3D), max_points, replace=False)
        pts3D_sub = pts3D[idx]
        pts3D_opt_sub = pts3D_opt[idx] if pts3D_opt is not None else None
    else:
        pts3D_sub = pts3D
        pts3D_opt_sub = pts3D_opt

    # Plot initial points
    ax.scatter(pts3D_sub[:, 0], pts3D_sub[:, 1], pts3D_sub[:, 2], 
               c='red', s=2, alpha=0.6, label='Initial Triangulated (DLT)')

    # Plot optimized points if provided
    if pts3D_opt_sub is not None:
        ax.scatter(pts3D_opt_sub[:, 0], pts3D_opt_sub[:, 1], pts3D_opt_sub[:, 2], 
                   c='blue', s=2, alpha=0.8, label='Optimized (Bundle Adjustment)')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Point Cloud Reconstruction')
    ax.legend()
    plt.show()


def compute_rmse(residuals: np.ndarray) -> float:
    """
    Calculates the Root Mean Square Error (RMSE) in pixels from a 1D residual vector.

    Parameters:
        residuals (np.ndarray): Flat 1D residual vector.

    Returns:
        rmse (float): Reprojection error in pixels.
    """
    return float(np.sqrt(np.mean(residuals ** 2)))

def build_intrinsic_matrix(width: int, height: int, focal_length: float = 18.0, sx: float = 22.3, sy: float = 14.9) -> np.ndarray:
    """Constructs the 3x3 camera intrinsic matrix K from sensor specs and image resolution."""
    return np.array([
        [focal_length * width / sx, 0, width / 2.0],
        [0, focal_length * height / sy, height / 2.0],
        [0, 0, 1]
    ], dtype=np.float32)