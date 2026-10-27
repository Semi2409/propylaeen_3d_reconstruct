import numpy as np
import cv2


def camera_center(R: np.ndarray, T: np.ndarray) -> np.ndarray:
    """(3,) optical center C = -R^T @ T of a camera with pose (R, T)."""
    return (-np.asarray(R).T @ np.asarray(T).reshape(3, 1)).ravel().astype(np.float64)


class Triangulator:
    def __init__(self, K: np.ndarray):
        """
        Constructor stores persistent camera parameters (State).
        
        Parameters:
            K (np.ndarray): 3x3 intrinsic camera matrix.
        """
        self.K = K

    def construct_projection_matrices(self, R: np.ndarray, T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Constructs P1 and P2 using stored self.K along with R and T.
        
        Parameters:
            R (np.ndarray): 3x3 rotation matrix.
            T (np.ndarray): 3x1 translation vector.
            
        Returns:
            P1 (np.ndarray): 3x4 projection matrix for Camera 1.
            P2 (np.ndarray): 3x4 projection matrix for Camera 2.
        """
        return self.projection_matrix(np.eye(3), np.zeros(3)), self.projection_matrix(R, T)

    def projection_matrix(self, R: np.ndarray, T: np.ndarray) -> np.ndarray:
        """3x4 projection matrix P = K [R | T] of a camera with pose (R, T)."""
        return self.K @ np.hstack((R, T.reshape(3, 1)))
    
    def triangulate_points(self, P1: np.ndarray, P2: np.ndarray, pts1: np.ndarray, pts2: np.ndarray) -> np.ndarray:
        """
        Solves W * X_hom = 0 using SVD and dehomogenizes output to (N, 3).
        
        Parameters:
            P1 (np.ndarray): 3x4 projection matrix for Camera 1.
            P2 (np.ndarray): 3x4 projection matrix for Camera 2.
            pts1 (np.ndarray): (N, 2) pixel coordinates from Image 1.
            pts2 (np.ndarray): (N, 2) pixel coordinates from Image 2.
            
        Returns:
            pts3D (np.ndarray): (N, 3) Euclidean 3D coordinates.
        """
        pts4D = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)

        # 1. Transpose to shape (N, 4)
        pts4D = pts4D.T

        # 2. Dehomogenize: divide (X, Y, Z) by W
        pts3D = pts4D[:, :3] / pts4D[:, 3:]
        return pts3D

    @staticmethod
    def point_depths(pts3D: np.ndarray, R: np.ndarray, T: np.ndarray) -> np.ndarray:
        """Depth (Z in the camera frame) of world points: X_cam = R @ X + T."""
        return (R @ pts3D.T + T.reshape(3, 1))[2]

    def filter_cheirality(self, pts3D: np.ndarray, R: np.ndarray, T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Two-view cheirality: keeps the points in front of both cameras (depth > 0), with camera 1 at
        [I | 0] and camera 2 at (R, T). For two arbitrary posed cameras use filter_triangulated_points.
        """
        # Both depths must be strictly positive (> 0)
        valid_mask = (self.point_depths(pts3D, np.eye(3), np.zeros(3)) > 0) & (self.point_depths(pts3D, R, T) > 0)
        return pts3D[valid_mask], valid_mask

    def reprojection_errors(self, pts3D: np.ndarray, R: np.ndarray, T: np.ndarray, pts2D: np.ndarray) -> np.ndarray:
        """
        Pixel distance between the observed 2D points and the projection x ~ K (R X + T).
        Pixels are a real unit, so this check does not depend on the arbitrary SfM scale.
        """
        proj = self.K @ (R @ pts3D.T + T.reshape(3, 1))
        with np.errstate(divide="ignore", invalid="ignore"):
            uv = (proj[:2] / proj[2:]).T
        return np.linalg.norm(uv - pts2D, axis=1)

    def track_reprojection_errors(self, cameras: dict, track_graph, all_kp: dict):
        """
        Reprojection error (px) and depth of every observation in a track graph, for registered
        cameras and existing points only. Computed per camera with reprojection_errors.
        Returns (observations, errors, depths) with observations = [((cam_id, kp_idx), point_id), ...].
        """
        obs = [(o, p) for o, p in track_graph.kp_to_point3D.items()
               if o[0] in cameras and p in track_graph.global_points3D]
        errors, depths = np.zeros(len(obs)), np.zeros(len(obs))
        rows_by_cam = {}
        for i, ((cam_id, _), _) in enumerate(obs):
            rows_by_cam.setdefault(cam_id, []).append(i)
        for cam_id, rows in rows_by_cam.items():
            rows = np.array(rows)
            X = np.array([np.asarray(track_graph.global_points3D[obs[i][1]], dtype=np.float64).ravel() for i in rows])
            uv = np.array([all_kp[cam_id][obs[i][0][1]].pt for i in rows])
            R, T = cameras[cam_id]["R"], cameras[cam_id]["T"]
            errors[rows] = self.reprojection_errors(X, R, T, uv)
            depths[rows] = self.point_depths(X, R, T)
        return obs, errors, depths

    @staticmethod
    def triangulation_angles(pts3D: np.ndarray, R1: np.ndarray, T1: np.ndarray,
                             R2: np.ndarray, T2: np.ndarray) -> np.ndarray:
        """
        Angle (degrees) between the two viewing rays C1 -> X and C2 -> X, with camera centers C = -R^T T.
        Small angle = rays nearly parallel = depth badly constrained. Angles are scale invariant.
        """
        ray1, ray2 = pts3D - camera_center(R1, T1), pts3D - camera_center(R2, T2)
        cos = np.sum(ray1 * ray2, axis=1) / (np.linalg.norm(ray1, axis=1) * np.linalg.norm(ray2, axis=1) + 1e-12)
        return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))

    def filter_triangulated_points(self, pts3D: np.ndarray,
                                   R1: np.ndarray, T1: np.ndarray, pts1: np.ndarray,
                                   R2: np.ndarray, T2: np.ndarray, pts2: np.ndarray,
                                   max_reproj_px: float = 4.0, min_angle_deg: float = 1.5,
                                   max_depth_factor: float = 10.0, min_points_for_median: int = 10) -> np.ndarray:
        """
        Scale-free validity checks for points triangulated from two posed cameras. Returns a boolean mask.
          1. Cheirality: depth > 0 in both cameras.
          2. Reprojection error <= max_reproj_px in both images.
          3. Triangulation angle >= min_angle_deg.
          4. Depth <= max_depth_factor * median depth of the points that passed 1-3 (per camera).
        """
        d1, d2 = self.point_depths(pts3D, R1, T1), self.point_depths(pts3D, R2, T2)
        mask = (d1 > 0) & (d2 > 0)
        mask &= self.reprojection_errors(pts3D, R1, T1, pts1) <= max_reproj_px
        mask &= self.reprojection_errors(pts3D, R2, T2, pts2) <= max_reproj_px
        mask &= self.triangulation_angles(pts3D, R1, T1, R2, T2) >= min_angle_deg

        # Relative depth: only meaningful with enough good points to take a median from
        if mask.sum() >= min_points_for_median:
            mask &= (d1 <= max_depth_factor * np.median(d1[mask])) & (d2 <= max_depth_factor * np.median(d2[mask]))
        return mask