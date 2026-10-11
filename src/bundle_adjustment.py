import numpy as np
import cv2
from scipy.optimize import least_squares

class BundleAdjuster:
    def __init__(self, K: np.ndarray, method: str = "lm"):
        """
        Constructor stores intrinsic camera parameters and solver configuration.
        
        Parameters:
            K (np.ndarray): 3x3 intrinsic camera matrix.
            method (str): Optimization method ('lm' for Levenberg-Marquardt, 'trf' for Trust Region Reflective).
        """
        self.K = K
        self.method = method

    def _project_point(self, pnt3D: np.ndarray, R: np.ndarray, T: np.ndarray) -> np.ndarray:
        """
        Helper method: Applies perspective division pi(R * X + T) using K.
        
        Parameters:
            point3D (np.ndarray): (3,) or (N, 3) 3D coordinate(s).
            R (np.ndarray): 3x3 rotation matrix.
            T (np.ndarray): 3x1 translation vector.
            
        Returns:
            uv (np.ndarray): (2,) or (N, 2) projected 2D pixel coordinates.
        """
        T = T.reshape(3, 1)

        # Step 1: Transform points to Camera 2 coordinates (3, N)
        pnt3D_cam = R @ pnt3D.T + T  # Shape: (3, N)

        # Step 2: Perspective division by depth Z_c (Row index 2)
        pnt2D_normalized = pnt3D_cam[:2, :] / pnt3D_cam[2:, :]  # Shape: (2, N)

        # Step 3: Map to pixel space using intrinsics K (2x2 focal scale + offset)
        focal = np.array([[self.K[0, 0]], [self.K[1, 1]]])  # [[fx], [fy]]
        principal_point = np.array([[self.K[0, 2]], [self.K[1, 2]]])  # [[cx], [cy]]

        pnt2D_pixel = (pnt2D_normalized * focal) + principal_point  # Shape: (2, N)

        return pnt2D_pixel.T  # Returns shape (N, 2)

    def compute_reprojection_residuals(self, params: np.ndarray, n_points: int, pts1_2d: np.ndarray, pts2_2d: np.ndarray,
    ) -> np.ndarray:
        """
        Computes the 1D vector of 2D pixel reprojection errors across both camera views[cite: 8].

        Parameters:
            R (np.ndarray): 3x3 rotation matrix for Camera 2[cite: 2].
            T (np.ndarray): 3x1 (or 3,) translation vector for Camera 2[cite: 2].
            pts3D (np.ndarray): (N, 3) Euclidean 3D coordinates[cite: 13].
            pts1_2d (np.ndarray): (N, 2) measured 2D keypoints in Image 1[cite: 8].
            pts2_2d (np.ndarray): (N, 2) measured 2D keypoints in Image 2[cite: 8].

        Returns:
            residuals (np.ndarray): 1D array of length 4 * N containing signed pixel errors[cite: 8, 13].
        """
        # 1. UNFLATTEN 1D vector into structured geometric parameters
        rvec = params[0:3]                         # 3D rotation vector (so3)
        T = params[3:6].reshape(3, 1)             # 3x1 translation vector
        pts3D = params[6:].reshape(n_points, 3)    # N x 3 point cloud coordinates

        # 2. Reconstruct rotation matrix R from rvec
        R, _ = cv2.Rodrigues(rvec)

        # 2. View 1 Projection: Camera 1 is at origin (R = Eye, T = 0)[cite: 2]
        pts1_proj = self._project_point(pts3D, np.eye(3), np.zeros((3, 1)))

        # 3. View 2 Projection: Camera 2 uses provided (R, T)[cite: 2]
        pts2_proj = self._project_point(pts3D, R, T)

        # 4. Calculate signed 2D pixel differences (Measured - Projected)[cite: 8, 13]
        err1 = (pts1_2d - pts1_proj).ravel()  # Length: 2 * N
        err2 = (pts2_2d - pts2_proj).ravel()  # Length: 2 * N

        # 5. Concatenate into a single flat residual vector[cite: 8, 13]
        return np.concatenate([err1, err2])

    def optimize_2view(self, pts3D_init: np.ndarray, R_init: np.ndarray, T_init: np.ndarray, 
                       pts1_2d: np.ndarray, pts2_2d: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Runs the iterative non-linear solver (Gauss-Newton / Levenberg-Marquardt) 
        to refine point cloud coordinates and relative camera pose.
        
        Parameters:
            pts3D_init (np.ndarray): (N, 3) initial triangulated points (X_0 from DLT/SVD).
            R_init (np.ndarray): 3x3 initial rotation matrix.
            T_init (np.ndarray): 3x1 initial translation vector.
            pts1_2d (np.ndarray): (N, 2) measured 2D keypoints in Image 1.
            pts2_2d (np.ndarray): (N, 2) measured 2D keypoints in Image 2.
            
        Returns:
            pts3D_opt (np.ndarray): (N, 3) refined 3D coordinates.
            R_opt (np.ndarray): 3x3 refined rotation matrix.
            T_opt (np.ndarray): 3x1 refined translation vector.
            final_rmse (float): Final Root Mean Square Error in pixels.
        """
        n_points = len(pts3D_init)
        rvec_init, _ = cv2.Rodrigues(R_init)

        # 1. FLATTEN
        params_init = np.hstack([rvec_init.flatten(), T_init.flatten(), pts3D_init.flatten()])

        # 2. RUN LEAST SQUARES SOLVER
        result = least_squares(
            fun=self.compute_reprojection_residuals,
            x0=params_init,
            args=(n_points, pts1_2d, pts2_2d),
            method='lm'  # Levenberg-Marquardt
        )

        # 3. UNFLATTEN back to clean, structured matrices
        rvec_opt = result.x[0:3]
        R_opt, _ = cv2.Rodrigues(rvec_opt)
        T_opt = result.x[3:6].reshape(3, 1)
        pts3D_opt = result.x[6:].reshape(n_points, 3)
        opt_rmse = np.sqrt(np.mean(result.fun**2))

        return pts3D_opt, R_opt, T_opt, opt_rmse