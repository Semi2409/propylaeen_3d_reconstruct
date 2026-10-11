import numpy as np
import cv2

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
        P1 = self.K @ np.hstack((np.eye(3), np.zeros((3, 1))))
        P2 = self.K @ np.hstack((R, T.reshape(3, 1)))
        return P1, P2
    
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

    def filter_cheirality(self, pts3D: np.ndarray, R: np.ndarray, T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Filters 3D points to keep only those in front of both cameras (Z > 0).
        """
        # Ensure T is a column vector (3, 1)
        T = T.reshape(3, 1)

        # Depth in Camera 1 frame (Z-coordinate)
        depth1 = pts3D[:, 2]

        # Transform points to Camera 2 frame: X2 = R @ X1 + T
        pts3D_cam2 = (R @ pts3D.T + T).T
        depth2 = pts3D_cam2[:, 2]

        # Both depths must be strictly positive (> 0)
        valid_mask = (depth1 > 0) & (depth2 > 0)
        valid_pts3D = pts3D[valid_mask]
        return valid_pts3D, valid_mask