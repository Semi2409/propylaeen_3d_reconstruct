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
        P1 = self.K @ np.hstack((np.eye(3)), np.zeros(3, 1))
        P2 = self.K @ np.hstack(R, T.reshape(3, 1))
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

    def filter_cheirality(self, pts3D: np.ndarray, R: np.ndarray, T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Filters points behind the camera (Z <= 0)."""
        pass