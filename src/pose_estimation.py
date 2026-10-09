import cv2
import numpy as np
from dataset import ImageDataset
from feature_matching import extract_features, match_features, filter_matches_lowe

class PoseEstimator:
    def __init__(self, K):
        """Store the intrinsic matrix as internal state."""
        self.K = K
    def extract_match_coordinates(kp1, kp2, matches):
        """
        Extract the (x, y) pixel coordinates from the filtered DMatch objects.
        
        Parameters:
            kp1 (tuple): Keypoints from the first image.
            kp2 (tuple): Keypoints from the second image.
            matches (list): The list of good DMatch objects from Phase 1.
            
        Returns:
            pts1 (np.ndarray): Array of shape (N, 2) containing coordinates for Image 1.
            pts2 (np.ndarray): Array of shape (N, 2) containing coordinates for Image 2.
        """
        # m.queryIdx is the index for the first image's keypoint 
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        
        # m.trainIdx is the index for the second image's keypoint 
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
        
        # Returns two aligned numpy arrays of shape (N, 2)
        return pts1, pts2

    def estimate_essential_matrix(self, pts1, pts2, K):
        """
        Compute the Essential Matrix using the 5-point algorithm and RANSAC
        
        Parameters:
            pts1 (np.ndarray): Aligned 2D coordinates from Image 1.
            pts2 (np.ndarray): Aligned 2D coordinates from Image 2.
            K (np.ndarray): 3x3 intrinsic camera matrix.
            
        Returns:
            E (np.ndarray): 3x3 Essential Matrix encoding the epipolar geometry
            mask (np.ndarray): Binary array tagging inliers (1) and outliers (0) identified by RANSAC
        """
        E, mask = cv2.findEssentialMat(
            pts1, 
            pts2, 
            K, 
            method=cv2.RANSAC, 
            prob=0.999,    # 99.9% confidence that we find a pure inlier set
            threshold=1.0  # Epipolar distance threshold in pixels to be considered an inlier
        )
        return E, mask
    def recover_camera_pose(self, E, pts1, pts2, mask):
        """
        Decompose the Essential Matrix to extract the relative rotation and translation,
        using the cheirality check to select the single physical pose with positive depth.
        
        Parameters:
            E (np.ndarray): 3x3 Essential Matrix.
            pts1 (np.ndarray): Aligned 2D coordinates from Image 1.
            pts2 (np.ndarray): Aligned 2D coordinates from Image 2.
            K (np.ndarray): 3x3 intrinsic camera matrix.
            mask (np.ndarray): Inlier mask from the Essential Matrix estimation to filter points.
            
        Returns:
            R (np.ndarray): 3x3 Rotation Matrix of the second camera relative to the first.
            T (np.ndarray): 3x1 Translation Vector of the second camera relative to the first[.
        """
        # cv2.recoverPose decomposes E, performs the cheirality check, and returns R and T
        _, R, T, _ = cv2.recoverPose(E, pts1, pts2, self.K, mask=mask)
        
        return R, T