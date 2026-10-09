import cv2
import numpy as np
from dataset import ImageDataset
from feature_matching import extract_features, match_features, filter_matches_lowe
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

def estimate_essential_matrix(pts1, pts2, K):
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
def recover_camera_pose(E, pts1, pts2, K, mask):
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
    _, R, T, _ = cv2.recoverPose(E, pts1, pts2, K, mask=mask)
    
    return R, T

def main():
    """
    Main execution pipeline:
    1. Define the 3x3 intrinsic matrix K.
    2. Load keypoints and matches from Phase 1.
    3. Extract the aligned (x, y) coordinates using extract_match_coordinates.
    4. Compute the Essential Matrix and inlier mask using estimate_essential_matrix.
    5. Recover the camera pose (R, T) using recover_camera_pose.
    """
    # 1. Define the 3x3 intrinsic camera matrix (K)
    # Using typical image dimensions (e.g., 1920x1080) and an estimated focal length in pixels
    dataset = ImageDataset("data/")
    img1 = dataset[0]  
    img2 = dataset[1]  
    height, width = img1.shape[:2]
    focal_length = 18
    sx = 22.3
    sy = 14.9
    K = np.array([
        [focal_length*width/sx, 0, width / 2.0],
        [0, focal_length*height/sy, height / 2.0],
        [0, 0, 1]
    ], dtype=np.float32)

    # 2. Load keypoints and filtered matches from Phase 1
    # (Assuming you are importing your dataset and Phase 1 feature extraction functions)
    

    dataset = ImageDataset("data/")
    img1 = dataset[0]  # First image in sequence
    img2 = dataset[1]  # Second image in sequence

    # Extract features and find good matches
    kp1, des1 = extract_features(img1)
    kp2, des2 = extract_features(img2)
    raw_matches = match_features(des1, des2)
    good_matches = filter_matches_lowe(raw_matches)

    # 3. Extract the aligned (x, y) pixel coordinates
    pts1, pts2 = extract_match_coordinates(kp1, kp2, good_matches)

    # 4. Compute the Essential Matrix using the 5-point algorithm and RANSAC
    E, mask = estimate_essential_matrix(pts1, pts2, K)

    # 5. Recover the relative camera pose (Rotation R and Translation T)[cite: 3, 7]
    R, T = recover_camera_pose(E, pts1, pts2, K, mask)

    # Print out results to verify
    print("Estimated Rotation Matrix (R):\n", R)
    print("Estimated Translation Vector (T):\n", T)
    print(f"Number of inlier points used: {mask.sum()} / {len(mask)}")

if __name__ == "__main__":
    main()