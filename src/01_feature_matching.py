import cv2
import numpy as np
import os
from pathlib import Path
from dataset import ImageDataset


def extract_features(image):
    """
    Detect keypoints and compute descriptors for a given image.
    
    Parameters:
        image (numpy.ndarray): Input BGR or Grayscale image array.
        
    Returns:
        keypoints (tuple): Detected OpenCV KeyPoint objects.
        descriptors (numpy.ndarray): Feature descriptors matrix (e.g. SIFT vectors).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()  #[cite: 1]
    keypoints, descriptors = sift.detectAndCompute(
        gray, None
    )  # Extract keypoints and descriptors[cite: 1]
    return keypoints, descriptors


def match_features(descriptors1, descriptors2):
    """
    Find raw matches between two descriptor matrices using a nearest-neighbor matcher (e.g. FLANN or BFMatcher).
    
    Parameters:
        descriptors1 (numpy.ndarray): Descriptors from image 1.
        descriptors2 (numpy.ndarray): Descriptors from image 2.
        
    Returns:
        matches (list): List of match objects (e.g., DMatch objects or knn pairs).
    """
    # NORM_L2 is the standard Euclidean distance used for SIFT float vectors
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    
    # k=2 returns the 2 closest matches for each descriptor
    matches = bf.knnMatch(descriptors1, descriptors2, k=2) 
    
    return matches


def filter_matches_lowe(matches, ratio=0.75):
    """
    Apply Lowe's Ratio Test to eliminate ambiguous or weak matches.
    
    Parameters:
        matches (list): Raw k-NN matches from matcher.
        ratio (float): Lowe's threshold factor (typically 0.7 to 0.8).
        
    Returns:
        good_matches (list): Filtered list of robust match objects.
    """
    good_matches = []
    
    # matches is a list of tuples containing the 2 nearest neighbors (m, n) for each point
    for m, n in matches:
        # m is the nearest neighbor; n is the second-nearest
        if m.distance < ratio * n.distance:
            good_matches.append(m)  #
            
    return good_matches


def draw_and_save_matches(img1, kp1, img2, kp2, matches, output_path):
    """
    Visualize keypoint correspondences side-by-side and save the output image.
    
    Parameters:
        img1 (numpy.ndarray): First image.
        kp1 (tuple): Keypoints of first image.
        img2 (numpy.ndarray): Second image.
        kp2 (tuple): Keypoints of second image.
        matches (list): Filtered matches to draw.
        output_path (Path or str): Path where output image should be saved.
    """
    match_img = cv2.drawMatches(
        img1, kp1, img2, kp2, matches, None, 
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )  #
    
    # Safely create the parent directories if they don't exist yet
    out_dir = Path(output_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the final stitched image to disk
    cv2.imwrite(str(output_path), match_img)  #


def main():
    """
    Main execution pipeline:
    1. Initialize ImageDataset.
    2. Load image pair (e.g., img0 and img1).
    3. Run feature extraction, matching, and filtering.
    4. Save visualization results.
    """
    dataset = ImageDataset("data")
    
   
    img0 = dataset[0]
    img1 = dataset[1]
    
    kp1, descriptor1 = extract_features(img0) 
    kp2, descriptor2 = extract_features(img1) 
    
    matches = match_features(descriptor1, descriptor2) 
    good_matches = filter_matches_lowe(matches)  

    output_path = "notebooks/outputs/feature_matches.png" 
    draw_and_save_matches(img0, kp1, img1, kp2, good_matches, output_path) 
    
    print(f"Successfully filtered down to {len(good_matches)} robust matches.")


if __name__ == "__main__":
    main()