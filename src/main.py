import numpy as np
from dataset import ImageDataset
from feature_matching import FeatureMatcher
from pose_estimation import PoseEstimator

def main():
    # 1. Initialize dataset and load images
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
    

    # 3. Initialize Pipeline Classes
    matcher = FeatureMatcher()
    estimator = PoseEstimator(K)

    # 4. Execute Pipeline in Memory (No disk redundancy)
    print("Matching features...")
    pts1, pts2 = matcher.process_pair(img1, img2)
    
    print("Estimating camera pose...")
    R, T, mask = estimator.estimate_pose(pts1, pts2)

    print("Rotation:\n", R)
    print("Translation:\n", T)

if __name__ == "__main__":
    main()