import numpy as np
from dataset import ImageDataset
from feature_matching import FeatureMatcher
from pose_estimation import PoseEstimator
from triangulation import Triangulator
from bundle_adjustment import BundleAdjuster
from utils import (
    build_intrinsic_matrix,
    export_ply,
    plot_point_cloud_matplotlib,
    compute_rmse,
    )


def main():
    # 1. Initialize dataset and load images
    print("Loading dataset...")
    dataset = ImageDataset("data/")
    img1 = dataset[0]
    img2 = dataset[1]

    # Camera Intrinsics Matrix K via utils helper
    height, width = img1.shape[:2]
    K = build_intrinsic_matrix(width=width, height=height)

    # 2. Initialize pipeline classes
    matcher = FeatureMatcher()
    estimator = PoseEstimator(K)
    triangulator = Triangulator(K)
    optimizer = BundleAdjuster(K)

    # 3. Feature Extraction & Matching
    print("Extracting and matching features...")
    kp1, des1 = matcher.extract_features(img1)
    kp2, des2 = matcher.extract_features(img2)
    raw_matches = matcher.match_features(des1, des2)
    good_matches = matcher.filter_matches_lowe(raw_matches)

    # 4. Pose Estimation
    print("Estimating camera pose...")
    pts1, pts2 = estimator.extract_match_coordinates(kp1, kp2, good_matches)
    E, mask = estimator.estimate_essential_matrix(pts1, pts2)
    R, T = estimator.recover_camera_pose(E, pts1, pts2, mask)

    # Filter 2D keypoints using the RANSAC inlier mask
    inlier_mask = mask.ravel() == 1
    pts1_inliers = pts1[inlier_mask]
    pts2_inliers = pts2[inlier_mask]

    # 5. Triangulation
    print("Triangulating initial 3D points...")
    P1, P2 = triangulator.construct_projection_matrices(R, T)
    pts3D_raw = triangulator.triangulate_points(P1, P2, pts1_inliers, pts2_inliers)
    pts3D_init, valid_mask = triangulator.filter_cheirality(pts3D_raw, R, T)

    pts1_valid = pts1_inliers[valid_mask]
    pts2_valid = pts2_inliers[valid_mask]

    # 6. Bundle Adjustment
    print("Optimizing point cloud via Bundle Adjustment...")
    init_residuals = optimizer.compute_reprojection_residuals(
        R, T, pts3D_init, pts1_valid, pts2_valid
    )
    init_rmse = compute_rmse(init_residuals)

    pts3D_opt, R_opt, T_opt, opt_rmse = optimizer.optimize_2view(
        pts3D_init=pts3D_init,
        R_init=R,
        T_init=T,
        pts1_2d=pts1_valid,
        pts2_2d=pts2_valid,
    )

    print(f"Initial Reprojection RMSE: {init_rmse:.4f} px")
    print(f"Optimized Reprojection RMSE: {opt_rmse:.4f} px")

    # 7. Export & Visualization
    print("Exporting .ply files and plotting point clouds...")
    export_ply("outputs/initial_point_cloud.ply", pts3D_init)
    export_ply("outputs/optimized_point_cloud.ply", pts3D_opt)
    plot_point_cloud_matplotlib(pts3D_init, pts3D_opt)


if __name__ == "__main__":
    main()