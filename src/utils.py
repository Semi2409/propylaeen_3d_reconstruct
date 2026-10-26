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

def initialize_seed_submodel(seed_pair, estimator, triangulator, optimizer, all_kp, all_des, match_graph, track_graph_class):
        """
        Initializes a baseline 2-view sub-model in its own local coordinate space.
        Returns registered_cameras dict and a new TrackGraph instance.
        """
        seed_a, seed_b = seed_pair

        # 1. Pull features from memory bank
        kp0, des0 = all_kp[seed_a], all_des[seed_a]
        kp1, des1 = all_kp[seed_b], all_des[seed_b]
        good_matches = match_graph[(seed_a, seed_b)]

        # 2. Extract coordinates & Estimate Essential Matrix
        pts0, pts1 = estimator.extract_match_coordinates(kp0, kp1, good_matches)
        E, ransac_mask = estimator.estimate_essential_matrix(pts0, pts1)
        R1, T1 = estimator.recover_camera_pose(E, pts0, pts1, ransac_mask)

        inlier_idx = (ransac_mask.ravel() == 1)
        pts0_in, pts1_in = pts0[inlier_idx], pts1[inlier_idx]

        # 3. Triangulate & Filter Cheirality
        P0, P1_rel = triangulator.construct_projection_matrices(R1, T1)
        pts3D_raw = triangulator.triangulate_points(P0, P1_rel, pts0_in, pts1_in)
        pts3D_init, cheirality_mask = triangulator.filter_cheirality(pts3D_raw, R1, T1)

        # 4. Optimize 2-View Seed
        pts3D_opt, R1_opt, T1_opt, init_rmse = optimizer.optimize_2view(
            pts3D_init=pts3D_init, R_init=R1, T_init=T1,
            pts1_2d=pts0_in[cheirality_mask], pts2_2d=pts1_in[cheirality_mask]
        )

        # 5. Build Sub-Model Local Cameras & Track Graph
        sub_cameras = {
            seed_a: {"R": np.eye(3), "T": np.zeros((3, 1))},
            seed_b: {"R": R1_opt, "T": T1_opt}
        }

        sub_track_graph = track_graph_class() # Instantiate local TrackGraph container

        inlier_matches = [m for m, is_inlier in zip(good_matches, inlier_idx) if is_inlier]
        valid_seed_matches = [m for m, is_valid in zip(inlier_matches, cheirality_mask) if is_valid]

        # Populate 3D point tracks into local track graph
        for pt3d, m in zip(pts3D_opt, valid_seed_matches):
            sub_track_graph.add_point3D(
                pt3d, 
                observations=[(seed_a, m.queryIdx), (seed_b, m.trainIdx)]
            )

        print(f"Sub-Model ({seed_a}, {seed_b}) Initialized: {len(sub_track_graph.global_points3D)} points | RMSE: {init_rmse:.4f} px")
        return sub_cameras, sub_track_graph


def register_submodel_cameras(registered_cameras: dict, track_graph, image_indices: list[int], all_kp: dict, all_des: dict, match_graph: dict, K: np.ndarray, estimator, triangulator, optimizer,
    max_passes: int = 3, max_height_m: float = 1.5, max_step_m: float = 10.0, min_baseline_m: float = 0.05
):
    """
    Runs multi-pass incremental PnP registration, DLT triangulation, and sliding window 
    Local BA over a target list of image indices for a sub-model.
    """
    unregistered = set(image_indices) - set(registered_cameras.keys())
    
    for pass_num in range(max_passes):
        registered_this_pass = 0
        
        # Process unregistered pool for this sub-model
        for curr_idx in sorted(list(unregistered)):
            if curr_idx in registered_cameras:
                continue
                
            kp_curr = all_kp[curr_idx]
            
            # 1. Gather 2D-to-3D correspondences from currently registered cameras via TrackGraph
            pts2D = []
            pts3D = []
            p3d_ids_used = []

            for reg_idx in list(registered_cameras.keys()):
                pair = (min(curr_idx, reg_idx), max(curr_idx, reg_idx))
                if pair in match_graph:
                    matches = match_graph[pair]
                    for m in matches:
                        kp_reg_idx = m.trainIdx if curr_idx < reg_idx else m.queryIdx
                        kp_curr_idx = m.queryIdx if curr_idx < reg_idx else m.trainIdx
                        
                        p3d_id = track_graph.get_point3D_id(reg_idx, kp_reg_idx)
                        if p3d_id is not None:
                            pts3D.append(track_graph.global_points3D[p3d_id])
                            pts2D.append(kp_curr[kp_curr_idx].pt)
                            p3d_ids_used.append((kp_curr_idx, p3d_id))

            # Filter unphysical 3D map matches
            clean_pts2D, clean_pts3D, clean_p3d_ids = [], [], []
            for pt2d, pt3d, p3d_info in zip(pts2D, pts3D, p3d_ids_used):
                if np.linalg.norm(pt3d) < 50.0 and 40.0 > pt3d[2] > 0.0:
                    clean_pts2D.append(pt2d)
                    clean_pts3D.append(pt3d)
                    clean_p3d_ids.append(p3d_info)

            pts2D, pts3D, p3d_ids_used = clean_pts2D, clean_pts3D, clean_p3d_ids

            if len(pts2D) < 15:
                continue

            # 2. Estimate Camera Pose via PnP RANSAC
            success, rvec, tvec, pnp_inliers = cv2.solvePnPRansac(
                np.float32(pts3D), np.float32(pts2D), K, distCoeffs=None,
                flags=cv2.SOLVEPNP_ITERATIVE, reprojectionError=5.0, confidence=0.99
            )

            if not success or pnp_inliers is None or len(pnp_inliers) < 15:
                continue

            R_curr, _ = cv2.Rodrigues(rvec)
            T_curr = tvec.reshape(3, 1)
            C_curr = (-R_curr.T @ T_curr).flatten()
                    
            # Height Guardrail (Z height relative to local seed origin)
            if abs(C_curr[1]) > max_height_m:
                continue

            # Motion Jump Guardrail (relative to last registered camera in this sub-model)
            if registered_cameras:
                last_reg_id = list(registered_cameras.keys())[-1]
                R_last = registered_cameras[last_reg_id]["R"]
                T_last = registered_cameras[last_reg_id]["T"]
                C_last = (-R_last.T @ T_last).flatten()
                
                step_distance = np.linalg.norm(C_curr - C_last)
                if step_distance > max_step_m:
                    continue

            # Pose accepted! Register camera & link PnP inliers to TrackGraph
            registered_cameras[curr_idx] = {"R": R_curr, "T": T_curr}
            unregistered.remove(curr_idx)
            registered_this_pass += 1
            
            for idx in pnp_inliers.flatten():
                kp_curr_idx, p3d_id = p3d_ids_used[idx]
                track_graph.link_keypoint_to_track(curr_idx, kp_curr_idx, p3d_id)

            # 3. DLT Triangulation to Expand Map
            P_curr = K @ np.hstack((R_curr, T_curr))
            
            for reg_idx in list(registered_cameras.keys()):
                if reg_idx == curr_idx:
                    continue

                pair = (min(curr_idx, reg_idx), max(curr_idx, reg_idx))
                if pair not in match_graph:
                    continue

                matches = match_graph[pair]
                R_prev = registered_cameras[reg_idx]["R"]
                T_prev = registered_cameras[reg_idx]["T"]
                P_prev = K @ np.hstack((R_prev, T_prev))
                C_prev = (-R_prev.T @ T_prev).flatten()
                
                # Pairwise baseline guardrail
                if np.linalg.norm(C_curr - C_prev) < min_baseline_m:
                    continue

                # Extract matching 2D coordinates
                if curr_idx < reg_idx:
                    pts_curr, pts_prev = estimator.extract_match_coordinates(
                        all_kp[curr_idx], all_kp[reg_idx], matches
                    )
                else:
                    pts_prev, pts_curr = estimator.extract_match_coordinates(
                        all_kp[reg_idx], all_kp[curr_idx], matches
                    )

                raw_pts3D = triangulator.triangulate_points(P_prev, P_curr, pts_prev, pts_curr)
                valid_pts3D, v_mask = triangulator.filter_cheirality(raw_pts3D, R_curr, T_curr)
                T_curr_col = T_curr.reshape(3, 1)

                valid_counter = 0
                for idx_in_pair, is_valid in enumerate(v_mask):
                    if not is_valid:
                        continue
                        
                    p3d_val = valid_pts3D[valid_counter]
                    valid_counter += 1

                    # Depth filtering in current camera coordinate space
                    pt3d_cam = R_curr @ p3d_val.reshape(3, 1) + T_curr_col
                    z_depth = pt3d_cam[2, 0]

                    if not (0.2 < z_depth < 40.0 and np.linalg.norm(p3d_val) < 50.0):
                        continue

                    m = matches[idx_in_pair]
                    kp_curr_i = m.queryIdx if curr_idx < reg_idx else m.trainIdx
                    kp_reg_i  = m.trainIdx if curr_idx < reg_idx else m.queryIdx

                    existing_id = track_graph.get_point3D_id(reg_idx, kp_reg_i)
                    if existing_id is not None:
                        track_graph.link_keypoint_to_track(curr_idx, kp_curr_i, existing_id)
                    else:
                        track_graph.add_point3D(
                            p3d_val, 
                            observations=[(reg_idx, kp_reg_i), (curr_idx, kp_curr_i)]
                        )

            # 4. Sliding Window Local Bundle Adjustment
            optimizer.optimize_local_window(
                registered_cameras=registered_cameras,
                track_graph=track_graph,
                all_kp=all_kp,
                window_size=3,
                max_iter=30
            )

        print(f"Pass {pass_num + 1} complete: Registered {registered_this_pass} cameras (Total in Sub-Model: {len(registered_cameras)}).")
        if registered_this_pass == 0:
            break

    return registered_cameras, track_graph


def umeyama_sim3(source_pts: np.ndarray, target_pts: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Computes the optimal Sim(3) transformation parameters (Scale s, Rotation R, Translation t)
    that align 3D source points/camera centers onto target points:
    
        Target = s * R * Source + t

    Uses Umeyama's algorithm via SVD.

    Parameters:
        source_pts (np.ndarray): (N, 3) 3D points from Sub-Model B (to be aligned).
        target_pts (np.ndarray): (N, 3) 3D points from Sub-Model A (reference frame).

    Returns:
        scale (float): Isotropic scaling factor s.
        R (np.ndarray): (3, 3) Rotation matrix.
        translation (np.ndarray): (3,) Translation vector.
    """
    assert source_pts.shape == target_pts.shape and source_pts.shape[1] == 3, "Input arrays must be (N, 3)"
    n, dim = source_pts.shape

    # 1. Compute Centroids
    mu_src = np.mean(source_pts, axis=0)
    mu_tgt = np.mean(target_pts, axis=0)

    # 2. Demean points
    src_demean = source_pts - mu_src
    tgt_demean = target_pts - mu_tgt

    # 3. Variance & Covariance
    var_src = np.sum(src_demean ** 2) / n
    cov = (tgt_demean.T @ src_demean) / n

    # 4. SVD Decomposition
    U, S, Vt = np.linalg.svd(cov)
    R = U @ Vt

    # Reflectivity Guard: Ensure det(R) == +1 (valid rotation, not reflection)
    if np.linalg.det(R) < 0:
        S[-1] = -S[-1]
        U[:, -1] = -U[:, -1]
        R = U @ Vt

    # 5. Scale & Translation Recovery
    scale = (1.0 / var_src) * np.sum(S)
    translation = mu_tgt - scale * (R @ mu_src)

    return scale, R, translation




# Helper function to align and merge a sub-model into the Master Model
def align_and_merge(master_cams, master_tck,sub_cams, sub_track, ref_cams):

    master_cameras = {**master_cams}
    master_track = master_tck
    # Find shared camera indices between the sub-model and reference cameras
    shared_ids = list(set(ref_cams.keys()) & set(sub_cams.keys()))
    print(f"Shared corner cameras for alignment: {shared_ids}")
    
    # Extract physical optical centers: C = -R^T @ T
    pts_source = np.array([(-sub_cams[idx]["R"].T @ sub_cams[idx]["T"]).flatten() for idx in shared_ids])
    pts_target = np.array([(-ref_cams[idx]["R"].T @ ref_cams[idx]["T"]).flatten() for idx in shared_ids])

    # 1. Compute Sim(3) parameters (Scale, Rotation, Translation)
    scale, R_sim, t_sim = umeyama_sim3(pts_source, pts_target)
    print(f"  -> Solved Sim(3): Scale = {scale:.4f}, Translation = {t_sim.round(2)}")

    # 2. Transform sub-model 3D points into Master space
    # X_master = s * R * X_local + t
    for p3d_id, pt3d in list(sub_track.global_points3D.items()):
        transformed_pt = scale * (R_sim @ pt3d) + t_sim
        sub_track.global_points3D[p3d_id] = transformed_pt

    # 3. Transform sub-model camera poses into Master space
    # C_master = s * R * C_local + t
    # R_master = R_local @ R_sim^T
    for idx, pose in sub_cams.items():
        if idx not in master_cameras:
            R_loc, T_loc = pose["R"], pose["T"]
            C_loc = (-R_loc.T @ T_loc).flatten()
            
            C_mast = scale * (R_sim @ C_loc) + t_sim
            R_mast = R_loc @ R_sim.T
            T_mast = -R_mast @ C_mast.reshape(3, 1)
            
            sub_cams[idx] = {"R": R_mast, "T": T_mast}
            master_cameras[idx] = {"R": R_mast, "T": T_mast}

    # 4. Merge 3D points AND their observations into master_track.
    # Each sub-model's TrackGraph numbers its points from 0, so sub-model IDs collide with
    # master IDs. Every sub point therefore gets a fresh master ID, unless one of its
    # keypoints is already a master observation; then it is the same physical point
    # and its observations are fused into that existing master point.
    sub_observations = {}
    for obs, p3d_id in sub_track.kp_to_point3D.items():
        sub_observations.setdefault(p3d_id, []).append(obs)

    n_new, n_fused = 0, 0
    for p3d_id, pt3d in sub_track.global_points3D.items():
        observations = sub_observations.get(p3d_id, [])
        existing_ids = [master_track.kp_to_point3D[o] for o in observations if o in master_track.kp_to_point3D]

        if existing_ids:
            target_id = existing_ids[0]
            for obs in observations:
                if obs not in master_track.kp_to_point3D:
                    master_track.link_keypoint_to_track(obs[0], obs[1], target_id)
            n_fused += 1
        else:
            master_track.add_point3D(pt3d, observations)
            n_new += 1

    print(f"  -> Merged sub-model points: {n_new} new, {n_fused} fused with existing master points")
    return master_cameras, master_track
