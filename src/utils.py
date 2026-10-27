import os
import copy
import itertools
import numpy as np
import cv2
import matplotlib.pyplot as plt

from triangulation import Triangulator, camera_center


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
    max_passes: int = 3, max_reproj_px: float = 4.0, min_angle_deg: float = 1.5,
    max_depth_factor: float = 10.0, max_step_factor: float = 10.0, min_inliers: int = 15
):
    """
    Runs multi-pass incremental PnP registration, DLT triangulation, and sliding window
    Local BA over a target list of image indices for a sub-model.

    All guardrails are scale free:
        max_reproj_px    : reprojection error in pixels (PnP RANSAC + new points + track linking)
        min_angle_deg    : minimum triangulation angle between the two viewing rays
        max_depth_factor : new points deeper than this * median depth of the pair are dropped
        max_step_factor  : a new camera must lie within this * median camera spacing
                           of its nearest registered camera (catches wild PnP poses)
    """
    unregistered = set(image_indices) - set(registered_cameras.keys())

    def match_kp_indices(curr_idx, reg_idx, m):
        """(kp index in curr image, kp index in reg image) of a match; match_graph keys are (low, high)."""
        return (m.queryIdx, m.trainIdx) if curr_idx < reg_idx else (m.trainIdx, m.queryIdx)

    for pass_num in range(max_passes):
        registered_this_pass = 0

        # Process unregistered pool for this sub-model
        for curr_idx in sorted(list(unregistered)):
            kp_curr = all_kp[curr_idx]

            # 1. Gather 2D-to-3D correspondences from currently registered cameras via TrackGraph.
            # No 3D filter needed here: map points were validated when they were triangulated,
            # remaining outliers are rejected by RANSAC.
            pts2D, pts3D, p3d_ids_used, seen = [], [], [], set()
            for reg_idx in registered_cameras:
                pair = (min(curr_idx, reg_idx), max(curr_idx, reg_idx))
                for m in match_graph.get(pair, []):
                    kp_curr_idx, kp_reg_idx = match_kp_indices(curr_idx, reg_idx, m)
                    p3d_id = track_graph.get_point3D_id(reg_idx, kp_reg_idx)
                    if p3d_id is not None and (kp_curr_idx, p3d_id) not in seen:
                        seen.add((kp_curr_idx, p3d_id))
                        pts3D.append(track_graph.global_points3D[p3d_id])
                        pts2D.append(kp_curr[kp_curr_idx].pt)
                        p3d_ids_used.append((kp_curr_idx, p3d_id))

            if len(pts2D) < min_inliers:
                continue

            # 2. Estimate Camera Pose via PnP RANSAC (inlier = reprojection error <= max_reproj_px)
            pts3D, pts2D = np.float64(pts3D), np.float64(pts2D)
            success, rvec, tvec, pnp_inliers = cv2.solvePnPRansac(
                pts3D, pts2D, K, distCoeffs=None,
                flags=cv2.SOLVEPNP_ITERATIVE, reprojectionError=max_reproj_px, confidence=0.99
            )
            if not success or pnp_inliers is None or len(pnp_inliers) < min_inliers:
                continue

            R_curr, _ = cv2.Rodrigues(rvec)
            T_curr = tvec.reshape(3, 1)
            C_curr = camera_center(R_curr, T_curr)
            inl = pnp_inliers.flatten()

            # Cheirality of the pose: the inlier points must lie in front of the new camera.
            # A point behind the camera can still project into the image (mirrored), so RANSAC alone misses it.
            if np.mean(triangulator.point_depths(pts3D[inl], R_curr, T_curr) > 0) < 0.9:
                continue

            # Motion jump guardrail, relative to the typical camera spacing of this sub-model
            C_reg = camera_centers(registered_cameras, list(registered_cameras))
            if len(C_reg) >= 3:
                pairwise = np.linalg.norm(C_reg[:, None] - C_reg[None], axis=2)
                np.fill_diagonal(pairwise, np.inf)
                typical_spacing = np.median(pairwise.min(axis=1))
                if np.linalg.norm(C_reg - C_curr, axis=1).min() > max_step_factor * typical_spacing:
                    continue

            # Pose accepted! Register camera & link PnP inliers to TrackGraph
            registered_cameras[curr_idx] = {"R": R_curr, "T": T_curr}
            unregistered.remove(curr_idx)
            registered_this_pass += 1

            for idx in inl:
                kp_curr_idx, p3d_id = p3d_ids_used[idx]
                if track_graph.get_point3D_id(curr_idx, kp_curr_idx) is None:
                    track_graph.link_keypoint_to_track(curr_idx, kp_curr_idx, p3d_id)

            # 3. DLT Triangulation to Expand Map
            P_curr = triangulator.projection_matrix(R_curr, T_curr)

            for reg_idx in list(registered_cameras.keys()):
                pair = (min(curr_idx, reg_idx), max(curr_idx, reg_idx))
                if reg_idx == curr_idx or pair not in match_graph:
                    continue

                matches = match_graph[pair]
                R_prev = registered_cameras[reg_idx]["R"]
                T_prev = registered_cameras[reg_idx]["T"]
                P_prev = triangulator.projection_matrix(R_prev, T_prev)

                # Extract matching 2D coordinates
                if curr_idx < reg_idx:
                    pts_curr, pts_prev = estimator.extract_match_coordinates(all_kp[curr_idx], all_kp[reg_idx], matches)
                else:
                    pts_prev, pts_curr = estimator.extract_match_coordinates(all_kp[reg_idx], all_kp[curr_idx], matches)

                raw_pts3D = triangulator.triangulate_points(P_prev, P_curr, pts_prev, pts_curr)
                valid = triangulator.filter_triangulated_points(
                    raw_pts3D, R_prev, T_prev, pts_prev, R_curr, T_curr, pts_curr,
                    max_reproj_px=max_reproj_px, min_angle_deg=min_angle_deg, max_depth_factor=max_depth_factor,
                )

                for i, m in enumerate(matches):
                    kp_curr_i, kp_reg_i = match_kp_indices(curr_idx, reg_idx, m)
                    id_reg = track_graph.get_point3D_id(reg_idx, kp_reg_i)
                    id_curr = track_graph.get_point3D_id(curr_idx, kp_curr_i)

                    if id_reg is None and id_curr is None:
                        # Brand-new point: must pass all scale-free checks
                        if valid[i]:
                            track_graph.add_point3D(raw_pts3D[i], observations=[(reg_idx, kp_reg_i), (curr_idx, kp_curr_i)])
                    elif id_curr is None:
                        # Extend an existing track to the new image, if the existing point reprojects there
                        X = np.asarray(track_graph.global_points3D[id_reg]).reshape(1, 3)
                        if (triangulator.point_depths(X, R_curr, T_curr)[0] > 0 and
                                triangulator.reprojection_errors(X, R_curr, T_curr, pts_curr[i:i + 1])[0] <= max_reproj_px):
                            track_graph.link_keypoint_to_track(curr_idx, kp_curr_i, id_reg)
                    elif id_reg is None:
                        # Extend the new image's track back to the registered image
                        X = np.asarray(track_graph.global_points3D[id_curr]).reshape(1, 3)
                        if (triangulator.point_depths(X, R_prev, T_prev)[0] > 0 and
                                triangulator.reprojection_errors(X, R_prev, T_prev, pts_prev[i:i + 1])[0] <= max_reproj_px):
                            track_graph.link_keypoint_to_track(reg_idx, kp_reg_i, id_curr)
                    # Both keypoints already belong to tracks: leave them as they are

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




def camera_centers(cams: dict, ids) -> np.ndarray:
    """(N, 3) optical centers of the cameras `ids` in a {id: {"R", "T"}} dict."""
    return np.array([camera_center(cams[i]["R"], cams[i]["T"]) for i in ids], dtype=np.float64).reshape(-1, 3)


def ransac_umeyama_sim3(source_pts: np.ndarray, target_pts: np.ndarray, inlier_frac: float = 0.03,
                        n_samples: int = 500, seed: int = 0):
    """
    Robust umeyama_sim3: RANSAC over minimal 3-point fits, then a final fit on all inliers.
    Inlier threshold = inlier_frac * median distance of the target points to their centroid.

    Returns:
        (scale, R, t), inlier mask (N,)
    """
    n = len(source_pts)
    thr = inlier_frac * np.median(np.linalg.norm(target_pts - target_pts.mean(axis=0), axis=1))

    def errors(idx):
        s, R, t = umeyama_sim3(source_pts[idx], target_pts[idx])
        return np.linalg.norm((s * (R @ source_pts.T)).T + t - target_pts, axis=1)

    if n * (n - 1) * (n - 2) // 6 <= n_samples:
        samples = list(itertools.combinations(range(n), 3))
    else:
        rng = np.random.default_rng(seed)
        samples = [rng.choice(n, 3, replace=False) for _ in range(n_samples)]

    best, best_count = np.ones(n, dtype=bool), -1
    for idx in samples:
        inl = errors(list(idx)) <= thr
        if inl.sum() > best_count:
            best, best_count = inl, inl.sum()
    if best.sum() < 3:
        best[:] = True

    # Refine: refit on inliers and re-select inliers once
    best = errors(np.where(best)[0]) <= thr
    if best.sum() < 3:
        best[:] = True
    return umeyama_sim3(source_pts[best], target_pts[best]), best


def find_shared_points3D(master_track, sub_track) -> tuple[np.ndarray, np.ndarray]:
    """3D-3D correspondences: sub-model points with a keypoint that is already a master observation."""
    pairs = {}
    for obs, p_sub in sub_track.kp_to_point3D.items():
        p_mst = master_track.kp_to_point3D.get(obs)
        if p_mst is not None and p_sub in sub_track.global_points3D and p_mst in master_track.global_points3D:
            pairs.setdefault(p_sub, p_mst)
    src = np.array([np.asarray(sub_track.global_points3D[p]).ravel() for p in pairs], dtype=np.float64)
    tgt = np.array([np.asarray(master_track.global_points3D[p]).ravel() for p in pairs.values()], dtype=np.float64)
    return src.reshape(-1, 3), tgt.reshape(-1, 3)


def solve_submodel_sim3(master_cams, master_track, sub_cams, sub_track, min_points: int = 50, min_cams: int = 3,
                        stats: dict = None):
    """
    Sim(3) that maps the sub-model into master space (Target = s * R * Source + t).
    Prefers the shared 3D points (thousands of correspondences, well conditioned); falls back to
    the shared camera centers, which are often nearly collinear and leave the rotation weakly
    constrained. Returns None if there is not enough overlap.
    If a `stats` dict is passed, the alignment numbers are written into it (for analysis).
    """
    stats = {} if stats is None else stats
    shared_ids = sorted(set(master_cams) & set(sub_cams))
    src_pts, tgt_pts = find_shared_points3D(master_track, sub_track)
    stats.update(shared_cameras=len(shared_ids), shared_points=len(src_pts))

    if len(src_pts) >= min_points:
        sim3, inl = ransac_umeyama_sim3(src_pts, tgt_pts, inlier_frac=0.03)
        stats.update(method="3D points", inliers=int(inl.sum()), candidates=len(inl))
        print(f"  -> Umeyama on shared 3D points: {int(inl.sum())}/{len(inl)} inliers")
    elif len(shared_ids) >= min_cams:
        sim3, inl = ransac_umeyama_sim3(camera_centers(sub_cams, shared_ids),
                                        camera_centers(master_cams, shared_ids), inlier_frac=0.05)
        stats.update(method="camera centers", inliers=int(inl.sum()), candidates=len(inl))
        print(f"  -> Umeyama on shared camera centers {shared_ids}: {int(inl.sum())}/{len(inl)} inliers")
    else:
        return None

    # Sanity check: how well do the shared camera centers agree after the transform?
    if shared_ids:
        s, R, t = sim3
        err = np.linalg.norm((s * (R @ camera_centers(sub_cams, shared_ids).T)).T + t
                             - camera_centers(master_cams, shared_ids), axis=1)
        stats.update(cam_err_median=float(np.median(err)), cam_err_max=float(err.max()))
        print(f"  -> {len(shared_ids)} shared cameras, center error after alignment: "
              f"median {np.median(err):.4f}, max {err.max():.4f}")
    stats.update(scale=float(sim3[0]))
    return sim3


# Helper function to align and merge a sub-model into the Master Model
def align_and_merge(master_cams, master_tck, sub_cams, sub_track, stats: dict = None):
    """
    Aligns a sub-model onto the master model via a robust Umeyama Sim(3) (solve_submodel_sim3)
    and merges its cameras + 3D points. Edits sub_cams / sub_track / master_tck in place.
    Returns (master_cameras, master_track), or None if the overlap is too small to align.
    If a `stats` dict is passed, alignment and merge numbers are written into it (for analysis).
    """
    stats = {} if stats is None else stats
    master_cameras = {**master_cams}
    master_track = master_tck

    # 1. Compute Sim(3) parameters (Scale, Rotation, Translation)
    sim3 = solve_submodel_sim3(master_cameras, master_track, sub_cams, sub_track, stats=stats)
    if sim3 is None:
        return None
    scale, R_sim, t_sim = sim3
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
            C_loc = camera_center(R_loc, T_loc)
            
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

    stats.update(new_points=n_new, fused_points=n_fused)
    print(f"  -> Merged sub-model points: {n_new} new, {n_fused} fused with existing master points")
    return master_cameras, master_track


def merge_submodels(submodels: dict, reference: str, stats: list = None):
    """
    Merges named sub-models {name: (cameras, track_graph)} into one master model,
    in the coordinate frame of `reference`. Greedy: always merges the sub-model with
    the most overlap (shared cameras, shared observations) with the current master next.
    Works on deep copies, the passed sub-models stay untouched.
    If a `stats` list is passed, one dict per merge step is appended (merge order, alignment numbers).

    Returns:
        master_cameras, master_track, cam_source ({cam_id: name of the sub-model it came from})
    """
    remaining = {name: copy.deepcopy(model) for name, model in submodels.items()}
    master_cameras, master_track = remaining.pop(reference)
    cam_source = {cid: reference for cid in master_cameras}

    def overlap(name):
        cams, track = remaining[name]
        return (len(set(cams) & set(master_cameras)),
                sum(o in master_track.kp_to_point3D for o in track.kp_to_point3D))

    while remaining:
        name = max(remaining, key=overlap)
        sub_cams, sub_track = remaining.pop(name)
        print(f"\nMerging '{name}' sub-model ({len(sub_cams)} cams, {len(sub_track.global_points3D)} pts)")

        new_cam_ids = set(sub_cams) - set(master_cameras)
        step = {"name": name, "cameras": len(sub_cams), "points": len(sub_track.global_points3D),
                "new_cameras": len(new_cam_ids)}
        merged = align_and_merge(master_cameras, master_track, sub_cams, sub_track, stats=step)
        if stats is not None:
            stats.append(step)
        if merged is None:
            print(f"  -> Not enough overlap with the master model. Could not merge: {[name, *remaining]}")
            break
        master_cameras, master_track = merged
        cam_source.update({cid: name for cid in new_cam_ids})

    print(f"\nMaster model: {len(master_cameras)} cameras | {len(master_track.global_points3D)} 3D points | "
          f"{len(master_track.kp_to_point3D)} observations")
    return master_cameras, master_track, cam_source


def sample_point_colors(track_graph, all_kp: dict, dataset) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns the (N, 3) points of the track graph and their (N, 3) uint8 RGB colors,
    sampled at the keypoint of the first image that observes each point.
    """
    point_ids = list(track_graph.global_points3D.keys())
    points = np.array([np.asarray(track_graph.global_points3D[p]).ravel() for p in point_ids]).reshape(-1, 3)

    first_obs = {}
    for (cam_id, kp_idx), p3d_id in track_graph.kp_to_point3D.items():
        if p3d_id in track_graph.global_points3D:
            first_obs.setdefault(p3d_id, (cam_id, kp_idx))

    rows_by_cam = {}
    for row, p3d_id in enumerate(point_ids):
        if p3d_id in first_obs:
            cam_id, kp_idx = first_obs[p3d_id]
            rows_by_cam.setdefault(cam_id, []).append((row, kp_idx))

    colors = np.zeros((len(point_ids), 3), dtype=np.uint8)
    for cam_id, rows in rows_by_cam.items():   # load each image only once
        img = dataset[cam_id]
        h, w = img.shape[:2]
        for row, kp_idx in rows:
            u, v = all_kp[cam_id][kp_idx].pt
            b, g, r = img[min(int(round(v)), h - 1), min(int(round(u)), w - 1)]
            colors[row] = (r, g, b)
    return points, colors


def point_viewing_centers(track_graph, cameras: dict) -> np.ndarray:
    """
    (N, 3) mean optical center of the cameras observing each point, in the same order as
    sample_point_colors / track_graph.global_points3D. Used to orient surface normals
    towards the side the point was seen from.
    """
    sums, counts = {}, {}
    for (cam_id, _), p3d_id in track_graph.kp_to_point3D.items():
        if cam_id in cameras and p3d_id in track_graph.global_points3D:
            C = camera_center(cameras[cam_id]["R"], cameras[cam_id]["T"])
            sums[p3d_id] = sums.get(p3d_id, 0) + C
            counts[p3d_id] = counts.get(p3d_id, 0) + 1
    fallback = camera_centers(cameras, list(cameras)).mean(axis=0)
    return np.array([sums[p] / counts[p] if p in counts else fallback
                     for p in track_graph.global_points3D]).reshape(-1, 3)


def point_quality(track_graph, cameras: dict, all_kp: dict, triangulator) -> dict:
    """
    Per-point quality measures, in the same order as sample_point_colors / track_graph.global_points3D:
        track_length : number of registered images observing the point
        max_angle    : largest triangulation angle (deg) between any two of its viewing rays
        reproj_err   : mean reprojection error (px) over its observations

    Why these matter: a point seen in only 2 views has no redundancy, an error along the epipolar
    line moves it in depth undetected. A 3rd view verifies it, and more views average the noise.
    """
    point_ids = list(track_graph.global_points3D.keys())
    row = {p: i for i, p in enumerate(point_ids)}
    X = np.array([np.asarray(track_graph.global_points3D[p]).ravel() for p in point_ids]).reshape(-1, 3)

    n = len(point_ids)
    obs, errors, _ = triangulator.track_reprojection_errors(cameras, track_graph, all_kp)
    rows = np.array([row[p] for _, p in obs], dtype=int)
    # bincount sums repeated rows correctly (a point can be linked to 2 keypoints of the same image)
    track_length = np.bincount(rows, minlength=n)
    err_sum = np.bincount(rows, weights=errors, minlength=n)

    centers = {c: camera_center(v["R"], v["T"]) for c, v in cameras.items()}
    d = X[rows] - np.array([centers[c] for (c, _), _ in obs]).reshape(-1, 3)
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    rays = [[] for _ in range(n)]
    for r, ray in zip(rows, d):
        rays[r].append(ray)

    max_angle = np.zeros(n)
    for i, r in enumerate(rays):
        if len(r) >= 2:
            r = np.array(r)
            max_angle[i] = np.degrees(np.arccos(np.clip((r @ r.T).min(), -1.0, 1.0)))

    reproj_err = np.where(track_length > 0, err_sum / np.maximum(track_length, 1), np.inf)
    return {"track_length": track_length, "max_angle": max_angle, "reproj_err": reproj_err}


def point_quality_mask(quality: dict, min_track_length: int = 3, min_angle_deg: float = 5.0,
                       max_reproj_px: float = 2.0) -> np.ndarray:
    """Boolean mask of points that pass all three quality thresholds (see point_quality)."""
    mask = ((quality["track_length"] >= min_track_length) & (quality["max_angle"] >= min_angle_deg)
            & (quality["reproj_err"] <= max_reproj_px))
    print(f"Point quality filter (track >= {min_track_length}, angle >= {min_angle_deg} deg, "
          f"reproj <= {max_reproj_px} px): keeps {mask.sum()} / {len(mask)} points ({mask.mean():.0%}) | "
          f"seen in only 2 views: {np.mean(quality['track_length'] == 2):.0%}")
    return mask
