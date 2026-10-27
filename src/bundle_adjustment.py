import cv2
import numpy as np
import torch
import torch.nn.functional as F

from triangulation import Triangulator

class BundleAdjuster:
    def __init__(self, K: np.ndarray, device="cuda"):
        # Automatically target the GPU if available
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.K_np = K  # Keep a NumPy copy for the CPU residual check
        self.K = torch.tensor(K, dtype=torch.float32, device=self.device)

    def compute_reprojection_residuals(self, params: np.ndarray, n_points: int, pts1_2d: np.ndarray, pts2_2d: np.ndarray,) -> np.ndarray:
        """
        CPU residuals of the two-view problem (camera 1 = [I | 0], camera 2 = (R, T)),
        used to report the reprojection RMSE before optimization.
        """
        rvec = params[0:3]
        T = params[3:6].reshape(3, 1)
        pts3D = params[6:].reshape(n_points, 3)

        R, _ = cv2.Rodrigues(rvec)

        # Camera 1 projection (origin: P1 = K [I | 0])
        proj1 = (self.K_np @ pts3D.T).T
        pts1_proj = proj1[:, :2] / proj1[:, 2:]

        # Camera 2 projection (P2 = K [R | T])
        proj2 = (self.K_np @ (R @ pts3D.T + T)).T
        pts2_proj = proj2[:, :2] / proj2[:, 2:]

        # Calculate pixel residuals
        res1 = (pts1_proj - pts1_2d).ravel()
        res2 = (pts2_proj - pts2_2d).ravel()

        return np.hstack([res1, res2])

    @staticmethod
    def _rodrigues_batch(rvecs: torch.Tensor) -> torch.Tensor:
        """Converts (N, 3) axis-angle vectors to (N, 3, 3) rotation matrices (differentiable)."""
        theta = rvecs.norm(dim=1, keepdim=True).clamp_min(1e-8)
        k = rvecs / theta
        z = torch.zeros_like(k[:, 0])
        Kx = torch.stack([z, -k[:, 2], k[:, 1],
                          k[:, 2], z, -k[:, 0],
                          -k[:, 1], k[:, 0], z], dim=1).view(-1, 3, 3)
        s = torch.sin(theta)[..., None]
        c = torch.cos(theta)[..., None]
        I = torch.eye(3, dtype=rvecs.dtype, device=rvecs.device).expand_as(Kx)
        return I + s * Kx + (1 - c) * (Kx @ Kx)

    def _run_ba(self, registered_cameras: dict, track_graph, all_kp: dict, cam_ids: list, fixed_cam_ids: set,
                max_iter: int, loss_scale: float, label: str):
        """
        Shared BA solver. Optimizes the poses of `cam_ids` (except `fixed_cam_ids`) and every 3D point
        they observe, using a Huber loss with threshold `loss_scale` (pixels).
        Updates registered_cameras and track_graph in place.
        Returns (rmse_before, rmse_after) in pixels, or None if there was nothing to optimize.
        """
        cam_id_to_idx = {cid: idx for idx, cid in enumerate(cam_ids)}

        # 1. Collect observations of the selected cameras from the TrackGraph
        obs_cam_ids, obs_p3d_ids, obs_uvs = [], [], []
        p3d_ids = set()
        for (cam_id, kp_idx), p3d_id in track_graph.kp_to_point3D.items():
            if cam_id in cam_id_to_idx and p3d_id in track_graph.global_points3D:
                obs_cam_ids.append(cam_id)
                obs_p3d_ids.append(p3d_id)
                obs_uvs.append(all_kp[cam_id][kp_idx].pt)
                p3d_ids.add(p3d_id)

        if len(obs_uvs) == 0:
            print(f"{label} skipped: No active observations found.")
            return None

        p3d_id_list = sorted(p3d_ids)
        p3d_id_to_idx = {pid: idx for idx, pid in enumerate(p3d_id_list)}

        # 2. Initial parameters
        pts3D_init = np.array([np.asarray(track_graph.global_points3D[pid]).ravel() for pid in p3d_id_list])
        rvecs_init, Ts_init = [], []
        for cid in cam_ids:
            rvec, _ = cv2.Rodrigues(np.asarray(registered_cameras[cid]["R"], dtype=np.float64))
            rvecs_init.append(rvec.ravel())
            Ts_init.append(np.asarray(registered_cameras[cid]["T"]).ravel())

        # 3. Tensors
        pts3D_t = torch.tensor(pts3D_init, dtype=torch.float32, device=self.device, requires_grad=True)
        rvecs_t = torch.tensor(np.array(rvecs_init), dtype=torch.float32, device=self.device, requires_grad=True)
        Ts_t = torch.tensor(np.array(Ts_init), dtype=torch.float32, device=self.device, requires_grad=True)

        obs_c_idx = torch.tensor([cam_id_to_idx[c] for c in obs_cam_ids], dtype=torch.long, device=self.device)
        obs_p_idx = torch.tensor([p3d_id_to_idx[p] for p in obs_p3d_ids], dtype=torch.long, device=self.device)
        obs_uv_t = torch.tensor(np.array(obs_uvs), dtype=torch.float32, device=self.device)

        # Gauge anchor: these cameras get zero gradient, so L-BFGS never moves them
        fixed_mask = torch.tensor([cid in fixed_cam_ids for cid in cam_ids], dtype=torch.bool, device=self.device)

        def project():
            R = self._rodrigues_batch(rvecs_t)[obs_c_idx]                       # (M, 3, 3)
            Xc = (R @ pts3D_t[obs_p_idx].unsqueeze(-1)).squeeze(-1) + Ts_t[obs_c_idx]
            proj = Xc @ self.K.T
            return proj[:, :2] / proj[:, 2:].clamp_min(1e-6)                    # (M, 2)

        def rmse():
            with torch.no_grad():
                return float(torch.sqrt(((project() - obs_uv_t) ** 2).mean()).item())  # same convention as utils.compute_rmse

        rmse_before = rmse()

        # 4. Optimize with L-BFGS on a robust (Huber) reprojection loss
        optimizer = torch.optim.LBFGS([pts3D_t, rvecs_t, Ts_t], max_iter=max_iter, line_search_fn="strong_wolfe")

        def closure():
            optimizer.zero_grad()
            loss = F.huber_loss(project(), obs_uv_t, delta=loss_scale)
            loss.backward()
            rvecs_t.grad[fixed_mask] = 0
            Ts_t.grad[fixed_mask] = 0
            return loss

        optimizer.step(closure)
        rmse_after = rmse()

        # 5. Write results back
        rvecs_opt = rvecs_t.detach().cpu().numpy().astype(np.float64)
        Ts_opt = Ts_t.detach().cpu().numpy().astype(np.float64)
        pts_opt = pts3D_t.detach().cpu().numpy().astype(np.float64)
        for cid, i in cam_id_to_idx.items():
            if cid in fixed_cam_ids:
                continue
            registered_cameras[cid]["R"] = cv2.Rodrigues(rvecs_opt[i])[0]
            registered_cameras[cid]["T"] = Ts_opt[i].reshape(3, 1)
        for pid, i in p3d_id_to_idx.items():
            track_graph.global_points3D[pid] = pts_opt[i]

        print(f"{label}: {len(cam_ids)} cams | {len(p3d_id_list)} pts | {len(obs_uvs)} obs | "
              f"RMSE {rmse_before:.3f} -> {rmse_after:.3f} px")
        return rmse_before, rmse_after

    def optimize_local_window(self, registered_cameras: dict, track_graph, all_kp: dict, window_size: int = 5,
                              max_iter: int = 30, loss_scale: float = 2.0):
        """
        Runs Local Bundle Adjustment on a sliding window of the last N registered cameras
        and all 3D landmarks visible to them. The oldest camera in the window is held fixed.
        Updates registered_cameras and track_graph in-place.
        """
        registered_ids = list(registered_cameras.keys())  # insertion order = registration order
        if len(registered_ids) < 2:
            return None

        window_cam_ids = registered_ids[-min(window_size, len(registered_ids)):]
        fixed_cam_ids = {window_cam_ids[0]}
        return self._run_ba(registered_cameras, track_graph, all_kp, window_cam_ids, fixed_cam_ids,
                            max_iter, loss_scale, label="Local BA")

    def optimize_2view(self, pts3D_init: np.ndarray, R_init: np.ndarray, T_init: np.ndarray, pts1_2d: np.ndarray, pts2_2d: np.ndarray, max_iter: int = 100):
        """
        Two-view bundle adjustment of the seed pair on the GPU (PyTorch autograd + L-BFGS):
        camera 1 stays at [I | 0], camera 2 and all points are optimized.
        """
        # 1. Convert NumPy arrays to GPU Tensors and enable gradient tracking
        pts3D = torch.tensor(pts3D_init, dtype=torch.float32, device=self.device, requires_grad=True)
        
        rvec_init, _ = cv2.Rodrigues(R_init)
        rvec = torch.tensor(rvec_init.ravel(), dtype=torch.float32, device=self.device, requires_grad=True)
        T = torch.tensor(T_init.ravel(), dtype=torch.float32, device=self.device, requires_grad=True)

        pts1_t = torch.tensor(pts1_2d, dtype=torch.float32, device=self.device)
        pts2_t = torch.tensor(pts2_2d, dtype=torch.float32, device=self.device)

        # 2. Initialize GPU Optimizer (L-BFGS handles smooth deterministic losses well)
        optimizer = torch.optim.LBFGS([pts3D, rvec, T], max_iter=max_iter, line_search_fn="strong_wolfe")

        def closure():
            optimizer.zero_grad()
            
            # Camera 1 Reprojection 
            proj1 = pts3D @ self.K.T
            proj1 = proj1[:, :2] / proj1[:, 2:]
            
            # Camera 2 Reprojection (GPU Rodrigues Conversion)
            R = self._rodrigues_batch(rvec[None])[0]

            pts3D_cam2 = (R @ pts3D.T).T + T
            proj2 = pts3D_cam2 @ self.K.T
            proj2 = proj2[:, :2] / proj2[:, 2:]

            # Compute Loss
            loss = F.mse_loss(proj1, pts1_t) + F.mse_loss(proj2, pts2_t)
            loss.backward()
            return loss

        # 3. Execute Optimization Step on CUDA
        optimizer.step(closure)

        # 4. Extract optimized data back to CPU NumPy arrays
        with torch.no_grad():
            # Compute final reprojection loss forward pass without backward()
            proj1 = pts3D @ self.K.T
            proj1 = proj1[:, :2] / proj1[:, 2:]
            
            R_opt = self._rodrigues_batch(rvec[None])[0]

            pts3D_cam2 = (R_opt @ pts3D.T).T + T.view(3, 1).T
            proj2 = pts3D_cam2 @ self.K.T
            proj2 = proj2[:, :2] / proj2[:, 2:]

            final_loss = F.mse_loss(proj1, pts1_t) + F.mse_loss(proj2, pts2_t)
            opt_rmse = float(torch.sqrt(final_loss / 2).item())  # mean over both views, same as utils.compute_rmse

            return pts3D.cpu().numpy(), R_opt.cpu().numpy(), T.cpu().numpy().reshape(3, 1), opt_rmse


    def run_global_bundle_adjustment(
        self,
        registered_cameras: dict,
        track_graph,
        all_kp: dict,
        max_iter: int = 200,
        loss_scale: float = 1.0,
        fixed_cam_id=None,
        outlier_thresholds_px=None,
    ) -> tuple[dict, object]:
        """
        Executes Global Bundle Adjustment using PyTorch autograd + L-BFGS.
        Refines ALL registered camera poses and ALL 3D points simultaneously.

        Parameters:
            loss_scale (float): Huber threshold in pixels. Residuals larger than this are
                                down-weighted (linear instead of quadratic), which limits outlier impact.
            fixed_cam_id: Camera held fixed to anchor the gauge (default: lowest camera id).
            outlier_thresholds_px: Optional list, e.g. [20, 4]. Runs one round per value:
                                   filter_reprojection_outliers(value) followed by a BA.
                                   None = a single BA without filtering.
        """
        cam_ids = sorted(registered_cameras.keys())
        if len(cam_ids) < 2:
            print("Global BA skipped: Less than 2 cameras registered.")
            return registered_cameras, track_graph

        if fixed_cam_id is None:
            fixed_cam_id = cam_ids[0]
        for max_err_px in (outlier_thresholds_px or [None]):
            if max_err_px is not None:
                self.filter_reprojection_outliers(registered_cameras, track_graph, all_kp, max_err_px)
            self._run_ba(registered_cameras, track_graph, all_kp, cam_ids, {fixed_cam_id},
                         max_iter, loss_scale, label="Global BA")
        return registered_cameras, track_graph

    def filter_reprojection_outliers(self, registered_cameras: dict, track_graph, all_kp: dict, max_err_px: float):
        """
        Removes observations with reprojection error > max_err_px or behind the camera,
        plus observations of unregistered cameras / deleted points. Then drops 3D points
        left with fewer than 2 observations. Edits track_graph in place.
        """
        obs, err, depth = Triangulator(self.K_np).track_reprojection_errors(registered_cameras, track_graph, all_kp)
        n_dangling = len(track_graph.kp_to_point3D) - len(obs)
        if not obs:
            return
        good = (depth > 1e-6) & (err <= max_err_px)

        kept = {o: p for (o, p), g in zip(obs, good) if g}
        obs_count = {}
        for p in kept.values():
            obs_count[p] = obs_count.get(p, 0) + 1

        n_pts_before = len(track_graph.global_points3D)
        track_graph.kp_to_point3D = {o: p for o, p in kept.items() if obs_count[p] >= 2}
        track_graph.global_points3D = {p: x for p, x in track_graph.global_points3D.items()
                                       if obs_count.get(p, 0) >= 2}

        print(f"Outlier filter (> {max_err_px:.1f} px): removed {int((~good).sum()) + n_dangling} obs, "
              f"{n_pts_before - len(track_graph.global_points3D)} pts -> "
              f"{len(track_graph.global_points3D)} pts | {len(track_graph.kp_to_point3D)} obs remain")
