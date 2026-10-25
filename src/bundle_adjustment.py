import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

class BundleAdjuster:
    def __init__(self, K: np.ndarray, device="cuda"):
        # Automatically target the GPU if available
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.K_np = K  # Keep a NumPy copy for the CPU residual check
        self.K = torch.tensor(K, dtype=torch.float32, device=self.device)

    def compute_reprojection_residuals(self, params: np.ndarray, n_points: int, pts1_2d: np.ndarray, pts2_2d: np.ndarray,) -> np.ndarray:
        """
        CPU-based residual computation strictly maintained so your main.py 
        can calculate and print the 'Initial Reprojection RMSE' before optimization.
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

    def optimize_local_window(self, registered_cameras: dict, track_graph, all_kp: dict, window_size: int = 5, max_iter: int = 30,):
        """Runs Local Bundle Adjustment on a sliding window of the last N registered cameras

        and all 3D landmarks visible to them. Updates registered_cameras and
        track_graph in-place.
        """
        registered_ids = list(registered_cameras.keys())
        if len(registered_ids) < 2:
            return

        # Select last N registered cameras for local window
        window_cam_ids = set(registered_ids[-window_size:])
        fixed_cam_id = registered_ids[-window_size]  # Gauge anchor camera

        # 1. Collect all observations for cameras in the sliding window
        obs_cam_ids = []
        obs_p3d_ids = []
        obs_uvs = []
        window_p3d_ids = set()  # MUST BE A SET to use .add()

        for (img_idx, kp_idx), p3d_id in track_graph.kp_to_point3D.items():
            if img_idx in window_cam_ids and p3d_id in track_graph.global_points3D:
                # Extract 2D pixel coordinate (u, v) from keypoints
                uv = all_kp[img_idx][kp_idx].pt
                obs_cam_ids.append(img_idx)
                obs_p3d_ids.append(p3d_id)
                obs_uvs.append(uv)
                window_p3d_ids.add(p3d_id)  # Works correctly on set!

        window_p3d_ids = list(window_p3d_ids)
        if len(window_p3d_ids) == 0 or len(obs_uvs) == 0:
            return

        # 2. Map global IDs to continuous array indices for PyTorch
        cam_id_to_idx = {cid: idx for idx, cid in enumerate(window_cam_ids)}
        p3d_id_to_idx = {pid: idx for idx, pid in enumerate(window_p3d_ids)}

        # Initial 3D landmark array
        pts3D_init = np.array(
            [track_graph.global_points3D[pid] for pid in window_p3d_ids]
        )

        # Initial camera pose parameters (rvec, T)
        cam_rvecs_init, cam_Ts_init = [], []
        for cid in window_cam_ids:
            R = registered_cameras[cid]["R"]
            T = registered_cameras[cid]["T"]
            rvec, _ = cv2.Rodrigues(R)
            cam_rvecs_init.append(rvec.flatten())
            cam_Ts_init.append(T.flatten())

        # 3. Create PyTorch Tensors
        pts3D_t = torch.tensor(
            pts3D_init, dtype=torch.float32, device=self.device, requires_grad=True
        )
        rvecs_t = torch.tensor(
            np.array(cam_rvecs_init),
            dtype=torch.float32,
            device=self.device,
            requires_grad=True,
        )
        Ts_t = torch.tensor(
            np.array(cam_Ts_init),
            dtype=torch.float32,
            device=self.device,
            requires_grad=True,
        )

        # Convert observations to index tensors
        obs_c_idx = torch.tensor(
            [cam_id_to_idx[cid] for cid in obs_cam_ids],
            dtype=torch.long,
            device=self.device,
        )
        obs_p_idx = torch.tensor(
            [p3d_id_to_idx[pid] for pid in obs_p3d_ids],
            dtype=torch.long,
            device=self.device,
        )
        obs_uv_t = torch.tensor(
            np.array(obs_uvs), dtype=torch.float32, device=self.device
        )

        # 4. Optimization via L-BFGS
        optimizer = torch.optim.LBFGS(
            [pts3D_t, rvecs_t, Ts_t], max_iter=max_iter, line_search_fn="strong_wolfe"
        )

    def optimize_2view(self, pts3D_init: np.ndarray, R_init: np.ndarray, T_init: np.ndarray, pts1_2d: np.ndarray, pts2_2d: np.ndarray, max_iter: int = 100):
        """
        Executes non-linear optimization natively on the GPU using PyTorch Autograd.
        Replaces scipy.optimize.least_squares.
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
            theta = torch.norm(rvec)
            if theta > 1e-6:
                r = rvec / theta
                K_mat = torch.zeros((3, 3), device=self.device)
                K_mat[0, 1] = -r[2]; K_mat[0, 2] = r[1]
                K_mat[1, 0] = r[2];  K_mat[1, 2] = -r[0]
                K_mat[2, 0] = -r[1]; K_mat[2, 1] = r[0]
                R = torch.eye(3, device=self.device) + torch.sin(theta)*K_mat + (1-torch.cos(theta))*(K_mat@K_mat)
            else:
                R = torch.eye(3, device=self.device)

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
            
            theta = torch.norm(rvec)
            if theta > 1e-6:
                r = rvec / theta
                K_mat = torch.zeros((3, 3), device=self.device)
                K_mat[0, 1] = -r[2]; K_mat[0, 2] = r[1]
                K_mat[1, 0] = r[2];  K_mat[1, 2] = -r[0]
                K_mat[2, 0] = -r[1]; K_mat[2, 1] = r[0]
                R_opt = torch.eye(3, device=self.device) + torch.sin(theta)*K_mat + (1-torch.cos(theta))*(K_mat@K_mat)
            else:
                R_opt = torch.eye(3, device=self.device)

            pts3D_cam2 = (R_opt @ pts3D.T).T + T.view(3, 1).T
            proj2 = pts3D_cam2 @ self.K.T
            proj2 = proj2[:, :2] / proj2[:, 2:]

            final_loss = F.mse_loss(proj1, pts1_t) + F.mse_loss(proj2, pts2_t)
            opt_rmse = float(torch.sqrt(final_loss).item())

            return pts3D.cpu().numpy(), R_opt.cpu().numpy(), T.cpu().numpy().reshape(3, 1), opt_rmse


    def run_global_bundle_adjustment(
        self,
        registered_cameras: dict,
        track_graph,
        all_kp: dict,
        max_iter: int = 50,
        loss_scale: float = 1.0
    ) -> tuple[dict, object]:
        """
        Executes Global Bundle Adjustment natively on CUDA using PyTorch autograd + L-BFGS.
        Refines ALL registered camera poses and ALL 3D points simultaneously.
        """
        cam_ids = sorted(list(registered_cameras.keys()))
        n_cams = len(cam_ids)
        if n_cams < 2:
            print("Global BA skipped: Less than 2 cameras registered.")
            return registered_cameras, track_graph

        cam_id_to_idx = {cid: idx for idx, cid in enumerate(cam_ids)}

        # 1. Collect all valid observations directly from TrackGraph & all_kp
        obs_cam_ids = []
        obs_p3d_ids = []
        obs_uvs = []
        global_p3d_ids = set()

        for (cam_id, kp_idx), p3d_id in track_graph.kp_to_point3D.items():
            if cam_id in cam_id_to_idx and p3d_id in track_graph.global_points3D:
                uv = all_kp[cam_id][kp_idx].pt  # Extract (u, v) pixel coordinate
                obs_cam_ids.append(cam_id)
                obs_p3d_ids.append(p3d_id)
                obs_uvs.append(uv)
                global_p3d_ids.add(p3d_id)

        p3d_id_list = sorted(list(global_p3d_ids))
        p3d_id_to_idx = {pid: idx for idx, pid in enumerate(p3d_id_list)}
        n_points = len(p3d_id_list)

        if len(obs_uvs) == 0:
            print("Global BA skipped: No active observations found.")
            return registered_cameras, track_graph

        # 2. Extract initial 3D point positions and camera parameters
        pts3D_init = np.array([track_graph.global_points3D[pid] for pid in p3d_id_list])
        
        cam_rvecs_init, cam_Ts_init = [], []
        for cid in cam_ids:
            R = registered_cameras[cid]["R"]
            T = registered_cameras[cid]["T"]
            rvec, _ = cv2.Rodrigues(R)
            cam_rvecs_init.append(rvec.flatten())
            cam_Ts_init.append(T.flatten())

        # 3. Create GPU Tensors with gradients enabled
        pts3D_t = torch.tensor(pts3D_init, dtype=torch.float32, device=self.device, requires_grad=True)
        rvecs_t = torch.tensor(np.array(cam_rvecs_init), dtype=torch.float32, device=self.device, requires_grad=True)
        Ts_t = torch.tensor(np.array(cam_Ts_init), dtype=torch.float32, device=self.device, requires_grad=True)

        # Observation index tensors
        obs_c_idx = torch.tensor([cam_id_to_idx[cid] for cid in obs_cam_ids], dtype=torch.long, device=self.device)
        obs_p_idx = torch.tensor([p3d_id_to_idx[pid] for pid in obs_p3d_ids], dtype=torch.long, device=self.device)
        obs_uv_t = torch.tensor(np.array(obs_uvs), dtype=torch.float32, device=self.device)

        # Anchor gauge freedom: Keep Camera 0 (or Sub-Model 1's origin) fixed
        fixed_cam_idx = 0

        # 4. Initialize PyTorch GPU L-BFGS Optimizer
        optimizer = torch.optim.LBFGS([pts3D_t, rvecs_t, Ts_t], max_iter=max_iter, line_search_fn="strong_wolfe")

        print(f"Running Global GPU BA on {n_cams} Cameras | {n_points} Points | {len(obs_uvs)} Observations...")