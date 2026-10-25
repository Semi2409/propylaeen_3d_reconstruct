import cv2
import numpy as np
import torch
import torch.nn.functional as F

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

            # Compute Los
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