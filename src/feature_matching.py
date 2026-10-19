import cv2
import numpy as np
import os
from pathlib import Path
from dataset import ImageDataset
import torch



class FeatureMatcher:
    def __init__(self, ratio_threshold=0.75, device = "cuda"):
        self.detector = cv2.SIFT_create()
        self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        self.ratio = ratio_threshold
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

    def extract_features(self, image):
        """
        Detect keypoints and compute descriptors for a given image.
        
        Parameters:
            image (numpy.ndarray): Input BGR or Grayscale image array.
            
        Returns:
            keypoints (tuple): Detected OpenCV KeyPoint objects.
            descriptors (numpy.ndarray): Feature descriptors matrix (e.g. SIFT vectors).
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.detector.detectAndCompute(
            gray, None
        )
        return keypoints, descriptors


    def match_and_filter_gpu(self, des1, des2, chunk_size=30000):
        """
        Computes L2 distances and applies Lowe's ratio test on the GPU.
        Automatically chunks large target descriptor matrices (des2) to prevent CUDA OOM.
        """
        if des1 is None or des2 is None or len(des1) == 0 or len(des2) == 0:
            return []

        t_des1 = des1 if isinstance(des1, torch.Tensor) else torch.tensor(des1, dtype=torch.float32, device=self.device)
        
        # Single-pass for small descriptor sets
        if len(des2) <= chunk_size:
            t_des2 = des2 if isinstance(des2, torch.Tensor) else torch.tensor(des2, dtype=torch.float32, device=self.device)
            dist_matrix = torch.cdist(t_des1, t_des2, p=2)
            top2_distances, top2_indices = torch.topk(dist_matrix, k=2, dim=1, largest=False)
            ratio_mask = top2_distances[:, 0] < (self.ratio * top2_distances[:, 1])
            
            valid_query_indices = torch.nonzero(ratio_mask).squeeze(-1).cpu().numpy()
            valid_train_indices = top2_indices[ratio_mask, 0].cpu().numpy()
            valid_distances = top2_distances[ratio_mask, 0].cpu().numpy()
            
        else:
            # Chunked pass over des2 to keep VRAM usage under 1.5 GB
            N1 = len(t_des1)
            best_d1 = torch.full((N1,), float('inf'), device=self.device)
            best_d2 = torch.full((N1,), float('inf'), device=self.device)
            best_idx1 = torch.zeros((N1,), dtype=torch.long, device=self.device)
            
            for start_idx in range(0, len(des2), chunk_size):
                end_idx = min(start_idx + chunk_size, len(des2))
                chunk = des2[start_idx:end_idx]
                t_chunk = chunk if isinstance(chunk, torch.Tensor) else torch.tensor(chunk, dtype=torch.float32, device=self.device)
                
                dist_chunk = torch.cdist(t_des1, t_chunk, p=2)
                top2_dists, top2_idxs = torch.topk(dist_chunk, k=min(2, t_chunk.shape[0]), dim=1, largest=False)
                top2_idxs += start_idx
                
                # Update running top-2 distances and indices for each query descriptor
                cand_dists = torch.cat([best_d1.unsqueeze(1), best_d2.unsqueeze(1), top2_dists], dim=1)
                cand_idxs = torch.cat([best_idx1.unsqueeze(1), torch.zeros_like(best_idx1).unsqueeze(1), top2_idxs], dim=1)
                
                sorted_dists, sort_perm = torch.sort(cand_dists, dim=1)
                best_d1 = sorted_dists[:, 0]
                best_d2 = sorted_dists[:, 1]
                best_idx1 = torch.gather(cand_idxs, 1, sort_perm[:, :1]).squeeze(1)
                
                del dist_chunk, t_chunk
                
            ratio_mask = best_d1 < (self.ratio * best_d2)
            valid_query_indices = torch.nonzero(ratio_mask).squeeze(-1).cpu().numpy()
            valid_train_indices = best_idx1[ratio_mask].cpu().numpy()
            valid_distances = best_d1[ratio_mask].cpu().numpy()

        good_matches = []
        for i in range(len(valid_query_indices)):
            m = cv2.DMatch(
                int(valid_query_indices[i]),
                int(valid_train_indices[i]),
                float(valid_distances[i])
            )
            good_matches.append(m)

        return good_matches

    def draw_and_save_matches(self, img1, kp1, img2, kp2, matches, output_path):
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
        )
        
        # Safely create the parent directories if they don't exist yet
        out_dir = Path(output_path).parent
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Save the final stitched image to disk
        cv2.imwrite(str(output_path), match_img)  
