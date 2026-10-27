import cv2
import numpy as np
import torch
from ultralytics import YOLO


class FeatureMatcher:
    def __init__(self, ratio_threshold=0.75, device = "cuda", yolo_model="yolov8n-seg.pt"):
        self.detector = cv2.SIFT_create()
        self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        self.ratio = ratio_threshold
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.yolo = YOLO(yolo_model)
        # 2: car, 0: person, 5: bus, 7: truck, 9: traffic light, 11: stop sign (adjust as needed)
        self.classes_to_remove = [0, 2, 5, 7, 9, 11]

    def create_dynamic_mask(self, image):
        """
        Creates a binary mask where dynamic/unwanted objects (cars, poles, traffic lights)
        and trees/foliage are blacked out (0), leaving only the static building/background (255).
        """
        h, w = image.shape[:2]
        # Start with everything active (white = 255)
        mask = np.ones((h, w), dtype=np.uint8) * 255

        # 1. Mask dynamic hardware/cars/traffic lights using YOLO
        results = self.yolo(image, verbose=False)
        for result in results:
            if result.masks is not None:
                boxes = result.boxes.cpu().numpy()
                masks = result.masks.cpu().numpy()
                for box, seg_mask in zip(boxes, masks.data):
                    if int(box.cls[0]) in self.classes_to_remove:
                        resized_mask = cv2.resize(seg_mask, (w, h))
                        mask[resized_mask > 0.5] = 0

        # 2. Mask trees/vegetation using HSV + Morphological Cleanup
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Broad HSV bounds for green leaves and yellowish foliage
        lower_foliage = np.array([35, 70, 30])
        upper_foliage = np.array([85, 255, 255])
        
        tree_mask = cv2.inRange(hsv, lower_foliage, upper_foliage)
        
        # Morphological Closing: fills small holes/gaps between leaves
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        tree_mask = cv2.morphologyEx(tree_mask, cv2.MORPH_CLOSE, kernel)
        
        # Apply tree mask to our master mask
        mask[tree_mask > 0] = 0

        return mask
    
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
        mask = self.create_dynamic_mask(image)
        keypoints, descriptors = self.detector.detectAndCompute(
                    gray, mask
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
