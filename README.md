# Incremental SFM and Dense Mesh Reconstruction of the Propyläen
I present a fun little project on incremental SFM and dense mesh 3D reconstruction of the Propyläen building in Munich. The main target was to implement functions I learned in a lecture. If you don't have any experience in 3DCV I highly recommend you watching the lecture of Prof. Dr. Daniel Cremers on youtube. If there is some background I highly recommend reading following paper:

- Distinctive Image Features from Scale-Invariant Keypoints (D. G. Lowe), 2004
- An Efficient Solution to the Five-Point Relative Pose Problem (D. Nistér), 2004 
- Screened Poisson Surface Reconstruction (M. Kazhdan, H. Hoppe), 2013

## Dependencies
I support Ubuntu, but it might work on a variety of other platforms that meet the dependency requirements.

### Required:
- python 3.10
- cmake 3.2 or later
- opencv-python
- opencv-contrib-python
- numpy
- scipy
- matplotlib
- open3d
- torch (feature matching and bundle adjustment on the GPU; `incremental_sfm.ipynb` expects a CUDA GPU)
- ultralytics (YOLOv8 segmentation for masking cars, people and traffic lights; downloads `yolov8n-seg.pt` on first use)

## System Setup
The main dependencies can be installed using conda and pip:

conda create -n vision_3d python=3.10 -y
conda activate vision_3d
pip install opencv-python opencv-contrib-python numpy scipy matplotlib open3d torch ultralytics

## Project Structure
```text
propylaeen_3d_reconstruct/
├── data/               # Raw image dataset (Ignored by Git)
├── notebooks/          # Interactive experimentation & visualization
│   ├── incremental_sfm.ipynb          # Full pipeline: sub-models, merge, global BA, export
│   ├── results_analysis.ipynb         # Plots and statistics of a reconstruction run
│   ├── surface_meshing.ipynb          # Poisson surface mesh of the point cloud
│   └── reconstruction_analysis.ipynb  # Earlier two-view reconstruction (one image pair)
├── outputs/            # Generated point clouds, meshes and run files (Ignored by Git)
├── src/                # Core modular Python engine
│   ├── dataset.py            # Image loading & dataset handling
│   ├── feature_matching.py   # SIFT extraction, dynamic-object mask & Lowe's ratio test matching (GPU)
│   ├── feature_tracks.py     # TrackGraph: 3D points and their keypoint observations
│   ├── pose_estimation.py    # Essential matrix (5-point + RANSAC) & relative pose
│   ├── triangulation.py      # DLT triangulation, cheirality, reprojection errors, triangulation angles
│   ├── bundle_adjustment.py  # Two-view, local and global bundle adjustment (PyTorch, Huber loss)
│   ├── utils.py              # K, seed initialisation, PnP registration, Sim(3) merging, point quality, PLY export
│   └── analysis.py           # Save/load a reconstruction run, metrics and plots for results_analysis
├── .gitignore          # Git exclusion rules for large data/outputs
└── README.md           # Project documentation
```

## Usage
1. Clone the repository and navigate into the package directory:
git clone git@github.com:YOUR_USERNAME/propylaeen_3d_reconstruct.git
cd propylaeen_3d_reconstruct

2. Place dataset images into the data/ directory.

3. Run the notebooks in this order (kernel: the `vision_3d` environment), each from top to bottom:

   1. `notebooks/incremental_sfm.ipynb`: features, matching, four sub-models, merging, global bundle adjustment.
      Writes `outputs/merged_global_ba_reconstruction.ply`, `outputs/merged_global_ba_viewpoints.npz` and `outputs/sfm_run.pkl`.
   2. `notebooks/results_analysis.ipynb`: plots and statistics of that run (reads `outputs/sfm_run.pkl`).
   3. `notebooks/surface_meshing.ipynb`: Poisson surface mesh of the point cloud, writes `outputs/building_mesh.ply`.

   Start Jupyter from the repository root and open the notebooks from there:

   ```bash
   jupyter notebook notebooks/
   ```

   `notebooks/reconstruction_analysis.ipynb` is the earlier two-view reconstruction (one image pair) and can be run on its own.