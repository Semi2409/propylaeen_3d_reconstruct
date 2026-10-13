# Incremental SFM and Dense Mesh Reconstruction of the Propyläen
I present a fun little project on incremental SFM and dense mesh 3D reconstruction of the Propyläen building in Munich. The main target was to implement functions I learned in a lecture. If you don't have any experience in 3DCV I highly recommend you watching the lecture of Prof. Dr. Daniel Cremers on youtube. If there is some background I highly recommend reading following paper:

Distinctive Image Features from Scale-Invariant Keypoints (D. G. Lowe), 2004
An Efficient Solution to the Five-Point Relative Pose Problem (D. Nistér), 2004
Screened Poisson Surface Reconstruction (M. Kazhdan, H. Hoppe), 2013

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

## System Setup
The main dependencies can be installed using conda and pip:

conda create -n vision_3d python=3.10 -y
conda activate vision_3d
pip install opencv-python opencv-contrib-python numpy scipy matplotlib open3d

## Project Structure
```text
propyläen_3d_reconstruct/
├── .vscode/
├── data/
├── notebooks/
│   ├── incremental_sfm.ipynb
│   ├── reconstruction_analysis.ipynb
│   └── surface_meshing.ipynb
├── outputs/
│   ├── building_mesh.ply
│   ├── final_reconstruction.ply
│   └── incremental_building_reconstruction.ply
├── src/
│   ├── bundle_adjustment.py
│   ├── dataset.py
│   ├── feature_matching.py
│   ├── main.py
│   ├── pose_estimation.py
│   ├── triangulation.py
│   └── utils.py
├── .gitignore
└── README.md
```

## Usage
1. Clone the repository and navigate into the package directory:
git clone git@github.com:YOUR_USERNAME/propylaeen_3d_reconstruct.git
cd propylaeen_3d_reconstruct

2. Place dataset images into the data/ directory.

3. Run the automated reconstruction pipeline:
python src/main.py

4. Alternatively, step through the pipeline interactively using Jupyter Notebooks
jupyter notebook notebooks/reconstruction_analysis.ipynb