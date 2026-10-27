"""
Analysis helpers for notebooks/results_analysis.ipynb.

incremental_sfm.ipynb saves one reconstruction run with save_sfm_run(); the analysis notebook
loads it with load_sfm_run() and only computes metrics and plots, so it runs in seconds.
"""
import pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt

from feature_tracks import TrackGraph
from pose_estimation import PoseEstimator
from triangulation import Triangulator
from utils import camera_centers


# ----------------------------------------------------------------------------- save / load

class KeypointXY:
    """Minimal stand-in for cv2.KeyPoint (only .pt), so pipeline functions work on loaded runs."""
    __slots__ = ("pt",)

    def __init__(self, pt):
        self.pt = (float(pt[0]), float(pt[1]))


def _cams_to_dict(cams: dict) -> dict:
    return {int(c): {"R": np.asarray(v["R"], dtype=np.float64), "T": np.asarray(v["T"], dtype=np.float64).reshape(3, 1)}
            for c, v in cams.items()}


def _track_to_dict(track) -> dict:
    return {"points": {int(p): np.asarray(x, dtype=np.float64).ravel() for p, x in track.global_points3D.items()},
            "obs": {(int(c), int(k)): int(p) for (c, k), p in track.kp_to_point3D.items()}}


def as_track_graph(track_dict: dict) -> TrackGraph:
    """Rebuilds a TrackGraph from its saved dict form."""
    tg = TrackGraph()
    tg.global_points3D = dict(track_dict["points"])
    tg.kp_to_point3D = dict(track_dict["obs"])
    tg.point_counter = max(tg.global_points3D, default=-1) + 1
    return tg


def save_sfm_run(path, *, K, all_kp, match_graph, n_images, submodels: dict, master_before_ba, master_after_ba,
                 cam_source: dict, merge_stats: list, image_groups: dict) -> None:
    """
    Saves everything the analysis notebook needs in one pickle file.
        submodels        : {name: (cameras, track_graph)} after registration, before merging
        master_before_ba : (cameras, track_graph) right after merging the sub-models
        master_after_ba  : (cameras, track_graph) after global bundle adjustment
        image_groups     : {group name: list of image indices}, e.g. which photos show the back facade
    The seed pair of each sub-model is its first two cameras (initialize_seed_submodel inserts them first).
    """
    seeds = {name: tuple(int(c) for c in list(cams)[:2]) for name, (cams, _) in submodels.items()}

    match_counts = np.zeros((n_images, n_images), dtype=np.int32)
    for (i, j), m in match_graph.items():
        match_counts[i, j] = match_counts[j, i] = len(m)

    # Match index pairs only for consecutive images and the seed pairs (enough for seed scoring)
    wanted = {(i, i + 1) for i in range(n_images - 1)} | {tuple(sorted(s)) for s in seeds.values()}
    pair_matches = {pair: np.array([[m.queryIdx, m.trainIdx] for m in match_graph[pair]], dtype=np.int32)
                    for pair in wanted if pair in match_graph}

    run = {
        "K": np.asarray(K, dtype=np.float64),
        "n_images": int(n_images),
        "kp_xy": {int(i): np.array([kp.pt for kp in kps], dtype=np.float32) for i, kps in all_kp.items()},
        "match_counts": match_counts,
        "pair_matches": pair_matches,
        "seeds": seeds,
        "submodels": {name: {"cameras": _cams_to_dict(cams), "track": _track_to_dict(track)}
                      for name, (cams, track) in submodels.items()},
        "before_ba": {"cameras": _cams_to_dict(master_before_ba[0]), "track": _track_to_dict(master_before_ba[1])},
        "after_ba": {"cameras": _cams_to_dict(master_after_ba[0]), "track": _track_to_dict(master_after_ba[1])},
        "cam_source": {int(c): s for c, s in cam_source.items()},
        "merge_stats": list(merge_stats),
        "image_groups": {g: sorted(int(i) for i in ids) for g, ids in image_groups.items()},
    }
    with open(path, "wb") as f:
        pickle.dump(run, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved reconstruction run for analysis to {path}")


def load_sfm_run(path) -> dict:
    """Loads a run saved by save_sfm_run. Adds 'all_kp' (KeypointXY lists) for the pipeline functions."""
    with open(path, "rb") as f:
        run = pickle.load(f)
    run["all_kp"] = {i: [KeypointXY(p) for p in xy] for i, xy in run["kp_xy"].items()}
    return run


# ----------------------------------------------------------------------------- metrics

def track_lengths(track_dict: dict, cameras: dict) -> np.ndarray:
    """Number of registered images observing each point (same order as track_dict['points'])."""
    count = {p: 0 for p in track_dict["points"]}
    for (c, _), p in track_dict["obs"].items():
        if c in cameras and p in count:
            count[p] += 1
    return np.array(list(count.values()))


def score_seed_pair(kp_xy_a: np.ndarray, kp_xy_b: np.ndarray, matches: np.ndarray, K: np.ndarray) -> dict:
    """
    Two-view quality of a candidate seed pair, using the pipeline's own estimator/triangulator:
        e_inliers : RANSAC inliers of the essential matrix
        angle     : median triangulation angle (deg) of the inlier points, i.e. parallax
        h_ratio   : homography inliers / E inliers; close to 1 or above means planar scene or
                    almost no translation, the degenerate case where E (and the pose) is ill-conditioned
    """
    est, tri = PoseEstimator(K), Triangulator(K)
    pts1 = kp_xy_a[matches[:, 0]].astype(np.float32)
    pts2 = kp_xy_b[matches[:, 1]].astype(np.float32)
    E, mask = est.estimate_essential_matrix(pts1, pts2)
    if E is None or E.shape != (3, 3):
        return None
    R, T = est.recover_camera_pose(E, pts1, pts2, mask)
    inl = mask.ravel() == 1
    H, hmask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 4.0)
    P0, P1 = tri.construct_projection_matrices(R, T)
    X = tri.triangulate_points(P0, P1, pts1[inl], pts2[inl])
    X, _ = tri.filter_cheirality(X, R, T)
    angles = tri.triangulation_angles(X, np.eye(3), np.zeros(3), R, T) if len(X) else np.zeros(1)
    return {"matches": len(matches), "e_inliers": int(inl.sum()), "points": len(X),
            "angle": float(np.median(angles)),
            "h_ratio": (int(hmask.sum()) if hmask is not None else 0) / max(int(inl.sum()), 1)}


def print_table(rows: list, columns: list, formats: dict = None) -> None:
    """Plain-text table: rows = list of dicts, columns = list of (key, header)."""
    formats = formats or {}
    def cell(r, k):   # missing value -> "-", intentionally empty string stays empty
        v = r.get(k)
        return "-" if v is None else "" if v == "" else formats.get(k, "{}").format(v)
    cells = [[cell(r, k) for k, _ in columns] for r in rows]
    widths = [max(len(h), *(len(c[i]) for c in cells)) for i, (_, h) in enumerate(columns)]
    print("  ".join(h.rjust(w) for (_, h), w in zip(columns, widths)))
    print("  ".join("-" * w for w in widths))
    for c in cells:
        print("  ".join(v.rjust(w) for v, w in zip(c, widths)))


# ----------------------------------------------------------------------------- plots

GROUP_COLORS = {"front facade": "tab:blue", "back facade": "tab:red", "side A": "tab:green",
                "side B": "tab:orange", "close-ups": "tab:gray"}
SUBMODEL_COLORS = {"front": "tab:blue", "back": "tab:red", "left": "tab:orange", "right": "tab:green"}


def plot_match_matrix(match_counts: np.ndarray, image_groups: dict, seeds: dict) -> None:
    """Number of matches for every image pair (log scale). Off-diagonal blocks between front and
    back photos are matches between two different facades: the symmetry problem made visible."""
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(np.log10(match_counts + 1), cmap="magma", interpolation="nearest")
    fig.colorbar(im, ax=ax, label="log10(matches + 1)")
    n = len(match_counts)
    for g, ids in image_groups.items():
        for run_start, run_end in _runs(ids):
            ax.add_patch(plt.Rectangle((-6, run_start - 0.5), 4, run_end - run_start + 1, clip_on=False,
                                       color=GROUP_COLORS.get(g, "k")))
    for name, (a, b) in seeds.items():
        ax.plot(b, a, "o", mfc="none", mec="cyan", ms=10)
        ax.annotate(name, (b, a), (b + 4, a - 4), color="cyan", fontsize=9)
    handles = [plt.Line2D([], [], color=GROUP_COLORS.get(g, "k"), lw=6, label=g) for g in image_groups]
    ax.legend(handles=handles, loc="lower left", fontsize=8)
    ax.set_xlim(-7, n - 0.5)
    ax.set_title("Match graph: matches per image pair (circles = seed pairs)")
    ax.set_xlabel("image index"); ax.set_ylabel("image index")
    plt.tight_layout(); plt.show()


def _runs(ids):
    """Consecutive runs [(start, end), ...] in a sorted index list."""
    ids = sorted(ids); runs = []
    for i in ids:
        if runs and i == runs[-1][1] + 1:
            runs[-1][1] = i
        else:
            runs.append([i, i])
    return runs


def rank_seed_candidates(pair_scores: dict, image_groups: dict, seeds: dict, min_points: int = 500,
                         min_angle_deg: float = 4.0, max_h_ratio: float = 0.8, top: int = 5,
                         groups=("front facade", "back facade", "side A", "side B")) -> None:
    """
    Prints, per part of the building, the neighbouring image pairs that are valid seeds
    (>= min_points triangulated inliers, median angle >= min_angle_deg, H/E < max_h_ratio),
    ranked by number of 3D points, and where the seeds actually used stand.
    Both images of a pair must belong to the same photo group, so a seed shows one facade.
    """
    group_of = {i: g for g, ids in image_groups.items() for i in ids}
    used = {tuple(sorted(s)): name for name, s in seeds.items()}
    for g in groups:
        cands = [(p, s) for p, s in pair_scores.items() if s and group_of.get(p[0]) == g and group_of.get(p[1]) == g]
        valid = [(p, s) for p, s in cands
                 if s["points"] >= min_points and s["angle"] >= min_angle_deg and s["h_ratio"] < max_h_ratio]
        valid.sort(key=lambda r: -r[1]["points"])
        rank = {p: i + 1 for i, (p, _) in enumerate(valid)}
        shown = valid[:top] + [(p, s) for p, s in cands if p in used and rank.get(p, top + 1) > top]
        print(f"\n{g}: {len(valid)} valid of {len(cands)} neighbour pairs")
        rows = [{"rank": rank.get(p, "-"), "pair": f"({p[0]}, {p[1]})", **s,
                 "used": f"<- seed '{used[p]}'" if p in used else ""} for p, s in shown]
        if rows:
            print_table(rows, [("rank", "rank"), ("pair", "pair"), ("points", "3D points"), ("angle", "median angle [deg]"),
                               ("h_ratio", "H/E"), ("used", "")], {"angle": "{:.1f}", "h_ratio": "{:.2f}"})


def plot_submodel_coverage(submodels: dict, n_images: int, image_groups: dict) -> None:
    """Which images each sub-model registered, on top of the photo groups."""
    fig, ax = plt.subplots(figsize=(13, 3.2))
    for g, ids in image_groups.items():
        for a, b in _runs(ids):
            ax.axvspan(a - 0.5, b + 0.5, color=GROUP_COLORS.get(g, "k"), alpha=0.12)
    for row, (name, sm) in enumerate(submodels.items()):
        ids = sorted(sm["cameras"])
        ax.scatter(ids, [row] * len(ids), s=14, marker="s", color=SUBMODEL_COLORS.get(name, "k"))
    ax.set_yticks(range(len(submodels))); ax.set_yticklabels(list(submodels))
    ax.set_xlim(-1, n_images); ax.set_xlabel("image index")
    handles = [plt.Line2D([], [], color=GROUP_COLORS.get(g, "k"), lw=6, alpha=0.3, label=g) for g in image_groups]
    ax.legend(handles=handles, ncol=len(handles), loc="upper center", bbox_to_anchor=(0.5, -0.35), fontsize=8)
    ax.set_title("Registered images per sub-model (background = photo groups)")
    plt.tight_layout(); plt.show()


def plot_cameras_topview(cameras: dict, points: np.ndarray, color_of: dict, colors: dict, title: str, ax=None) -> None:
    """Top view (X-Z) of points (grey) and camera centers colored by a label per camera."""
    show = ax is None
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 7))
    lo, hi = np.percentile(points, 1, axis=0), np.percentile(points, 99, axis=0)
    P = points[np.all((points >= lo) & (points <= hi), axis=1)]
    ax.scatter(P[:, 0], P[:, 2], s=0.2, c="0.6")
    for label, col in colors.items():
        ids = [c for c in cameras if color_of.get(c) == label]
        if ids:
            C = camera_centers(cameras, ids)
            ax.scatter(C[:, 0], C[:, 2], s=22, marker="^", color=col, label=f"{label} ({len(ids)})")
    ax.set_aspect("equal"); ax.set_xlabel("X"); ax.set_ylabel("Z (depth of reference camera)")
    ax.set_title(title); ax.legend(fontsize=8)
    if show:
        plt.tight_layout(); plt.show()


def plot_error_histograms(err_before: np.ndarray, err_after: np.ndarray) -> None:
    """Reprojection error of all observations before and after global BA."""
    fig, ax = plt.subplots(figsize=(9, 4))
    bins = np.linspace(0, 20, 81)
    ax.hist(np.clip(err_before, 0, 20), bins=bins, alpha=0.55, label=f"after merge, before BA (median {np.median(err_before):.2f} px)")
    ax.hist(np.clip(err_after, 0, 20), bins=bins, alpha=0.55, label=f"after global BA (median {np.median(err_after):.2f} px)")
    ax.set_yscale("log"); ax.set_xlabel("reprojection error [px] (clipped at 20)"); ax.set_ylabel("observations")
    ax.set_title("Reprojection error per observation"); ax.legend()
    plt.tight_layout(); plt.show()


def plot_point_quality(quality: dict, min_track_length=3, min_angle_deg=5.0, max_reproj_px=2.0) -> None:
    """Histograms of the three point quality measures with the filter thresholds."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    tl = quality["track_length"]
    axes[0].hist(tl, bins=np.arange(1.5, min(tl.max(), 20) + 1.5), color="tab:blue")
    axes[0].axvline(min_track_length - 0.5, color="k", ls="--"); axes[0].set_xlabel("track length [views]")
    axes[0].set_title(f"Track length (median {np.median(tl):.0f}, {np.mean(tl == 2):.0%} only 2 views)")
    axes[1].hist(np.clip(quality["max_angle"], 0, 60), bins=60, color="tab:green")
    axes[1].axvline(min_angle_deg, color="k", ls="--"); axes[1].set_xlabel("max triangulation angle [deg] (clipped 60)")
    axes[1].set_title(f"Triangulation angle (median {np.median(quality['max_angle']):.1f}°)")
    axes[2].hist(np.clip(quality["reproj_err"], 0, 6), bins=60, color="tab:purple")
    axes[2].axvline(max_reproj_px, color="k", ls="--"); axes[2].set_xlabel("mean reprojection error [px] (clipped 6)")
    axes[2].set_title(f"Reprojection error (median {np.median(quality['reproj_err']):.2f} px)")
    for ax in axes:
        ax.set_ylabel("points")
    plt.tight_layout(); plt.show()


def plot_cloud_views(points: np.ndarray, cameras: dict, cam_source: dict, title: str) -> None:
    """Top view (X-Z) and front view (X-Y, image y points down) of a point cloud with the cameras."""
    lo, hi = np.percentile(points, 1, axis=0), np.percentile(points, 99, axis=0)
    P = points[np.all((points >= lo) & (points <= hi), axis=1)]
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, (a, b, lab) in zip(axes, [(0, 2, "top view (X-Z)"), (0, 1, "front view (X-Y)")]):
        ax.scatter(P[:, a], P[:, b], s=0.3, c=P[:, 2 if b == 1 else 1], cmap="viridis")
        if b == 2:
            for name, col in SUBMODEL_COLORS.items():
                ids = [c for c in cameras if cam_source.get(c) == name]
                if ids:
                    C = camera_centers(cameras, ids)
                    ax.scatter(C[:, a], C[:, b], s=20, marker="^", color=col, label=f"{name} ({len(ids)})")
            ax.legend(fontsize=8)
        else:
            ax.invert_yaxis()
        ax.set_aspect("equal"); ax.set_title(f"{title}: {lab}")
    plt.tight_layout(); plt.show()
