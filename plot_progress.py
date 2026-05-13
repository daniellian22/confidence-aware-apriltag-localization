#!/usr/bin/env python3
import argparse
import glob
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pupil_apriltags import Detector


def make_T(R, t):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t.reshape(3)
    return T


def inv_T(T):
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4, dtype=np.float64)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def radial_span_mm(vectors_m):
    arr = np.asarray(vectors_m, dtype=np.float64)
    return np.linalg.norm(np.max(arr, axis=0) - np.min(arr, axis=0)) * 1000.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_glob", required=True)
    ap.add_argument("--static_glob", required=True, help="Repeated static images, e.g. static/IMG_243[2-6].jpeg")
    ap.add_argument("--intrinsics", default="camera_intrinsics.npz")
    ap.add_argument("--tag_size_mm", type=float, default=50.8)
    ap.add_argument("--origin_id", type=int, default=0)
    ap.add_argument("--moved_id", type=int, default=11)
    ap.add_argument("--out", default="progress_plot.png")
    args = ap.parse_args()

    imgs = sorted(glob.glob(args.images_glob))
    static_imgs = sorted(glob.glob(args.static_glob))
    if not imgs:
        raise RuntimeError("No images found for --images_glob")
    if not static_imgs:
        raise RuntimeError("No images found for --static_glob")

    data = np.load(args.intrinsics)
    K = data["K"].astype(np.float64)
    dist = data["dist"].astype(np.float64)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    det = Detector(
        families="tag36h11",
        nthreads=4,
        quad_decimate=1.0,
        quad_sigma=0.0,
        refine_edges=1,
        decode_sharpening=0.25,
        debug=0,
    )

    tag_size_m = args.tag_size_mm / 1000.0

    # -------- Panel A data: per-tag motion ranking in origin frame if possible, else camera frame --------
    per_tag_vectors = {}
    for p in imgs:
        img = cv2.imread(p)
        if img is None:
            continue
        und = cv2.undistort(img, K, dist)
        gray = cv2.cvtColor(und, cv2.COLOR_BGR2GRAY)
        ds = det.detect(
            gray,
            estimate_tag_pose=True,
            camera_params=(fx, fy, cx, cy),
            tag_size=tag_size_m,
        )

        frame = {}
        for d in ds:
            tid = int(d.tag_id)
            R = np.array(d.pose_R, dtype=np.float64)
            t = np.array(d.pose_t, dtype=np.float64).reshape(3)
            frame[tid] = make_T(R, t)

        if args.origin_id in frame:
            T_cam_origin = frame[args.origin_id]
            T_origin_cam = inv_T(T_cam_origin)
            for tid, T_cam_tid in frame.items():
                T_origin_tid = T_origin_cam @ T_cam_tid
                per_tag_vectors.setdefault(tid, []).append(T_origin_tid[:3, 3].copy())
        else:
            for tid, T_cam_tid in frame.items():
                per_tag_vectors.setdefault(tid, []).append(T_cam_tid[:3, 3].copy())

    tag_ids = sorted(per_tag_vectors.keys())
    spans = [radial_span_mm(per_tag_vectors[tid]) for tid in tag_ids]

    # -------- Panel B data: repeated-image XY cluster for moved tag --------
    xy_mm = []
    labels = []
    for p in static_imgs:
        img = cv2.imread(p)
        if img is None:
            continue
        und = cv2.undistort(img, K, dist)
        gray = cv2.cvtColor(und, cv2.COLOR_BGR2GRAY)
        ds = det.detect(
            gray,
            estimate_tag_pose=True,
            camera_params=(fx, fy, cx, cy),
            tag_size=tag_size_m,
        )

        by = {int(d.tag_id): d for d in ds}
        if args.origin_id not in by or args.moved_id not in by:
            continue

        do = by[args.origin_id]
        dm = by[args.moved_id]

        Tco = make_T(np.array(do.pose_R), np.array(do.pose_t).reshape(3))
        Tcm = make_T(np.array(dm.pose_R), np.array(dm.pose_t).reshape(3))
        Tom = inv_T(Tco) @ Tcm
        t_mm = Tom[:3, 3] * 1000.0
        xy_mm.append([t_mm[0], t_mm[1]])
        labels.append(os.path.basename(p))

    xy_mm = np.asarray(xy_mm, dtype=np.float64)
    if len(xy_mm) == 0:
        raise RuntimeError("No valid repeated-image points found for static cluster plot")

    mean_xy = np.mean(xy_mm, axis=0)
    std_xy = np.std(xy_mm, axis=0)

    # -------- Plot --------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A
    ax = axes[0]
    x = np.arange(len(tag_ids))
    bars = ax.bar(x, spans)
    for i, tid in enumerate(tag_ids):
        if tid == args.moved_id:
            bars[i].set_hatch("//")
    ax.set_xticks(x)
    ax.set_xticklabels([str(t) for t in tag_ids])
    ax.set_xlabel("Tag ID")
    ax.set_ylabel("Radial span (mm)")
    ax.set_title("Per-tag motion ranking")
    ax.text(
        0.02, 0.98,
        f"Expected moved tag: {args.moved_id}",
        transform=ax.transAxes,
        va="top"
    )

    # Panel B
    ax = axes[1]
    ax.scatter(xy_mm[:, 0], xy_mm[:, 1], s=60)
    for (xv, yv), lab in zip(xy_mm, labels):
        ax.annotate(lab.replace(".jpeg", ""), (xv, yv), xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.scatter([mean_xy[0]], [mean_xy[1]], marker="x", s=120)
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_title("Static repeatability cluster")
    ax.text(
        0.02, 0.98,
        f"std_x={std_xy[0]:.2f} mm\nstd_y={std_xy[1]:.2f} mm",
        transform=ax.transAxes,
        va="top"
    )

    plt.tight_layout()
    plt.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"Saved plot to {args.out}")


if __name__ == "__main__":
    main()