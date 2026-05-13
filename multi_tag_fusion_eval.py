#!/usr/bin/env python3
import argparse
import glob
import os
import numpy as np
import cv2
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


def detect_frame(img_path, K, dist, det, tag_size_m):
    img = cv2.imread(img_path)
    if img is None:
        return {}

    und = cv2.undistort(img, K, dist)
    gray = cv2.cvtColor(und, cv2.COLOR_BGR2GRAY)

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    detections = det.detect(
        gray,
        estimate_tag_pose=True,
        camera_params=(fx, fy, cx, cy),
        tag_size=tag_size_m,
    )

    frame = {}
    for d in detections:
        tid = int(d.tag_id)
        R = np.array(d.pose_R, dtype=np.float64)
        t = np.array(d.pose_t, dtype=np.float64).reshape(3)
        frame[tid] = make_T(R, t)
    return frame


def build_tag_map(map_images, K, dist, det, tag_size_m, origin_id):
    """
    Build average pose of each fixed tag in the origin-tag frame:
        T_origin_tag
    """
    tag_poses = {}  # tag_id -> list of T_origin_tag

    for path in map_images:
        frame = detect_frame(path, K, dist, det, tag_size_m)
        if origin_id not in frame:
            continue

        T_cam_origin = frame[origin_id]
        T_origin_cam = inv_T(T_cam_origin)

        for tid, T_cam_tid in frame.items():
            T_origin_tid = T_origin_cam @ T_cam_tid
            tag_poses.setdefault(tid, []).append(T_origin_tid)

    if origin_id not in tag_poses:
        raise RuntimeError(f"Origin tag {origin_id} was never seen in map images.")

    tag_map = {}
    for tid, Ts in tag_poses.items():
        # Average only translations; keep first rotation for simplicity
        Rs = [T[:3, :3] for T in Ts]
        ts = np.array([T[:3, 3] for T in Ts], dtype=np.float64)
        T_avg = np.eye(4, dtype=np.float64)
        T_avg[:3, :3] = Rs[0]
        T_avg[:3, 3] = np.mean(ts, axis=0)
        tag_map[tid] = T_avg

    return tag_map


def vec_stats_mm(vectors_m):
    arr = np.asarray(vectors_m, dtype=np.float64)
    if len(arr) == 0:
        return None
    std_mm = np.std(arr, axis=0) * 1000.0
    span_mm = (np.max(arr, axis=0) - np.min(arr, axis=0)) * 1000.0
    radial_span_mm = np.linalg.norm(np.max(arr, axis=0) - np.min(arr, axis=0)) * 1000.0
    return std_mm, span_mm, radial_span_mm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map_glob", required=True, help="Static images with fixed tags visible")
    ap.add_argument("--test_glob", required=True, help="Images for single-tag vs multi-tag comparison")
    ap.add_argument("--intrinsics", default="camera_intrinsics.npz")
    ap.add_argument("--tag_size_mm", type=float, required=True)
    ap.add_argument("--origin_id", type=int, required=True)
    ap.add_argument("--moved_id", type=int, required=True)
    ap.add_argument(
        "--fusion_refs",
        default="",
        help="Comma-separated fixed tag ids to use for fusion, e.g. 0,1,2,3,10,11. "
             "If empty, all mapped tags except moved_id are used."
    )
    args = ap.parse_args()

    map_images = sorted(glob.glob(args.map_glob))
    test_images = sorted(glob.glob(args.test_glob))
    if not map_images:
        raise RuntimeError(f"No map images found for {args.map_glob}")
    if not test_images:
        raise RuntimeError(f"No test images found for {args.test_glob}")

    data = np.load(args.intrinsics)
    K = data["K"].astype(np.float64)
    dist = data["dist"].astype(np.float64)

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

    tag_map = build_tag_map(map_images, K, dist, det, tag_size_m, args.origin_id)
    print("Built tag map in origin frame:")
    for tid in sorted(tag_map.keys()):
        t = tag_map[tid][:3, 3] * 1000.0
        print(f"  tag {tid}: x_mm={t[0]:.1f}, y_mm={t[1]:.1f}, z_mm={t[2]:.1f}")

    if args.fusion_refs.strip():
        fusion_refs = [int(x) for x in args.fusion_refs.split(",") if x.strip()]
    else:
        fusion_refs = [tid for tid in sorted(tag_map.keys()) if tid != args.moved_id]

    print(f"\nFusion refs: {fusion_refs}\n")

    single_results = []  # (name, t_origin_moved)
    fused_results = []   # (name, t_origin_moved_fused)

    for path in test_images:
        name = os.path.basename(path)
        frame = detect_frame(path, K, dist, det, tag_size_m)

        if args.moved_id not in frame:
            print(f"{name}: moved tag {args.moved_id} not visible, skipping")
            continue

        T_cam_moved = frame[args.moved_id]

        # Single-tag baseline: use origin tag directly
        single_t = None
        if args.origin_id in frame:
            T_cam_origin = frame[args.origin_id]
            T_origin_cam = inv_T(T_cam_origin)
            T_origin_moved = T_origin_cam @ T_cam_moved
            single_t = T_origin_moved[:3, 3].copy()
            single_results.append((name, single_t))

        # Multi-tag fusion: each visible reference gives an estimate in the origin frame
        candidate_ts = []
        used_refs = []

        for ref_id in fusion_refs:
            if ref_id == args.moved_id:
                continue
            if ref_id not in frame:
                continue
            if ref_id not in tag_map:
                continue

            # Known map pose of ref in origin frame
            T_origin_ref = tag_map[ref_id]

            # Observed pose of ref in camera frame
            T_cam_ref = frame[ref_id]

            # Estimate camera pose in origin frame using this ref
            T_origin_cam_est = T_origin_ref @ inv_T(T_cam_ref)

            # Estimate moved tag in origin frame
            T_origin_moved_est = T_origin_cam_est @ T_cam_moved
            candidate_ts.append(T_origin_moved_est[:3, 3].copy())
            used_refs.append(ref_id)

        fused_t = None
        if candidate_ts:
            fused_t = np.mean(np.asarray(candidate_ts, dtype=np.float64), axis=0)
            fused_results.append((name, fused_t))

        print(name)
        if single_t is not None:
            print(f"  single-tag  (origin {args.origin_id}) -> "
                  f"x={single_t[0]*1000:.1f}, y={single_t[1]*1000:.1f}, z={single_t[2]*1000:.1f} mm")
        else:
            print("  single-tag  -> unavailable (origin tag not visible)")

        if fused_t is not None:
            print(f"  multi-tag   (refs {used_refs}) -> "
                  f"x={fused_t[0]*1000:.1f}, y={fused_t[1]*1000:.1f}, z={fused_t[2]*1000:.1f} mm")
        else:
            print("  multi-tag   -> unavailable (no fusion refs visible)")
        print()

    print("\n=== Stability Summary ===")
    if single_results:
        single_vecs = [v for _, v in single_results]
        s = vec_stats_mm(single_vecs)
        print("Single-tag:")
        print(f"  std_mm  = [{s[0][0]:.2f}, {s[0][1]:.2f}, {s[0][2]:.2f}]")
        print(f"  span_mm = [{s[1][0]:.2f}, {s[1][1]:.2f}, {s[1][2]:.2f}]")
        print(f"  radial_span_mm = {s[2]:.2f}")
    else:
        print("Single-tag: no valid results")

    if fused_results:
        fused_vecs = [v for _, v in fused_results]
        s = vec_stats_mm(fused_vecs)
        print("Multi-tag fusion:")
        print(f"  std_mm  = [{s[0][0]:.2f}, {s[0][1]:.2f}, {s[0][2]:.2f}]")
        print(f"  span_mm = [{s[1][0]:.2f}, {s[1][1]:.2f}, {s[1][2]:.2f}]")
        print(f"  radial_span_mm = {s[2]:.2f}")
    else:
        print("Multi-tag fusion: no valid results")


if __name__ == "__main__":
    main()