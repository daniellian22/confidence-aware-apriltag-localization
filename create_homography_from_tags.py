#!/usr/bin/env python3
import argparse
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="Single image with >=4 visible tags")
    ap.add_argument("--intrinsics", default="camera_intrinsics.npz")
    ap.add_argument("--tag_size_mm", type=float, default=50.8, help="Physical side length of the black tag square")
    ap.add_argument("--origin_id", type=int, default=0)
    ap.add_argument("--tag_ids", default="0,1,2,3,10", help="IDs to use for homography")
    ap.add_argument("--out", default="pixel_to_grid_H.npz")
    args = ap.parse_args()

    ids = [int(x) for x in args.tag_ids.split(",") if x.strip()]

    data = np.load(args.intrinsics)
    K = data["K"].astype(np.float64)
    dist = data["dist"].astype(np.float64)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    img = cv2.imread(args.image)
    if img is None:
        raise RuntimeError(f"Cannot read image: {args.image}")

    und = cv2.undistort(img, K, dist)
    gray = cv2.cvtColor(und, cv2.COLOR_BGR2GRAY)

    det = Detector(
        families="tag36h11",
        nthreads=4,
        quad_decimate=1.0,
        quad_sigma=0.0,
        refine_edges=1,
        decode_sharpening=0.25,
        debug=0,
    )

    detections = det.detect(
        gray,
        estimate_tag_pose=True,
        camera_params=(fx, fy, cx, cy),
        tag_size=args.tag_size_mm / 1000.0,
    )

    by_id = {int(d.tag_id): d for d in detections}
    if args.origin_id not in by_id:
        raise RuntimeError(f"Origin tag id={args.origin_id} not found in image")

    d_origin = by_id[args.origin_id]
    T_cam_origin = make_T(np.array(d_origin.pose_R, dtype=np.float64), np.array(d_origin.pose_t, dtype=np.float64).reshape(3))
    T_origin_cam = inv_T(T_cam_origin)

    uv_pts = []
    xy_pts = []
    used_ids = []

    for tid in ids:
        if tid not in by_id:
            continue
        d = by_id[tid]

        # Undistorted image pixel center (u,v)
        center_uv = np.array(d.center, dtype=np.float64).reshape(2)

        # Corresponding origin-frame XY in mm
        T_cam_tag = make_T(np.array(d.pose_R, dtype=np.float64), np.array(d.pose_t, dtype=np.float64).reshape(3))
        T_origin_tag = T_origin_cam @ T_cam_tag
        xy_mm = T_origin_tag[:2, 3] * 1000.0

        uv_pts.append(center_uv)
        xy_pts.append(xy_mm)
        used_ids.append(tid)

    if len(uv_pts) < 4:
        raise RuntimeError(f"Need at least 4 correspondences, found {len(uv_pts)}")

    uv = np.asarray(uv_pts, dtype=np.float64)
    xy = np.asarray(xy_pts, dtype=np.float64)

    H, mask = cv2.findHomography(uv, xy, method=0)
    if H is None:
        raise RuntimeError("findHomography failed")

    np.savez(args.out, H=H, uv=uv, xy_mm=xy, ids=np.asarray(used_ids, dtype=np.int32), image=args.image)

    print(f"Saved {args.out}")
    print(f"Used ids: {used_ids}")
    print(f"Correspondences: {len(used_ids)}")


if __name__ == "__main__":
    main()
