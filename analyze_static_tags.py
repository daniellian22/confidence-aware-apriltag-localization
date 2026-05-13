#!/usr/bin/env python3
"""
Analyze static AprilTag photos and determine which tag moved.

Now supports GRID coordinates using a pixel->grid homography.

Important for this setup:
- Tag 10 is OFF the main plane.
- Tag 10 is excluded from grid-frame analysis.
- Grid coordinates are only computed for tags on the main plane.
"""

import argparse
import glob
import os
import numpy as np
import cv2
from pupil_apriltags import Detector


OFF_PLANE_TAG_ID = 10


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


def vector_stats(vectors_m):
    arr = np.asarray(vectors_m, dtype=np.float64)
    std_mm = np.std(arr, axis=0) * 1000.0
    span_mm = (np.max(arr, axis=0) - np.min(arr, axis=0)) * 1000.0
    radial_span_mm = np.linalg.norm(np.max(arr, axis=0) - np.min(arr, axis=0)) * 1000.0
    return std_mm, span_mm, radial_span_mm


def vector_stats_2d(vectors_mm):
    arr = np.asarray(vectors_mm, dtype=np.float64)
    std_mm = np.std(arr, axis=0)
    span_mm = np.max(arr, axis=0) - np.min(arr, axis=0)
    radial_span_mm = np.linalg.norm(np.max(arr, axis=0) - np.min(arr, axis=0))
    return std_mm, span_mm, radial_span_mm


def undistort_points(pts_uv, K, dist, P=None):
    pts = np.asarray(pts_uv, dtype=np.float32).reshape(-1, 1, 2)
    if P is None:
        P = K
    und = cv2.undistortPoints(pts, K, dist, P=P)
    return und.reshape(-1, 2)


def apply_homography_to_point(u, v, H):
    p = np.array([[[u, v]]], dtype=np.float32)
    out = cv2.perspectiveTransform(p, H)[0, 0]
    return float(out[0]), float(out[1])  # X_mm, Y_mm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_glob", default="static/*.jpeg")
    ap.add_argument("--intrinsics", default="camera_intrinsics.npz")
    ap.add_argument("--tag_size_mm", type=float, default=50.8, help="Physical side length of the black tag square")
    ap.add_argument("--origin_id", type=int, default=0, help="fixed reference tag")
    ap.add_argument("--expected_moved_id", type=int, default=11)
    ap.add_argument("--annotated_dir", default="static/annotated")
    ap.add_argument(
        "--annotate_on",
        choices=["original", "undistorted"],
        default="original",
        help="visualization image space for annotations",
    )
    ap.add_argument(
        "--homography",
        default=None,
        help="Optional pixel->grid homography npz containing H"
    )
    args = ap.parse_args()

    if args.origin_id == OFF_PLANE_TAG_ID:
        print(
            f"Warning: origin_id={OFF_PLANE_TAG_ID} is off-plane. "
            "Origin-frame is okay for 3D relative pose, but grid-frame ignores this tag."
        )

    images = sorted(glob.glob(args.images_glob))
    if not images:
        raise RuntimeError(f"No images found for glob: {args.images_glob}")

    data = np.load(args.intrinsics)
    K = data["K"].astype(np.float64)
    dist = data["dist"].astype(np.float64)

    H = None
    if args.homography is not None:
        if not os.path.exists(args.homography):
            print(f"Warning: homography file not found: {args.homography}. Continuing without grid-frame analysis.")
        else:
            H_data = np.load(args.homography)
            if "H" not in H_data:
                raise RuntimeError(f"Homography file {args.homography} missing key 'H'")
            H = H_data["H"].astype(np.float64)

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

    cam_frame = {}         # tag_id -> list of (image, t_cam)
    origin_frame = {}      # tag_id -> list of (image, t_origin)
    grid_frame = {}        # tag_id -> list of (image, [X_mm, Y_mm])

    per_image_dets = {}       # image basename -> list of (tag_id, corners, center)
    per_image_cam_t = {}      # image basename -> {tag_id: t_cam}
    per_image_origin_t = {}   # image basename -> {tag_id: t_origin}
    per_image_grid_xy = {}    # image basename -> {tag_id: [X_mm, Y_mm]}

    # Store raw detections separately so original-image annotation does not redo detection
    per_image_raw_dets = {}   # image basename -> list of (tag_id, corners, center)

    for path in images:
        img = cv2.imread(path)
        if img is None:
            continue

        h, w = img.shape[:2]

        # Compute camera matrix for undistorted image
        newK, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h))
        fx_u, fy_u = newK[0, 0], newK[1, 1]
        cx_u, cy_u = newK[0, 2], newK[1, 2]

        # Undistort image and use matching intrinsics for pose
        und = cv2.undistort(img, K, dist, None, newK)
        gray = cv2.cvtColor(und, cv2.COLOR_BGR2GRAY)

        detections = det.detect(
            gray,
            estimate_tag_pose=True,
            camera_params=(fx_u, fy_u, cx_u, cy_u),
            tag_size=tag_size_m,
        )

        frame = {}
        basename = os.path.basename(path)

        # Also cache raw-image detections for annotation in original space
        gray_raw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        raw_detections = det.detect(gray_raw, estimate_tag_pose=False)
        raw_draw_dets = []
        for d in raw_detections:
            tid = int(d.tag_id)
            corners = np.array(d.corners, dtype=np.float64).reshape(-1, 2)
            center = np.array(d.center, dtype=np.float64).reshape(2)
            raw_draw_dets.append((tid, corners, center))
        per_image_raw_dets[basename] = raw_draw_dets

        for d in detections:
            tid = int(d.tag_id)
            R = np.array(d.pose_R, dtype=np.float64)
            t = np.array(d.pose_t, dtype=np.float64).reshape(3)

            frame[tid] = make_T(R, t)
            per_image_cam_t.setdefault(basename, {})[tid] = t.copy()
            cam_frame.setdefault(tid, []).append((basename, t.copy()))

            corners = np.array(d.corners, dtype=np.float64).reshape(-1, 2)
            center = np.array(d.center, dtype=np.float64).reshape(2)
            per_image_dets.setdefault(basename, []).append((tid, corners, center))

            # Only compute grid coordinates for tags on the main plane.
            # Detection center is in UNDISTORTED image coordinates, so apply H directly there.
            if H is not None and tid != OFF_PLANE_TAG_ID:
                X_mm, Y_mm = apply_homography_to_point(center[0], center[1], H)
                per_image_grid_xy.setdefault(basename, {})[tid] = np.array([X_mm, Y_mm], dtype=np.float64)
                grid_frame.setdefault(tid, []).append((basename, np.array([X_mm, Y_mm], dtype=np.float64)))

        if args.origin_id in frame:
            T_cam_origin = frame[args.origin_id]
            T_origin_cam = inv_T(T_cam_origin)
            for tid, T_cam_tid in frame.items():
                T_origin_tid = T_origin_cam @ T_cam_tid
                per_image_origin_t.setdefault(basename, {})[tid] = T_origin_tid[:3, 3].copy()
                origin_frame.setdefault(tid, []).append((basename, T_origin_tid[:3, 3].copy()))

    print(f"Images processed: {len(images)}")
    print(f"Tags seen: {sorted(cam_frame.keys())}")
    print(f"Off-plane tag excluded from grid analysis: id={OFF_PLANE_TAG_ID}")

    print("\nCamera-frame motion summary (bigger span => moved more):")
    cam_rank = []
    for tid in sorted(cam_frame.keys()):
        vectors = [v for _, v in cam_frame[tid]]
        std_mm, span_mm, radial_span_mm = vector_stats(vectors)
        cam_rank.append((radial_span_mm, tid, len(vectors), std_mm, span_mm))
    cam_rank.sort(reverse=True)

    for radial_span_mm, tid, n, std_mm, span_mm in cam_rank:
        print(
            f"id={tid:>2} n={n:>2} radial_span={radial_span_mm:8.2f} mm "
            f"std_mm=[{std_mm[0]:6.2f},{std_mm[1]:6.2f},{std_mm[2]:6.2f}] "
            f"span_mm=[{span_mm[0]:6.2f},{span_mm[1]:6.2f},{span_mm[2]:6.2f}]"
        )

    if args.expected_moved_id in cam_frame:
        vectors = [v for _, v in cam_frame[args.expected_moved_id]]
        _, _, span = vector_stats(vectors)
        print(f"\nExpected moved tag id={args.expected_moved_id} camera-frame radial span: {span:.2f} mm")

    if args.origin_id in origin_frame:
        print(f"\nOrigin-frame motion summary (origin id={args.origin_id}):")
        origin_rank = []
        for tid in sorted(origin_frame.keys()):
            vectors = [v for _, v in origin_frame[tid]]
            std_mm, span_mm, radial_span_mm = vector_stats(vectors)
            origin_rank.append((radial_span_mm, tid, len(vectors), std_mm, span_mm))
        origin_rank.sort(reverse=True)

        for radial_span_mm, tid, n, std_mm, span_mm in origin_rank:
            print(
                f"id={tid:>2} n={n:>2} radial_span={radial_span_mm:8.2f} mm "
                f"std_mm=[{std_mm[0]:6.2f},{std_mm[1]:6.2f},{std_mm[2]:6.2f}] "
                f"span_mm=[{span_mm[0]:6.2f},{span_mm[1]:6.2f},{span_mm[2]:6.2f}]"
            )
    else:
        print(f"\nOrigin tag id={args.origin_id} was not available for origin-frame summary.")

    grid_rank = []
    if H is not None and grid_frame:
        print("\nGrid-frame motion summary (using homography; off-plane tag excluded):")
        for tid in sorted(grid_frame.keys()):
            vectors = [v for _, v in grid_frame[tid]]
            std_mm, span_mm, radial_span_mm = vector_stats_2d(vectors)
            grid_rank.append((radial_span_mm, tid, len(vectors), std_mm, span_mm))
        grid_rank.sort(reverse=True)

        for radial_span_mm, tid, n, std_mm, span_mm in grid_rank:
            print(
                f"id={tid:>2} n={n:>2} radial_span={radial_span_mm:8.2f} mm "
                f"std_mm=[{std_mm[0]:6.2f},{std_mm[1]:6.2f}] "
                f"span_mm=[{span_mm[0]:6.2f},{span_mm[1]:6.2f}]"
            )

        if args.expected_moved_id in grid_frame:
            vectors = [v for _, v in grid_frame[args.expected_moved_id]]
            _, _, span = vector_stats_2d(vectors)
            print(f"\nExpected moved tag id={args.expected_moved_id} grid-frame radial span: {span:.2f} mm")

    # Choose which ranking to trust most
    if grid_rank:
        top_tid = grid_rank[0][1]
        ranking_source = "grid frame"
    elif cam_rank:
        top_tid = cam_rank[0][1]
        ranking_source = "camera frame"
    else:
        raise RuntimeError("No valid detections found.")

    print(f"\nMost-moving tag in {ranking_source}: id={top_tid}")
    if top_tid == args.expected_moved_id:
        print("Result: expected moved tag matches top-moving tag ✅")
    else:
        print("Result: expected moved tag does NOT match top-moving tag ⚠️")

    os.makedirs(args.annotated_dir, exist_ok=True)

    for path in images:
        img_raw = cv2.imread(path)
        if img_raw is None:
            continue

        h, w = img_raw.shape[:2]
        newK, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h))

        if args.annotate_on == "undistorted":
            img = cv2.undistort(img_raw, K, dist, None, newK)
            draw_dets = per_image_dets.get(os.path.basename(path), [])
        else:
            img = img_raw.copy()
            draw_dets = per_image_raw_dets.get(os.path.basename(path), [])

        name = os.path.basename(path)

        ax0 = (30, 30)
        cv2.arrowedLine(img, ax0, (ax0[0] + 70, ax0[1]), (0, 255, 255), 2, tipLength=0.2)
        cv2.arrowedLine(img, ax0, (ax0[0], ax0[1] + 70), (255, 255, 0), 2, tipLength=0.2)
        cv2.putText(img, "+u", (ax0[0] + 75, ax0[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, "+v", (ax0[0] - 5, ax0[1] + 88), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)

        for tid, corners, center in draw_dets:
            pts = corners.astype(np.int32)
            center_draw = center.astype(np.int32)

            if tid == top_tid:
                color = (0, 0, 255)
                label = f"MOVED id={tid}"
                thickness = 3
            else:
                color = (80, 255, 80)
                label = f"id={tid}"
                if tid == OFF_PLANE_TAG_ID:
                    label = f"id={tid} (off-plane)"
                thickness = 2

            cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness)
            c = tuple(center_draw)
            cv2.circle(img, c, 6, color, -1)

            tx = int(np.clip(c[0] + 8, 0, img.shape[1] - 1))
            ty = int(np.clip(c[1] - 8, 0, img.shape[0] - 1))
            cv2.putText(
                img,
                label,
                (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA,
            )

            if tid == top_tid:
                grid_xy = per_image_grid_xy.get(name, {}).get(tid, None)
                t_cam = per_image_cam_t.get(name, {}).get(tid, None)
                t_origin = per_image_origin_t.get(name, {}).get(tid, None)

                if grid_xy is not None:
                    coord_text = f"grid xy[mm]=({grid_xy[0]:.1f}, {grid_xy[1]:.1f})"
                elif t_origin is not None:
                    coord_text = f"origin xy[mm]=({t_origin[0]*1000.0:.1f}, {t_origin[1]*1000.0:.1f})"
                elif t_cam is not None:
                    coord_text = f"camera xy[mm]=({t_cam[0]*1000.0:.1f}, {t_cam[1]*1000.0:.1f})"
                else:
                    coord_text = "xy unavailable"

                tx2 = int(np.clip(c[0] + 8, 0, img.shape[1] - 1))
                ty2 = int(np.clip(c[1] + 18, 0, img.shape[0] - 1))
                cv2.putText(
                    img,
                    coord_text,
                    (tx2, ty2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

        cv2.putText(
            img,
            f"Inferred moved tag: id={top_tid}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

        out_path = os.path.join(args.annotated_dir, name)
        cv2.imwrite(out_path, img)

    print(f"Annotated images saved to: {args.annotated_dir}")
    print(f"Annotation space: {args.annotate_on}")

    print("\nPer-image origin-frame coordinates:\n")

    for name in sorted(per_image_origin_t.keys()):
        print(name)
        for tid in sorted(per_image_origin_t[name].keys()):
            t = per_image_origin_t[name][tid]
            print(f"  tag {tid}: x_mm={t[0]*1000:.1f}, y_mm={t[1]*1000:.1f}, z_mm={t[2]*1000:.1f}")
        print()


if __name__ == "__main__":
    main()