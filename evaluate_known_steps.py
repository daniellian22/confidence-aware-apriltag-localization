#!/usr/bin/env python3
import argparse
import cv2
import numpy as np
from pupil_apriltags import Detector


def undistort_points(pts_uv, K, dist):
    pts = np.asarray(pts_uv, dtype=np.float32).reshape(-1, 1, 2)
    und = cv2.undistortPoints(pts, K, dist, P=K)
    return und.reshape(-1, 2)


def apply_homography_to_point(u, v, H):
    p = np.array([[[u, v]]], dtype=np.float32)
    out = cv2.perspectiveTransform(p, H)[0, 0]
    return float(out[0]), float(out[1])


def detect_tag_center(img_path, tag_id):
    img = cv2.imread(img_path)
    if img is None:
        raise RuntimeError(f"Could not read image: {img_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    det = Detector(
        families="tag36h11",
        nthreads=4,
        quad_decimate=1.0,
        quad_sigma=0.0,
        refine_edges=1,
        decode_sharpening=0.25,
        debug=0,
    )

    detections = det.detect(gray, estimate_tag_pose=False)
    for d in detections:
        if int(d.tag_id) == tag_id:
            c = np.array(d.center, dtype=np.float64)
            return c[0], c[1]

    raise RuntimeError(f"Tag {tag_id} not found in {img_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--intrinsics", default="camera_intrinsics.npz")
    ap.add_argument("--homography", default="pixel_to_grid_H.npz")
    ap.add_argument("--tag_id", type=int, required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--img_5mm", required=True)
    ap.add_argument("--img_10mm", required=True)
    args = ap.parse_args()

    intr = np.load(args.intrinsics)
    K = intr["K"].astype(np.float64)
    dist = intr["dist"].astype(np.float64)

    Hdata = np.load(args.homography)
    H = Hdata["H"].astype(np.float64)

    def image_to_grid(img_path):
        u, v = detect_tag_center(img_path, args.tag_id)
        und = undistort_points([[u, v]], K, dist)[0]
        x_mm, y_mm = apply_homography_to_point(und[0], und[1], H)
        return x_mm, y_mm

    x_ref, y_ref = image_to_grid(args.ref)
    x5, y5 = image_to_grid(args.img_5mm)
    x10, y10 = image_to_grid(args.img_10mm)

    dx5, dy5 = x5 - x_ref, y5 - y_ref
    d5 = float(np.hypot(dx5, dy5))
    err5 = abs(d5 - 5.0)

    dx10, dy10 = x10 - x_ref, y10 - y_ref
    d10 = float(np.hypot(dx10, dy10))
    err10 = abs(d10 - 10.0)

    print("Reference grid position:")
    print(f"  ({x_ref:.2f}, {y_ref:.2f}) mm")

    print("\n5 mm test:")
    print(f"  moved position = ({x5:.2f}, {y5:.2f}) mm")
    print(f"  dx = {dx5:.2f} mm, dy = {dy5:.2f} mm")
    print(f"  measured distance = {d5:.2f} mm")
    print(f"  absolute error = {err5:.2f} mm")

    print("\n10 mm test:")
    print(f"  moved position = ({x10:.2f}, {y10:.2f}) mm")
    print(f"  dx = {dx10:.2f} mm, dy = {dy10:.2f} mm")
    print(f"  measured distance = {d10:.2f} mm")
    print(f"  absolute error = {err10:.2f} mm")


if __name__ == "__main__":
    main()