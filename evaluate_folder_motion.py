#!/usr/bin/env python3
import glob
import os
import cv2
import numpy as np
from pupil_apriltags import Detector
import matplotlib.pyplot as plt

TAG_ID = 11


def undistort_points(pts_uv, K, dist):
    pts = np.asarray(pts_uv, dtype=np.float32).reshape(-1, 1, 2)
    und = cv2.undistortPoints(pts, K, dist, P=K)
    return und.reshape(-1, 2)


def apply_homography(u, v, H):
    p = np.array([[[u, v]]], dtype=np.float32)
    out = cv2.perspectiveTransform(p, H)[0, 0]
    return float(out[0]), float(out[1])


def get_positions(folder, K, dist, H, detector):
    rows = []
    paths = sorted(glob.glob(folder + "/*.jpeg"))
    for path in paths:
        img = cv2.imread(path)
        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        detections = detector.detect(gray, estimate_tag_pose=False)

        found = False
        for d in detections:
            if int(d.tag_id) == TAG_ID:
                u, v = d.center
                und = undistort_points([[u, v]], K, dist)[0]
                x, y = apply_homography(und[0], und[1], H)
                rows.append((os.path.basename(path), x, y))
                found = True
                break

        if not found:
            print(f"Warning: tag {TAG_ID} not found in {path}")

    return rows


def evaluate_steps(rows, true_step_mm, label):
    print(f"\n===== {true_step_mm} mm STEP TEST =====")

    if len(rows) < 2:
        print("Not enough images")
        return

    dists = []

    for i in range(len(rows) - 1):
        a = rows[i]
        b = rows[i + 1]
        dx = b[1] - a[1]
        dy = b[2] - a[2]
        dist = float(np.hypot(dx, dy))
        dists.append(dist)

        err = abs(dist - true_step_mm)

        print(f"{a[0]} -> {b[0]}:")
        print(f"  measured = {dist:.2f} mm, error = {err:.2f} mm")

    dists = np.asarray(dists)
    steps = np.arange(1, len(dists) + 1)

    # ---- create output folder ----
    import os
    os.makedirs("plots", exist_ok=True)

    # ---- plot 1: measured step ----
    plt.figure()
    plt.plot(steps, dists, marker='o')
    plt.axhline(true_step_mm)

    plt.ylim(2, 11)  # <-- ADD THIS

    plt.xlabel("Step Index")
    plt.ylabel("Measured Step (mm)")
    plt.title(f"{label}: Measured Step vs Ground Truth")

    plt.savefig(f"plots/{label}_steps.png")
    plt.close()
    # ---- plot 2: error ----
    errors = np.abs(dists - true_step_mm)

    plt.figure()
    plt.plot(steps, errors, marker='o')

    plt.xlabel("Step Index")
    plt.ylabel("Absolute Error (mm)")
    plt.title(f"{label}: Error per Step")

    plt.savefig(f"plots/{label}_error.png")
    plt.close()


def main():
    intr = np.load("camera_intrinsics.npz")
    K = intr["K"]
    dist = intr["dist"]

    H = np.load("pixel_to_grid_H.npz")["H"]

    detector = Detector(families="tag36h11")

    rows_5 = get_positions("5mm", K, dist, H, detector)
    rows_10 = get_positions("10mm", K, dist, H, detector)

    print("5mm positions:")
    for r in rows_5:
        print(f"  {r[0]}: ({r[1]:.2f}, {r[2]:.2f})")

    print("\n10mm positions:")
    for r in rows_10:
        print(f"  {r[0]}: ({r[1]:.2f}, {r[2]:.2f})")

    evaluate_steps(rows_5, 5.0, "5mm Test")
    evaluate_steps(rows_10, 10.0, "10mm Test")


if __name__ == "__main__":
    main()