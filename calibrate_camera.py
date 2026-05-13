import argparse
import glob
import numpy as np
import cv2


def compute_per_view_rmse(objpoints, imgpoints, rvecs, tvecs, K, dist):
    errors = []
    for objp, imgp, rvec, tvec in zip(objpoints, imgpoints, rvecs, tvecs):
        projected, _ = cv2.projectPoints(objp, rvec, tvec, K, dist)
        err = cv2.norm(imgp, projected, cv2.NORM_L2) / np.sqrt(len(projected))
        errors.append(float(err))
    return errors

def calibrate_from_images(image_glob: str, pattern_size, square_size_m):
    images = sorted(glob.glob(image_glob))
    if len(images) < 10:
        raise RuntimeError(f"Need >=10 calibration images, found {len(images)}")

    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    objp *= square_size_m

    objpoints = []
    imgpoints = []
    used_images = []
    img_shape = None

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)

    for path in images:
        img = cv2.imread(path)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_shape = gray.shape[::-1]

        ok, corners = cv2.findChessboardCorners(gray, pattern_size, None)
        if not ok:
            continue

        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        objpoints.append(objp)
        imgpoints.append(corners)
        used_images.append(path)

    if len(objpoints) < 10:
        raise RuntimeError(f"Too few valid chessboard detections: {len(objpoints)}")

    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, img_shape, None, None
    )
    return ret, K, dist, rvecs, tvecs, objpoints, imgpoints, used_images

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_glob", required=True, help="e.g., calib/*.jpg")
    ap.add_argument("--pattern_cols", type=int, required=True, help="inner corners along x")
    ap.add_argument("--pattern_rows", type=int, required=True, help="inner corners along y")
    ap.add_argument("--square_size_mm", type=float, required=True)
    ap.add_argument("--out", default="camera_intrinsics.npz")
    ap.add_argument("--report_csv", default="calibration_report.csv")
    ap.add_argument("--top_bad", type=int, default=5, help="how many worst images to print")
    args = ap.parse_args()

    pattern_size = (args.pattern_cols, args.pattern_rows)
    square_size_m = args.square_size_mm / 1000.0

    rms, K, dist, rvecs, tvecs, objpoints, imgpoints, used_images = calibrate_from_images(
        args.images_glob, pattern_size, square_size_m
    )
    per_view_rmse = compute_per_view_rmse(objpoints, imgpoints, rvecs, tvecs, K, dist)

    np.savez(args.out, K=K, dist=dist, rms=rms)
    print(f"Saved {args.out}")
    print(f"RMS reprojection error: {rms:.4f}")

    rmse_arr = np.array(per_view_rmse, dtype=np.float64)
    print(f"Used images: {len(used_images)}")
    print(f"Per-image RMSE mean: {rmse_arr.mean():.4f} px")
    print(f"Per-image RMSE median: {np.median(rmse_arr):.4f} px")
    print(f"Per-image RMSE max: {rmse_arr.max():.4f} px")

    order = np.argsort(rmse_arr)[::-1]
    top_bad = max(0, min(args.top_bad, len(used_images)))
    if top_bad > 0:
        print(f"Worst {top_bad} images by per-image RMSE:")
        for rank, idx in enumerate(order[:top_bad], start=1):
            print(f"  {rank:>2}. {used_images[idx]} -> {rmse_arr[idx]:.4f} px")

    with open(args.report_csv, "w", encoding="utf-8") as f:
        f.write("image_path,per_image_rmse_px\n")
        for path, err in zip(used_images, per_view_rmse):
            f.write(f"{path},{err:.6f}\n")
    print(f"Saved report: {args.report_csv}")

if __name__ == "__main__":
    main()
