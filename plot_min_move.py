import cv2
import numpy as np
import glob
import os
import re
import matplotlib.pyplot as plt
from collections import defaultdict
from pupil_apriltags import Detector

images_glob = "min_move_2/*.JPG"
intrinsics_path = "camera_intrinsics.npz"
tag_size_mm = 64
origin_id = 11
target_id = 13

# grouping:
# IMG_2762-2766 = 1 mm
# IMG_2767-2771 = 2 mm
# IMG_2772-2776 = 3 mm
# IMG_2777-2778 = 0 mm
def get_position_from_filename(path):
    name = os.path.basename(path)
    m = re.search(r"IMG_(\d+)\.JPG", name)
    if not m:
        return None

    num = int(m.group(1))

    if 2802 <= num <= 2804:
        return 0
    elif 2805 <= num <= 2807:
        return 3
    elif 2808 <= num <= 2810:
        return 5
    elif 2811 <= num <= 2813:
        return 6
    elif 2814 <= num <= 2816:
        return 7
    elif 2817 <= num <= 2819:
        return 10
    elif 2820 <= num <= 2824:
        return 20

    return None

def make_T(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t.reshape(3)
    return T

def inv_T(T):
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti

data = np.load(intrinsics_path)
K = data["K"].astype(np.float64)
dist = data["dist"].astype(np.float64)

detector = Detector(
    families="tag36h11",
    nthreads=4,
    quad_decimate=1.0,
    quad_sigma=0.0,
    refine_edges=1,
    decode_sharpening=0.25,
    debug=0,
)

data_by_pos = defaultdict(list)

for path in sorted(glob.glob(images_glob)):
    pos = get_position_from_filename(path)
    if pos is None:
        continue

    img = cv2.imread(path)
    if img is None:
        continue

    und = cv2.undistort(img, K, dist)
    gray = cv2.cvtColor(und, cv2.COLOR_BGR2GRAY)

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    detections = detector.detect(
        gray,
        estimate_tag_pose=True,
        camera_params=(fx, fy, cx, cy),
        tag_size=tag_size_mm / 1000.0,
    )

    poses = {}
    for d in detections:
        poses[int(d.tag_id)] = make_T(np.array(d.pose_R), np.array(d.pose_t).reshape(3))

    if origin_id not in poses or target_id not in poses:
        print(f"Skipping {path}: missing origin or target tag")
        continue

    T_origin_cam = inv_T(poses[origin_id])
    T_origin_target = T_origin_cam @ poses[target_id]
    xyz_mm = T_origin_target[:3, 3] * 1000.0

    data_by_pos[pos].append(xyz_mm[0])  # x-coordinate

print("Detected groups:")
for pos in sorted(data_by_pos):
    print(pos, "mm:", len(data_by_pos[pos]), "images")

positions = sorted(data_by_pos.keys())
means = [np.mean(data_by_pos[p]) for p in positions]
stds = [np.std(data_by_pos[p]) for p in positions]

plt.figure()
plt.errorbar(positions, means, yerr=stds, fmt="o-", capsize=5)
plt.xlabel("True Movement (mm)")
plt.ylabel("Measured X Position of Tag 13 (mm)")
plt.title("Minimum Detectable Movement Test")
plt.grid(True)
plt.savefig("min_movement_plot.png", dpi=300, bbox_inches="tight")

print("Saved plot to min_movement_plot.png")
