import cv2
import numpy as np
from pupil_apriltags import Detector

# ===== CONFIG =====
image_path = "ground_check/IMG_2580.jpeg"
intrinsics_path = "camera_intrinsics.npz"
tag_size_mm = 64
output_path = "annotated_test.jpg"
# ==================

data = np.load(intrinsics_path)
K = data["K"].astype(np.float64)
dist = data["dist"].astype(np.float64)

img = cv2.imread(image_path)
if img is None:
    raise RuntimeError(f"Could not read image: {image_path}")

# Match analyze_static_tags.py exactly
und = cv2.undistort(img, K, dist)
gray = cv2.cvtColor(und, cv2.COLOR_BGR2GRAY)

fx, fy = K[0, 0], K[1, 1]
cx, cy = K[0, 2], K[1, 2]

detector = Detector(
    families="tag36h11",
    nthreads=4,
    quad_decimate=1.0,
    quad_sigma=0.0,
    refine_edges=1,
    decode_sharpening=0.25,
    debug=0,
)

results = detector.detect(
    gray,
    estimate_tag_pose=True,
    camera_params=(fx, fy, cx, cy),
    tag_size=tag_size_mm / 1000.0,
)

print(f"Detected {len(results)} tags\n")

for d in results:
    tid = int(d.tag_id)
    center = np.array(d.center).reshape(2)
    t = np.array(d.pose_t).reshape(3)

    print(f"Tag {tid}")
    print(f"  center (u,v): ({center[0]:.1f}, {center[1]:.1f})")
    print(f"  pose_t [mm]: x={t[0]*1000:.1f}, y={t[1]*1000:.1f}, z={t[2]*1000:.1f}")
    print()

    corners = np.array(d.corners, dtype=np.int32).reshape(-1, 2)
    c = tuple(center.astype(int))

    cv2.polylines(und, [corners], True, (0, 255, 0), 2)
    cv2.circle(und, c, 5, (0, 0, 255), -1)
    cv2.putText(
        und,
        f"id={tid}",
        (c[0] + 8, c[1] - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        und,
        f"z={t[2]*1000:.1f}mm",
        (c[0] + 8, c[1] + 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )

cv2.imwrite(output_path, und)
print(f"Saved annotated image to: {output_path}")