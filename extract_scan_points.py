import argparse
import numpy as np
import cv2
import pandas as pd
from pupil_apriltags import Detector  # pose_R, pose_t available with estimate_tag_pose=True

def make_T(R, t):
    T = np.eye(4, dtype=np.float64)
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

def median_T(transforms):
    # Median only over translation; rotations ignored for this use-case.
    ts = np.stack([T[:3, 3] for T in transforms], axis=0)
    tmed = np.median(ts, axis=0)
    T = np.eye(4)
    T[:3, 3] = tmed
    return T

def build_relative_map(samples_cam_T_tag, origin_id):
    """
    samples_cam_T_tag: list of dict {tag_id: T_cam_tag} for initial frames
    Returns:
      rel_map[tag_id] = T_origin_tag  (origin <- tag)
    """
    rel = {}
    for frame_dict in samples_cam_T_tag:
        if origin_id not in frame_dict:
            continue
        T_cam_origin = frame_dict[origin_id]
        T_origin_cam = inv_T(T_cam_origin)
        for tid, T_cam_tid in frame_dict.items():
            if tid == origin_id:
                continue
            T_origin_tid = T_origin_cam @ T_cam_tid
            rel.setdefault(tid, []).append(T_origin_tid)
    # Robust aggregate (median translation)
    rel_map = {}
    for tid, Ts in rel.items():
        rel_map[tid] = median_T(Ts)
    return rel_map

def estimate_T_cam_origin(frame_dict, origin_id, static_rel_map):
    """
    frame_dict: {tag_id: T_cam_tag} for current frame
    If origin visible -> return T_cam_origin
    else use any other static tag i:
      T_cam_origin = T_cam_i * inv(T_origin_i)
    where T_origin_i = origin <- i from static_rel_map
    """
    if origin_id in frame_dict:
        return frame_dict[origin_id], True

    # fallback using any static tag present
    for tid, T_cam_tid in frame_dict.items():
        if tid in static_rel_map:
            T_origin_tid = static_rel_map[tid]
            T_tid_origin = inv_T(T_origin_tid)  # tid <- origin
            # We want cam <- origin:
            # T_cam_tid = (cam <- origin) * (origin <- tid) ??? careful:
            # We have T_origin_tid = origin <- tid.
            # From derivation: T_cam_origin = T_cam_tid * inv(T_origin_tid).
            T_cam_origin = T_cam_tid @ inv_T(T_origin_tid)
            return T_cam_origin, True

    return None, False

def estimate_T_cam_tool(frame_dict, tool_id, tool_rel_map):
    """
    tool frame is defined as the tool_id tag frame.
    If tool_id visible -> return T_cam_toolid
    else fallback using another tool tag j with known T_toolid_j (toolid <- j):
      T_cam_toolid = T_cam_j * inv(T_toolid_j)
    """
    if tool_id in frame_dict:
        return frame_dict[tool_id], True

    for tid, T_cam_tid in frame_dict.items():
        if tid in tool_rel_map:
            T_tool_tid = tool_rel_map[tid]  # tool_id <- tid
            T_cam_tool = T_cam_tid @ inv_T(T_tool_tid)
            return T_cam_tool, True

    return None, False

def segment_stops(times, positions, v_thresh=0.01, min_stop_s=0.12, merge_gap_s=0.10, merge_dist_m=0.002):
    """
    times: (N,) seconds
    positions: (N,3) meters (NaN where invalid)
    Returns list of segments [ (start_idx, end_idx) ] inclusive indices.
    """
    valid = np.all(np.isfinite(positions), axis=1)
    # Speed estimate
    v = np.full(len(times), np.inf, dtype=np.float64)
    for i in range(1, len(times)):
        if valid[i] and valid[i-1]:
            dt = times[i] - times[i-1]
            if dt > 0:
                v[i] = np.linalg.norm(positions[i] - positions[i-1]) / dt

    is_stop = valid & (v < v_thresh)

    segs = []
    i = 0
    while i < len(times):
        if not is_stop[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(times) and is_stop[j + 1]:
            j += 1
        # duration
        if times[j] - times[i] >= min_stop_s:
            segs.append([i, j])
        i = j + 1

    # merge close segments (tiny gaps or detection dropouts)
    merged = []
    for s in segs:
        if not merged:
            merged.append(s)
            continue
        prev = merged[-1]
        gap = times[s[0]] - times[prev[1]]
        prev_pos = np.nanmedian(positions[prev[0]:prev[1]+1], axis=0)
        cur_pos = np.nanmedian(positions[s[0]:s[1]+1], axis=0)
        if gap <= merge_gap_s and np.linalg.norm(cur_pos - prev_pos) <= merge_dist_m:
            prev[1] = s[1]
        else:
            merged.append(s)
    return merged

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--intrinsics", required=True, help="camera_intrinsics.npz from calibration")
    ap.add_argument("--tag_size_mm", type=float, default=50.8, help="Physical side length of the black tag square")
    ap.add_argument("--origin_static_id", type=int, default=0)
    ap.add_argument("--tool_id", type=int, default=10)
    ap.add_argument("--static_ids", type=str, default="0,1,2,3")
    ap.add_argument("--tool_ids", type=str, default="10,11,12,13")
    ap.add_argument("--target_fps", type=float, default=10.0, help="process rate")
    ap.add_argument("--skip_seconds", type=float, default=0.0, help="skip initial seconds")
    ap.add_argument("--max_points", type=int, default=6000, help="expected number of scan stops")
    ap.add_argument("--out_csv", default="scan_points.csv")
    args = ap.parse_args()

    static_ids = set(int(x) for x in args.static_ids.split(",") if x.strip() != "")
    tool_ids = set(int(x) for x in args.tool_ids.split(",") if x.strip() != "")

    data = np.load(args.intrinsics)
    K = data["K"].astype(np.float64)
    dist = data["dist"].astype(np.float64)

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    tag_size_m = args.tag_size_mm / 1000.0

    det = Detector(
        families="tag36h11",
        nthreads=4,
        quad_decimate=2.0,
        quad_sigma=0.0,
        refine_edges=1,
        decode_sharpening=0.25,
        debug=0,
    )

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 1e-3:
        fps = 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ds = max(1, int(round(fps / args.target_fps)))

    # Move to skip_seconds
    if args.skip_seconds > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, args.skip_seconds * 1000.0)

    # Collect a short initialization window for relative maps (first ~5 seconds processed)
    init_frames = int(round(5.0 * args.target_fps))
    samples = []

    positions = []
    times = []

    idx_processed = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        # downsample in time
        if (frame_idx % ds) != 0:
            continue

        t_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        # Undistort before detection
        und = cv2.undistort(frame, K, dist)
        gray = cv2.cvtColor(und, cv2.COLOR_BGR2GRAY)

        detections = det.detect(
            gray,
            estimate_tag_pose=True,
            camera_params=(fx, fy, cx, cy),
            tag_size=tag_size_m,
        )

        frame_dict = {}
        for d in detections:
            tid = int(d.tag_id)
            if (tid in static_ids) or (tid in tool_ids):
                R = np.array(d.pose_R, dtype=np.float64)
                t = np.array(d.pose_t, dtype=np.float64).reshape(3)
                frame_dict[tid] = make_T(R, t)

        if idx_processed < init_frames:
            samples.append(frame_dict.copy())

        times.append(t_sec)
        positions.append([np.nan, np.nan, np.nan])  # filled after maps exist
        idx_processed += 1

    cap.release()
    times = np.array(times, dtype=np.float64)
    positions = np.array(positions, dtype=np.float64)

    # Build static relative map (origin <- static_i)
    static_rel_map = build_relative_map(samples, args.origin_static_id)

    # Build tool relative map (tool_id <- tool_j)
    # Define tool frame as tool_id tag frame
    tool_rel_map = {}
    for frame_dict in samples:
        if args.tool_id not in frame_dict:
            continue
        T_cam_tool = frame_dict[args.tool_id]
        T_tool_cam = inv_T(T_cam_tool)
        for tid, T_cam_tid in frame_dict.items():
            if tid == args.tool_id:
                continue
            if tid in tool_ids:
                T_tool_tid = T_tool_cam @ T_cam_tid  # tool_id <- tid
                tool_rel_map.setdefault(tid, []).append(T_tool_tid)
    for tid, Ts in tool_rel_map.items():
        tool_rel_map[tid] = median_T(Ts)

    # Second pass: recompute positions from stored detections by re-reading video (simpler & robust).
    # (We re-read because we didn’t store all frame_dicts to keep memory down for long videos.)
    cap = cv2.VideoCapture(args.video)
    if args.skip_seconds > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, args.skip_seconds * 1000.0)

    pos_list = []
    t_list = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if (frame_idx % ds) != 0:
            continue

        t_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        und = cv2.undistort(frame, K, dist)
        gray = cv2.cvtColor(und, cv2.COLOR_BGR2GRAY)

        detections = det.detect(
            gray,
            estimate_tag_pose=True,
            camera_params=(fx, fy, cx, cy),
            tag_size=tag_size_m,
        )

        frame_dict = {}
        for d in detections:
            tid = int(d.tag_id)
            if (tid in static_ids) or (tid in tool_ids):
                R = np.array(d.pose_R, dtype=np.float64)
                t = np.array(d.pose_t, dtype=np.float64).reshape(3)
                frame_dict[tid] = make_T(R, t)

        T_cam_origin, ok_origin = estimate_T_cam_origin(frame_dict, args.origin_static_id, static_rel_map)
        T_cam_tool, ok_tool = estimate_T_cam_tool(frame_dict, args.tool_id, tool_rel_map)

        if ok_origin and ok_tool:
            T_origin_tool = inv_T(T_cam_origin) @ T_cam_tool
            pos = T_origin_tool[:3, 3]  # meters
        else:
            pos = np.array([np.nan, np.nan, np.nan], dtype=np.float64)

        t_list.append(t_sec)
        pos_list.append(pos)

    cap.release()
    t_list = np.array(t_list, dtype=np.float64)
    pos_list = np.vstack(pos_list)

    # Segment stops -> one point per stop
    segs = segment_stops(t_list, pos_list, v_thresh=0.01, min_stop_s=0.12)

    stop_points = []
    stop_times = []
    for (a, b) in segs:
        p = np.nanmedian(pos_list[a:b+1], axis=0)
        tt = np.nanmedian(t_list[a:b+1])
        stop_points.append(p)
        stop_times.append(tt)

    stop_points = np.vstack(stop_points) if stop_points else np.zeros((0, 3))
    stop_times = np.array(stop_times)

    # Keep the first max_points stops (you can adjust skip_seconds to align)
    if stop_points.shape[0] > args.max_points:
        stop_points = stop_points[:args.max_points]
        stop_times = stop_times[:args.max_points]

    df = pd.DataFrame({
        "idx": np.arange(len(stop_times)),
        "t_sec": stop_times,
        "x_m": stop_points[:, 0],
        "y_m": stop_points[:, 1],
        "z_m": stop_points[:, 2],
    })
    df.to_csv(args.out_csv, index=False)
    print(f"Wrote {args.out_csv} with {len(df)} points")

if __name__ == "__main__":
    main()
