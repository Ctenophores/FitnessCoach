import os
import json
import numpy as np
import cv2
from typing import List
from scipy.interpolate import interp1d
import mediapipe as mp
import re

POSE_CONNECTIONS = mp.solutions.holistic.POSE_CONNECTIONS

def load_skeleton_sequence(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    frames = []
    for frame_data in data:
        pose = frame_data.get('pose', [])
        if not pose:
            pose = [[0, 0, 0, 0]] * 33
        arr = np.array(pose, dtype=np.float32).reshape(33, 4)
        frames.append(arr)
    if len(frames) == 0:
        return None
    return np.stack(frames, axis=0)  # (T, 33, 4)

def compute_transform_params(pose, action):
    eps = 1e-8
    left_ankle = pose[27, :3]
    right_ankle = pose[28, :3]
    left_wrist = pose[15, :3]
    right_wrist = pose[16, :3]
    left_shoulder = pose[11, :3]
    right_shoulder = pose[12, :3]
    left_hip = pose[23, :3]
    right_hip = pose[24, :3]

    shoulder_mid = (left_shoulder + right_shoulder) / 2.0
    hip_mid = (left_hip + right_hip) / 2.0
    ankle_mid = (left_ankle + right_ankle) / 2.0
    wrist_mid = (left_wrist + right_wrist) / 2.0

    if action == "BodyWeightSquats":
        origin = ankle_mid
        scale = np.linalg.norm(shoulder_mid - origin)
        x_axis = right_hip - left_hip
        if np.linalg.norm(x_axis) < eps:
            x_axis = np.array([1, 0, 0], dtype=np.float32)
        else:
            x_axis /= (np.linalg.norm(x_axis) + eps)
        up_vector = shoulder_mid - hip_mid
        y_axis = up_vector - np.dot(up_vector, x_axis) * x_axis
        if np.linalg.norm(y_axis) < eps:
            y_axis = np.array([0, 1, 0], dtype=np.float32)
        else:
            y_axis /= (np.linalg.norm(y_axis) + eps)
    elif action == "PushUps":
        origin = (left_wrist + right_wrist + left_ankle + right_ankle) / 4.0
        scale = np.linalg.norm(ankle_mid - shoulder_mid)
        x_axis = right_wrist - left_wrist
        if np.linalg.norm(x_axis) < eps:
            x_axis = np.array([1, 0, 0], dtype=np.float32)
        else:
            x_axis /= (np.linalg.norm(x_axis) + eps)
        body_vector = ankle_mid - wrist_mid
        if np.linalg.norm(body_vector) < eps:
            body_vector = np.array([0, 1, 0], dtype=np.float32)
        else:
            body_vector /= (np.linalg.norm(body_vector) + eps)
        y_axis = body_vector
    else:
        origin = hip_mid
        scale = np.linalg.norm(shoulder_mid - origin)
        x_axis = right_hip - left_hip
        if np.linalg.norm(x_axis) < eps:
            x_axis = np.array([1, 0, 0], dtype=np.float32)
        else:
            x_axis /= (np.linalg.norm(x_axis) + eps)

        up_vector = shoulder_mid - origin
        y_axis = up_vector - np.dot(up_vector, x_axis) * x_axis
        if np.linalg.norm(y_axis) < eps:
            y_axis = np.array([0, 1, 0], dtype=np.float32)
        else:
            y_axis /= (np.linalg.norm(y_axis) + eps)

    z_axis = np.cross(x_axis, y_axis)
    if np.linalg.norm(z_axis) < eps:
        z_axis = np.array([0, 0, 1], dtype=np.float32)
    else:
        z_axis /= (np.linalg.norm(z_axis) + eps)
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= (np.linalg.norm(y_axis) + eps)

    R = np.stack([x_axis, y_axis, z_axis], axis=1)  # (3,3)

    return origin, R, scale

def normalize_pose_and_store_params(sequence, action=None):
    sequence = np.array(sequence, dtype=np.float32)
    num_frames = sequence.shape[0]
    norm_seq = np.empty_like(sequence)
    transform_params = []

    origin = None
    R = None
    scale = None

    origin, R, scale = compute_transform_params(sequence[0], action)

    for i in range(num_frames):
        pose = sequence[i]  # (33,4)

        assert (origin is not None and R is not None and scale is not None), "origin/R/scale need to be NOT None!"

        xyz = pose[:, :3] - origin
        xyz_rot = xyz @ R

        scale = max(scale, 1e-10)
        xyz_rot /= scale

        norm_seq[i, :, :3] = xyz_rot
        norm_seq[i, :, 3] = pose[:, 3]
        transform_params.append({
            "center": origin,
            "R": R,
            "scale": scale
        })

    return norm_seq, transform_params

def time_resample(sequence, new_len=100):
    T = sequence.shape[0]
    if T == 0:
        return sequence
    x_old = np.linspace(0, 1, T)
    x_new = np.linspace(0, 1, new_len)
    out_seq = np.zeros((new_len, 33, 4), dtype=np.float32)
    for i in range(33):
        for j in range(4):
            f = interp1d(x_old, sequence[:, i, j], kind='linear', fill_value='extrapolate')
            out_seq[:, i, j] = f(x_new)
    return out_seq

def build_perfect_action(json_paths: List[str], target_len, action):
    sequences = []
    for p in json_paths:
        seq = load_skeleton_sequence(p)
        if seq is None or seq.shape[0] < 2:
            continue
        norm_seq, _ = normalize_pose_and_store_params(seq, action)
        seq_resampled = time_resample(norm_seq, new_len=target_len)
        sequences.append(seq_resampled)
    if not sequences:
        return None
    all_data = np.stack(sequences, axis=0)  # (N, target_len, 33, 4)
    perfect_seq = np.mean(all_data, axis=0)  # (target_len, 33, 4)
    return perfect_seq

def project_with_stored_params(normalized_seq, transform_params):
    num_frames = normalized_seq.shape[0]
    projected_seq = np.empty_like(normalized_seq)
    for i in range(num_frames):
        xyz_norm = normalized_seq[i, :, :3]  # (33,3)
        vis = normalized_seq[i, :, 3]
        params = transform_params[i]
        center = params["center"]
        R = params["R"]
        scale = params["scale"]
        R_inv = R.T
        xyz = xyz_norm * scale
        xyz = xyz @ R_inv
        xyz = xyz + center
        projected_seq[i, :, :3] = xyz
        projected_seq[i, :, 3] = vis
    return projected_seq

def draw_skeleton_on_frame_custom(frame, skeleton_33x4, width, height, scale=1.0, offset_x=0, offset_y=0, line_color=(0,255,0), point_color=(0,0,255), inverse=False, point_radius=4, line_thickness=2):
    points_2d = []
    for i in range(33):
        x, y = skeleton_33x4[i, 0], skeleton_33x4[i, 1]
        x_px = int(x * width * scale + offset_x)
        if inverse:
            y_px = int((-y) * height * scale + offset_y)
        else:
            y_px = int(y * height * scale + offset_y)
        points_2d.append((x_px, y_px))
    for conn in POSE_CONNECTIONS:
        start_idx, end_idx = conn
        x1, y1 = points_2d[start_idx]
        x2, y2 = points_2d[end_idx]
        cv2.line(frame, (x1, y1), (x2, y2), line_color, thickness=line_thickness)
    for (px, py) in points_2d:
        cv2.circle(frame, (px, py), point_radius, point_color, -1)

def overlay_skeleton_in_corner(frame, skeleton, width, height, region_ratio=1 / 3, scale=1.0, offset_x=0, offset_y=0, line_color=(255, 255, 255), point_color=(0, 0, 255), point_radius=4, line_thickness=2):
    region_w = int(width * region_ratio)
    region_h = int(height * region_ratio)

    roi = np.ones((region_h, region_w, 3), dtype=np.uint8) * 255

    draw_skeleton_on_frame_custom(roi, skeleton, region_w, region_h,
                                  scale=scale, offset_x=offset_x, offset_y=offset_y,
                                  line_color=line_color, point_color=point_color,
                                  point_radius=point_radius, line_thickness=line_thickness)

    x_start = width - region_w
    y_start = 0
    frame[y_start:y_start + region_h, x_start:x_start + region_w] = roi

def compute_action_standard_score(original_seq, projected_seq, base_weight=10.0, keypoint_weights=None):
    diff = original_seq[:, :, :3] - projected_seq[:, :, :3]
    distances = np.linalg.norm(diff, axis=-1)
    if keypoint_weights is None:
        keypoint_weights = np.ones(33)
    else:
        keypoint_weights = np.array(keypoint_weights)
        if keypoint_weights.shape[0] != 33:
            raise ValueError("keypoint_weights must be of length 33.")
    weighted_avg_per_frame = np.sum(distances * keypoint_weights, axis=1) / np.sum(keypoint_weights)
    overall_weighted_avg = np.mean(weighted_avg_per_frame)
    score = max(0, 100 - base_weight * overall_weighted_avg)
    return round(score, 2)

def compute_action_angle_score(original_seq, perfect_seq, action, base_weight=30.0):
    def compute_angle(p1, p2, p3):
        v1 = p1 - p2
        v2 = p3 - p2
        dot = np.dot(v1, v2)
        norm = np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8
        angle = np.arccos(np.clip(dot / norm, -1.0, 1.0))
        return angle

    angle_definitions = {
        "BodyWeightSquats": [
            (23, 25, 27),
            (24, 26, 28),
            (12, 24, 26),
            (11, 23, 25),

        ],
        "JumpingJack": [
            (11, 13, 15),
            (12, 14, 16),
            (12, 24, 26),
            (11, 23, 25),
            (14, 12, 24),
            (13, 11, 23)
        ],
        "PushUps": [
            (11, 13, 15),
            (12, 14, 16),
            (12, 24, 26),
            (11, 23, 25),
            (14, 12, 24),
            (13, 11, 23)
        ]
    }
    angle_defs = angle_definitions.get(action, [])
    if not angle_defs:
        print("No angle definitions for action:", action)
        return None

    errors = []
    for (p1, p2, p3) in angle_defs:
        max_angle_orig = 0
        max_angle_perfect = 0
        T = original_seq.shape[0]
        for i in range(T):
            orig_pose = original_seq[i, :, :3]      # (33,3)
            perfect_pose = perfect_seq[i, :, :3]    # (33,3)
            angle_orig = compute_angle(orig_pose[p1], orig_pose[p2], orig_pose[p3])
            angle_perfect = compute_angle(perfect_pose[p1], perfect_pose[p2], perfect_pose[p3])
            if angle_orig > max_angle_orig:
                max_angle_orig = angle_orig
            if angle_perfect > max_angle_perfect:
                max_angle_perfect = angle_perfect
        errors.append(abs(max_angle_orig - max_angle_perfect))
    overall_error = np.mean(errors)
    score = max(0, 100 - base_weight * overall_error)
    return round(score, 2)

def smooth_sequence(seq, window_size=5):
    T = seq.shape[0]
    smoothed = np.empty_like(seq)
    half = window_size // 2
    for t in range(T):
        start = max(0, t - half)
        end = min(T, t + half + 1)
        smoothed[t] = np.mean(seq[start:end], axis=0)
    return smoothed

def procrustes_alignment(X, Y):
    mu_X = np.mean(X, axis=0)
    mu_Y = np.mean(Y, axis=0)
    Xc = X - mu_X
    Yc = Y - mu_Y
    s = np.trace(Xc.T @ Yc) / (np.linalg.norm(Xc, 'fro')**2 + 1e-8)
    M = Xc.T @ Yc
    U, _, Vt = np.linalg.svd(M)
    R = Vt.T @ U.T
    t = mu_Y - s * mu_X @ R
    return s, R, t

def align_frame_perfect_to_original(perfect_frame, original_frame):
    X = perfect_frame[:, :3]
    Y = original_frame[:, :3]
    s, R, t = procrustes_alignment(X, Y)
    X_aligned = s * X @ R + t
    aligned_frame = perfect_frame.copy()
    aligned_frame[:, :3] = X_aligned
    return aligned_frame

def parse_filename(filename):
    basename = os.path.splitext(filename)[0]
    m = re.match(r"([A-Za-z]+)(\d+)", basename)
    if m:
        action = m.group(1)
        num_actions = int(m.group(2))
        return action, num_actions
    else:
        return None, None

def test():
    # action = "BodyWeightSquats", "JumpingJack", "PushUps"
    video_dir = "test_video"
    json_dir = "test_skeleton_json"
    output_dir = "result_videos"

    for video_filename in os.listdir(video_dir):
        if not (video_filename.lower().endswith(".mp4") or video_filename.lower().endswith(".avi")):
            continue
        action, num_actions = parse_filename(video_filename)
        if action is None or num_actions is None:
            print(f"Cannot parse {video_filename}")
            continue
        standard_jsons = [
            'perfect_skeleton_data/' + action + '/perfect1.json',
            'perfect_skeleton_data/' + action + '/perfect2.json',
            'perfect_skeleton_data/' + action + '/perfect3.json',
            'perfect_skeleton_data/' + action + '/perfect4.json',
            'perfect_skeleton_data/' + action + '/perfect5.json',
        ]

        video_path = os.path.join(video_dir, video_filename)
        json_path = os.path.join(json_dir, os.path.splitext(video_filename)[0] + ".json")

        if not os.path.exists(json_path):
            print(f"JSON file not found for {video_filename}")
            continue

        original_seq_total = load_skeleton_sequence(json_path)
        if original_seq_total is None:
            print(f"Failed to load skeleton for {video_filename}")
            continue

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        frames_per_action = total_frames // num_actions

        perfect_seq = build_perfect_action(standard_jsons, frames_per_action, action)
        if perfect_seq is None:
            print("Failed to build perfect action.")
            return

        output_frames = []
        total_score = 0
        curr_score = 0
        for seg in range(num_actions):
            start_frame = seg * frames_per_action
            end_frame = (seg + 1) * frames_per_action
            original_seq = original_seq_total[start_frame:end_frame]

            orig_norm_seq, transform_params = normalize_pose_and_store_params(original_seq, action)

            # point method
            # weights = None
            # if action == "BodyWeightSquats":
            #     weights = np.zeros(33)
            #     for idx in [27, 28]:
            #         weights[idx] = 1.0
            #     for idx in [25, 26]:
            #         weights[idx] = 3.0
            #     for idx in [11, 12, 23, 24]:
            #         weights[idx] = 5.0
            # elif action == "JumpingJack":
            #     weights = np.zeros(33)
            #     for idx in [13, 14, 15, 16]:
            #         weights[idx] = 2.0
            #     for idx in [11, 12, 23, 24]:
            #         weights[idx] = 1.0
            #     for idx in [25, 26, 27, 28]:
            #         weights[idx] = 5.0
            # elif action == "PushUps":
            #     weights = np.zeros(33)
            #     for idx in [13, 14]:
            #         weights[idx] = 1.0
            #     for idx in [23, 24]:
            #         weights[idx] = 5.0
            #     for idx in [11, 12, 15, 16, 25, 26, 27, 28]:
            #         weights[idx] = 3.0
            # score = compute_action_standard_score(orig_norm_seq, perfect_seq, base_weight=20.0,
            #                                       keypoint_weights=weights)

            score = compute_action_angle_score(original_seq, perfect_seq, action, base_weight=50.0)
            print(f"Segment {seg} Action Standard Score: {score}")

            # Update
            curr_score = score
            total_score += curr_score

            projected_original_seq = project_with_stored_params(orig_norm_seq, transform_params)
            projected_perfect_seq = project_with_stored_params(perfect_seq, transform_params)
            projected_perfect_seq = smooth_sequence(projected_perfect_seq, window_size=5)

            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            seg_frames = []
            for i in range(frames_per_action):
                ret, frame = cap.read()
                if not ret:
                    break

                draw_skeleton_on_frame_custom(frame, projected_original_seq[i],
                                              width, height, scale=1.0, offset_x=0, offset_y=0,
                                              line_color=(255, 255, 255), point_color=(0, 0, 255))

                overlay_skeleton_in_corner(frame, projected_perfect_seq[i],
                                              width, height, scale=1.3, offset_x=0, offset_y=0,
                                              line_color=(0, 255, 0), point_color=(255, 0, 0))


                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1.5
                thickness = 2
                margin = 10
                avg_score = round(total_score / (seg+1), 2)
                texts = [
                    f"Action: {action}",
                    f"Count: {(seg+1)}/{num_actions}",
                    f"Score: {curr_score}",
                    f"Avg Score: {avg_score}"
                ]

                y0 = margin + 50
                line_spacing = 50

                for idx, text in enumerate(texts):
                    x_text = int(width/2 - margin - 200)
                    y_text = y0 + idx * line_spacing
                    cv2.putText(frame, text, (x_text, y_text), font, font_scale, (0, 0, 255), thickness)

                seg_frames.append(frame)

            cap.release()
            output_frames.extend(seg_frames)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_video = os.path.join(output_dir, os.path.splitext(video_filename)[0] + "_result_overlay.mp4")
        out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
        for frame in output_frames:
            out.write(frame)
        out.release()
        print(f"Combined overlay video saved to: {output_video}")


if __name__ == "__main__":
    test()