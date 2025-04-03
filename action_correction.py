import os
import json
import numpy as np
import cv2
from typing import List
from scipy.interpolate import interp1d
import mediapipe as mp

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

def normalize_pose_with_rotation(sequence):
    sequence = np.array(sequence, dtype=np.float32)
    num_frames = sequence.shape[0]
    rotated_sequence = np.empty_like(sequence)

    for i in range(num_frames):
        pose = sequence[i]  # (33, 4)

        left_hip = pose[23, :3]
        right_hip = pose[24, :3]
        shoulder_mid = (pose[11, :3] + pose[12, :3]) / 2.0

        origin = (left_hip + right_hip) / 2.0

        x_axis = right_hip - left_hip
        x_axis /= np.linalg.norm(x_axis) + 1e-8

        up_vector = shoulder_mid - origin
        y_axis = up_vector - np.dot(up_vector, x_axis) * x_axis
        y_axis /= np.linalg.norm(y_axis) + 1e-8

        z_axis = np.cross(x_axis, y_axis)
        z_axis /= np.linalg.norm(z_axis) + 1e-8

        y_axis = np.cross(z_axis, x_axis)

        R = np.stack([x_axis, y_axis, z_axis], axis=1)  # shape (3, 3)

        xyz = pose[:, :3] - origin
        xyz_rot = xyz @ R

        scale = np.linalg.norm(xyz_rot[11] - xyz_rot[23])
        scale = max(scale, 1e-5)
        xyz_rot /= scale

        rotated_sequence[i, :, :3] = xyz_rot
        rotated_sequence[i, :, 3] = pose[:, 3]

    return rotated_sequence

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

def build_perfect_action(json_paths: List[str], target_len=100):
    sequences = []
    for p in json_paths:
        seq = load_skeleton_sequence(p)
        if seq is None or seq.shape[0] < 2:
            continue

        # normalize
        seq = normalize_pose_with_rotation(seq)

        seq_resampled = time_resample(seq, new_len=target_len)
        sequences.append(seq_resampled)

    if not sequences:
        return None
    all_data = np.stack(sequences, axis=0)  # shape=(N, target_len, 33,4)
    perfect_seq = np.mean(all_data, axis=0)  # (target_len,33,4)
    return perfect_seq


def overlay_perfect_skeleton_on_video(perfect_seq, video_path, output_path, scale, offset_x, offset_y):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open video {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    T = perfect_seq.shape[0]
    valid_frames = min(T, total_frames)
    print(f"Video frames={total_frames}, perfect_seq frames={T}, use {valid_frames} frames.")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx >= valid_frames:
            out.write(frame)
            frame_idx += 1
            continue

        skeleton = perfect_seq[frame_idx]  # (33,4)

        draw_skeleton_on_frame(frame, skeleton, width, height, scale, offset_x, offset_y)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"Overlay video saved to: {output_path}")

def draw_skeleton_on_frame(frame, skeleton_33x4, width, height,
                           scale, offset_x, offset_y, point_radius=4, line_thickness=2):
    xy = skeleton_33x4[:, :2]

    min_xy = np.min(xy, axis=0)   # (min_x, min_y)
    max_xy = np.max(xy, axis=0)   # (max_x, max_y)
    center_xy = (min_xy + max_xy) / 2.0
    range_xy = np.maximum(max_xy - min_xy, 1e-5)

    points_2d = []
    for i in range(33):
        x, y = skeleton_33x4[i, 0], skeleton_33x4[i, 1]

        x_px = int((x - center_xy[0]) * scale * width + width // 2 + offset_x)
        y_px = int((y - center_xy[1]) * scale * height + height // 2 + offset_y)

        points_2d.append((x_px, y_px))

    for conn in POSE_CONNECTIONS:
        start_idx, end_idx = conn
        x1, y1 = points_2d[start_idx]
        x2, y2 = points_2d[end_idx]
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), thickness=line_thickness)

    for (px, py) in points_2d:
        cv2.circle(frame, (px, py), point_radius, (0, 0, 255), -1)

def project_perfect_to_original(perfect_seq, original_seq):
    assert perfect_seq.shape[0] == original_seq.shape[0], "The number of frames are not equal!"

    projected_seq = np.empty_like(perfect_seq)

    for i in range(perfect_seq.shape[0]):
        p_std = perfect_seq[i]
        p_ori = original_seq[i]

        left_hip = p_ori[23, :3]
        right_hip = p_ori[24, :3]
        shoulder_mid = (p_ori[11, :3] + p_ori[12, :3]) / 2.0
        origin = (left_hip + right_hip) / 2.0

        x_axis = right_hip - left_hip
        x_axis /= np.linalg.norm(x_axis) + 1e-8

        up_vector = shoulder_mid - origin
        y_axis = up_vector - np.dot(up_vector, x_axis) * x_axis
        y_axis /= np.linalg.norm(y_axis) + 1e-8

        z_axis = np.cross(x_axis, y_axis)
        z_axis /= np.linalg.norm(z_axis) + 1e-8

        y_axis = np.cross(z_axis, x_axis)

        R = np.stack([x_axis, y_axis, z_axis], axis=1)  # (3,3)
        R_inv = R.T

        scale = np.linalg.norm(p_ori[11, :3] - p_ori[23, :3])
        scale = max(scale, 1e-5)

        xyz = p_std[:, :3] * scale
        xyz = xyz @ R_inv
        xyz = xyz + origin

        projected_seq[i, :, :3] = xyz
        projected_seq[i, :, 3] = p_ori[:, 3]

    return projected_seq


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

def test_BodyWeightSquats():
    action = "BodyWeightSquats"

    standard_jsons = [
        'perfect_skeleton_data/BodyWeightSquats/perfect1.json',
        'perfect_skeleton_data/BodyWeightSquats/perfect3.json',
        'perfect_skeleton_data/BodyWeightSquats/perfect4.json'
    ]
    original_video = "PerfectAction/BodyWeightSquats/perfect2.mp4"
    original_seq = load_skeleton_sequence('perfect_skeleton_data/BodyWeightSquats/perfect2.json')

    cap = cv2.VideoCapture(original_video)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    perfect_seq = build_perfect_action(standard_jsons, target_len=total_frames)
    if perfect_seq is None:
        print("Failed to build perfect action.")
        return

    if original_seq is None:
        print("Failed to load original skeleton sequence.")
        return

    projected_seq = project_perfect_to_original(perfect_seq, original_seq)

    weights = None
    if action == "BodyWeightSquats":
        weights = np.zeros(33)
        for idx in [11,12,23,24,25,26,27,28]:
            weights[idx] = 1.0
    elif action == "JumpingJack":
        weights = np.zeros(33)
        for idx in [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]:
            weights[idx] = 1.0
    elif action == "PushUps":
        weights = np.zeros(33)
        for idx in [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]:
            weights[idx] = 1.0

    score = compute_action_standard_score(original_seq, projected_seq, base_weight=10.0, keypoint_weights=weights)
    print("Action Standard Score:", score)

    output_video = "BodyWeightSquats_result.mp4"
    overlay_perfect_skeleton_on_video(projected_seq,
                                      video_path=original_video,
                                      output_path=output_video,
                                      scale=0.6,
                                      offset_x=150,
                                      offset_y=80)

def test_JumpingJack():
    action = "JumpingJack"

    standard_jsons = [
        'perfect_skeleton_data/JumpingJacks/perfect1.json',
        'perfect_skeleton_data/JumpingJacks/perfect3.json',
        'perfect_skeleton_data/JumpingJacks/perfect4.json'
    ]
    original_video = "PerfectAction/JumpingJack/perfect2.mp4"
    original_seq = load_skeleton_sequence('perfect_skeleton_data/JumpingJacks/perfect2.json')

    cap = cv2.VideoCapture(original_video)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    perfect_seq = build_perfect_action(standard_jsons, target_len=total_frames)
    if perfect_seq is None:
        print("Failed to build perfect action.")
        return

    if original_seq is None:
        print("Failed to load original skeleton sequence.")
        return

    projected_seq = project_perfect_to_original(perfect_seq, original_seq)

    weights = None
    if action == "BodyWeightSquats":
        weights = np.zeros(33)
        for idx in [11,12,23,24,25,26,27,28]:
            weights[idx] = 1.0
    elif action == "JumpingJack":
        weights = np.zeros(33)
        for idx in [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]:
            weights[idx] = 1.0
    elif action == "PushUps":
        weights = np.zeros(33)
        for idx in [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]:
            weights[idx] = 1.0

    score = compute_action_standard_score(original_seq, projected_seq, base_weight=10.0, keypoint_weights=weights)
    print("Action Standard Score:", score)

    output_video = "JumpingJack_result5.mp4"
    overlay_perfect_skeleton_on_video(projected_seq,
                                      video_path=original_video,
                                      output_path=output_video,
                                      scale=0.6,
                                      offset_x=150,
                                      offset_y=80)

if __name__ == "__main__":
    test_BodyWeightSquats()
    test_JumpingJack()
