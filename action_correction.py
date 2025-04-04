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

def normalize_pose_and_store_params(sequence):
    sequence = np.array(sequence, dtype=np.float32)
    num_frames = sequence.shape[0]
    norm_seq = np.empty_like(sequence)
    transform_params = []
    for i in range(num_frames):
        pose = sequence[i]  # (33, 4)
        left_hip = pose[23, :3]
        right_hip = pose[24, :3]
        shoulder_mid = (pose[11, :3] + pose[12, :3]) / 2.0
        origin = (left_hip + right_hip) / 2.0

        x_axis = right_hip - left_hip
        x_axis /= (np.linalg.norm(x_axis) + 1e-8)

        up_vector = shoulder_mid - origin
        y_axis = up_vector - np.dot(up_vector, x_axis) * x_axis
        y_axis /= (np.linalg.norm(y_axis) + 1e-8)

        z_axis = np.cross(x_axis, y_axis)
        z_axis /= (np.linalg.norm(z_axis) + 1e-8)
        y_axis = np.cross(z_axis, x_axis)
        y_axis /= (np.linalg.norm(y_axis) + 1e-8)

        R = np.stack([x_axis, y_axis, z_axis], axis=1)  # (3,3)
        xyz = pose[:, :3] - origin
        xyz_rot = xyz @ R
        # 采用更稳定的尺度：肩膀中点与左右髋中心的距离
        scale = np.linalg.norm(shoulder_mid - ((left_hip+right_hip)/2.0))
        scale = max(scale, 1e-5)
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

def build_perfect_action(json_paths: List[str], target_len=100):
    sequences = []
    for p in json_paths:
        seq = load_skeleton_sequence(p)
        if seq is None or seq.shape[0] < 2:
            continue
        norm_seq, _ = normalize_pose_and_store_params(seq)
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

def overlay_two_skeletons_on_video(original_seq, perfect_seq, video_path, output_path, scale_orig, offset_x_orig, offset_y_orig, scale_perfect, offset_x_perfect, offset_y_perfect, inverse=False):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open video {video_path}")
        return
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    T = min(original_seq.shape[0], perfect_seq.shape[0], total_frames)
    print(f"Video frames={total_frames}, using {T} frames.")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret or frame_idx >= T:
            break
        # 绘制原始骨架，使用自定义颜色
        draw_skeleton_on_frame_custom(frame, original_seq[frame_idx],
                                      width, height, scale_orig, offset_x_orig, offset_y_orig,
                                      line_color=(255, 255, 255), point_color=(0, 0, 255))
        # 绘制完美骨架
        draw_skeleton_on_frame_custom(frame, perfect_seq[frame_idx],
                                      width, height, scale_perfect, offset_x_perfect, offset_y_perfect,
                                      line_color=(0,255,0), point_color=(255,0,0), inverse=inverse)
        out.write(frame)
        frame_idx += 1
    cap.release()
    out.release()
    print(f"Overlay video saved to: {output_path}")

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

def test():
    action = "PushUps" # BodyWeightSquats, JumpingJack, PushUps
    standard_jsons = [
        'perfect_skeleton_data/' + action + '/perfect1.json',
        'perfect_skeleton_data/' + action + '/perfect2.json',
        'perfect_skeleton_data/' + action + '/perfect3.json',
        'perfect_skeleton_data/' + action + '/perfect4.json',
        'perfect_skeleton_data/' + action + '/perfect5.json',
    ]
    original_video = "test_video/1b91c4e6dbb036579fb38ac1303f028e.mp4"
    original_seq_total = load_skeleton_sequence('test_skeleton_json/1b91c4e6dbb036579fb38ac1303f028e.json')

    cap = cv2.VideoCapture(original_video)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    num_of_action = 3 # need to modified
    num_of_frames_per_action = int(total_frames / num_of_action)

    original_seq = original_seq_total[0:num_of_frames_per_action]

    perfect_seq = build_perfect_action(standard_jsons, target_len=num_of_frames_per_action)
    if perfect_seq is None:
        print("Failed to build perfect action.")
        return
    if original_seq is None:
        print("Failed to load original skeleton sequence.")
        return

    orig_norm_seq, transform_params = normalize_pose_and_store_params(original_seq)

    weights = None
    if action == "BodyWeightSquats":
        weights = np.zeros(33)
        for idx in [11, 12, 23, 24, 25, 26, 27, 28]:
            weights[idx] = 1.0
    elif action == "JumpingJack":
        weights = np.zeros(33)
        for idx in [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]:
            weights[idx] = 1.0
    elif action == "PushUps":
        weights = np.zeros(33)
        for idx in [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]:
            weights[idx] = 1.0

    score = compute_action_standard_score(orig_norm_seq, perfect_seq, base_weight=30.0, keypoint_weights=weights)
    print("Action Standard Score:", score)

    projected_original_seq = project_with_stored_params(orig_norm_seq, transform_params)
    projected_perfect_seq = project_with_stored_params(perfect_seq, transform_params)

    projected_perfect_seq = smooth_sequence(projected_perfect_seq, window_size=5)

    output_video = "result_videos/"+ action + "_result_overlay.mp4"
    overlay_two_skeletons_on_video(projected_original_seq, projected_perfect_seq,
                                   video_path=original_video,
                                   output_path=output_video,
                                   scale_orig=1.0, offset_x_orig=0, offset_y_orig=0,
                                   scale_perfect=1.0, offset_x_perfect=0, offset_y_perfect=0, inverse=False)

def parse_filename(filename):
    basename = os.path.splitext(filename)[0]
    # 使用正则表达式：匹配一段字母后跟一段数字
    m = re.match(r"([A-Za-z]+)(\d+)", basename)
    if m:
        action = m.group(1)
        num_actions = int(m.group(2))
        return action, num_actions
    else:
        return None, None

def test_all_videos():
    # action = "PushUps"  # 你可以根据实际动作修改此变量: BodyWeightSquats, JumpingJack, PushUps
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

        # 获取视频总帧数及视频属性
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        # 假设视频中动作匀速分布，分为 num_actions 段（每段帧数相同）
        frames_per_action = total_frames // num_actions

        # 构建完美骨架，目标帧数为当前段的帧数
        perfect_seq = build_perfect_action(standard_jsons, target_len=frames_per_action)
        if perfect_seq is None:
            print("Failed to build perfect action.")
            return

        # 存储所有处理后的帧
        output_frames = []

        for seg in range(num_actions):
            start_frame = seg * frames_per_action
            end_frame = (seg + 1) * frames_per_action
            original_seq = original_seq_total[start_frame:end_frame]

            # 对原始骨架标准化，并保存变换参数
            orig_norm_seq, transform_params = normalize_pose_and_store_params(original_seq)

            # 计算动作标准分数（在标准化空间下）
            weights = None
            if action == "BodyWeightSquats":
                weights = np.zeros(33)
                for idx in [11, 12, 23, 24, 25, 26, 27, 28]:
                    weights[idx] = 1.0
            elif action == "JumpingJack":
                weights = np.zeros(33)
                for idx in [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]:
                    weights[idx] = 1.0
            elif action == "PushUps":
                weights = np.zeros(33)
                for idx in [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]:
                    weights[idx] = 1.0
            score = compute_action_standard_score(orig_norm_seq, perfect_seq, base_weight=20.0,
                                                  keypoint_weights=weights)
            print(f"Segment {seg} Action Standard Score: {score}")

            # 反归一化：将标准化的原始骨架和完美骨架恢复到原始空间
            projected_original_seq = project_with_stored_params(orig_norm_seq, transform_params)
            projected_perfect_seq = project_with_stored_params(perfect_seq, transform_params)
            projected_perfect_seq = smooth_sequence(projected_perfect_seq, window_size=5)

            # 读取当前动作段对应的视频帧，并叠加骨架
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            seg_frames = []
            for i in range(frames_per_action):
                ret, frame = cap.read()
                if not ret:
                    break
                # 叠加原始骨架（使用默认绘制函数，此处假设原始骨架数据为 projected_original_seq）
                draw_skeleton_on_frame_custom(frame, projected_original_seq[i],
                                              width, height, scale=1.0, offset_x=0, offset_y=0,
                                              line_color=(0, 255, 0), point_color=(255, 0, 0))
                # 叠加完美骨架
                draw_skeleton_on_frame_custom(frame, projected_perfect_seq[i],
                                              width, height, scale=1.0, offset_x=0, offset_y=0,
                                              line_color=(255, 255, 255), point_color=(0, 0, 255))
                # 可选：在帧上显示分数（例如右上角）
                text = f"Score: {score}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1.0
                thickness = 2
                (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
                margin = 10
                x_text = width - text_w - margin
                y_text = margin + text_h
                cv2.putText(frame, text, (x_text, y_text), font, font_scale, (0, 0, 255), thickness)
                seg_frames.append(frame)

            cap.release()
            output_frames.extend(seg_frames)

        # 将所有处理后的帧写入一个输出视频文件
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_video = os.path.join(output_dir, os.path.splitext(video_filename)[0] + "_result_overlay.mp4")
        out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
        for frame in output_frames:
            out.write(frame)
        out.release()
        print(f"Combined overlay video saved to: {output_video}")


if __name__ == "__main__":
    # test()
    test_all_videos()