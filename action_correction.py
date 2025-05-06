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


def compute_action_angle_score(orig_seq, perf_seq, action, base_weight):
    def compute_angle(p1, p2, p3):
        v1 = p1 - p2
        v2 = p3 - p2
        dot = np.dot(v1, v2)
        norm = np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8
        angle = np.arccos(np.clip(dot / norm, -1.0, 1.0))
        angle_deg = angle * 180.0 / np.pi  # 0° - 180°

        return angle_deg

    angle_definitions = {
        "BodyWeightSquats": [
            (23, 25, 27),
            (24, 26, 28)
        ],
        "JumpingJack": [
            (23, 24, 26),
            (24, 23, 25),
            (14, 12, 24),
            (13, 11, 23)
        ],
        "PushUps": [
            (11, 13, 15),
            (12, 14, 16),
            (12, 24, 26),
            (11, 23, 25)
        ]
    }

    angle_defs = angle_definitions.get(action, [])
    if not angle_defs:
        print("No angle definitions for action:", action)
        return None

    T = orig_seq.shape[0]
    M = len(angle_defs)
    errors = np.zeros((T, M), dtype=np.float32)

    for idx, (p1, p2, p3) in enumerate(angle_defs):
        v1_o = orig_seq[:, p1] - orig_seq[:, p2]
        v2_o = orig_seq[:, p3] - orig_seq[:, p2]
        v1_p = perf_seq[:, p1] - perf_seq[:, p2]
        v2_p = perf_seq[:, p3] - perf_seq[:, p2]

        cos_o = np.einsum('ij,ij->i', v1_o, v2_o) / (
                np.linalg.norm(v1_o, axis=1) * np.linalg.norm(v2_o, axis=1) + 1e-8)
        cos_p = np.einsum('ij,ij->i', v1_p, v2_p) / (
                np.linalg.norm(v1_p, axis=1) * np.linalg.norm(v2_p, axis=1) + 1e-8)

        ang_o = np.arccos(np.clip(cos_o, -1, 1)) * 180 / np.pi
        ang_p = np.arccos(np.clip(cos_p, -1, 1)) * 180 / np.pi

        errors[:, idx] = np.abs(ang_o - ang_p)

    overall_error = errors.mean()
    score = max(0, 100 - base_weight * (overall_error / 180.0))
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

def beautify_frame(frame, action, count, total, score, avg_score, perfect_skeleton):
    H, W = frame.shape[:2]
    PANEL_W = W // 2               # 右侧面板宽度
    canvas = np.ones((H, W + PANEL_W, 3), dtype=np.uint8) * 255  # 纯白背景
    # 1) 把原帧拷贝到左侧
    canvas[:, :W] = frame

    # 一些公用位置参数
    pad = 20
    FONT = cv2.FONT_HERSHEY_SIMPLEX

    # 2) 顶部动作按钮
    actions = ["BodyWeightSquats", "JumpingJack", "PushUps"]
    btn_w = (PANEL_W - pad * (len(actions) + 1)) // len(actions)
    btn_h = 50
    for i, name in enumerate(actions):
        x0 = W + pad + i * (btn_w + pad)
        y0 = pad
        x1, y1 = x0 + btn_w, y0 + btn_h
        for i, name in enumerate(actions):
            x0 = W + pad + i * (btn_w + pad)
            y0 = pad
            x1, y1 = x0 + btn_w, y0 + btn_h

            if name == action:
                # 1) 填充背景色
                cv2.rectangle(canvas, (x0, y0), (x1, y1), (0, 200, 0), thickness=-1, lineType=cv2.LINE_AA)
                # 2) 再画个深一点的边框
                cv2.rectangle(canvas, (x0, y0), (x1, y1), (0, 150, 0), thickness=3, lineType=cv2.LINE_AA)
                text_color = (255, 255, 255)
            else:
                cv2.rectangle(canvas, (x0, y0), (x1, y1), (0, 0, 0), thickness=2, lineType=cv2.LINE_AA)
                text_color = (0, 0, 0)

            # 文本也改成选中白字，未选黑字
            if name == "BodyWeightSquats":
                name = "Squats"
            cv2.putText(canvas, name, (x0 + 10, y0 + btn_h - 15),
                        FONT, 0.7, text_color, 2, cv2.LINE_AA)

    # 3) 中间圆形计数器
    #    画一个圆环，填充当前 fraction
    radius = min(int(PANEL_W/4), int(H/7))
    thickness = int(radius/5)
    center_x = W + PANEL_W // 2
    center_y = y0 + btn_h + radius + 50
    # 底部圆环
    cv2.circle(canvas, (center_x, center_y), radius, (200,200,200), thickness, cv2.LINE_AA)
    # 填充扇形
    fraction = count / total
    end_angle = int(360 * fraction)
    cv2.ellipse(canvas,
                (center_x, center_y),
                (radius, radius),
                0,           # 旋转角度
                -90,         # 起始角度：从顶部开始
                -90 + end_angle,
                (0,200,0),
                thickness, cv2.LINE_AA)
    # 计数文字
    count_text = f"{count}/{total}"
    # 用 fontScale=1.0, thickness=2 测量文字高度
    (text_w, text_h), baseline = cv2.getTextSize(count_text, FONT, fontScale=1.0, thickness=2)
    # 希望文字高度占 radius 的比例（例如 60%）
    target_h = radius * 0.4
    fontScale_count = target_h / text_h
    thickness_count = max(int(fontScale_count * 2), 1)
    # 缩放后的尺寸
    text_w_scaled = int(text_w * fontScale_count)
    text_h_scaled = int(text_h * fontScale_count)
    # 居中坐标
    org_count = (center_x - text_w_scaled // 2,
                 center_y + text_h_scaled // 2)
    cv2.putText(canvas,
                count_text,
                org_count,
                FONT,
                fontScale_count,
                (0, 200, 0),
                thickness_count,
                cv2.LINE_AA)

    # 4) 分数与提示
    txt_y = center_y + radius + pad*2
    cv2.putText(canvas, f"Score: {score:.1f}",
                (W+pad, txt_y), FONT, 0.8, (0,0,0), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"Avg:   {avg_score:.1f}",
                (W+pad, txt_y+30), FONT, 0.8, (0,0,0), 2, cv2.LINE_AA)

    # 如果接近满分，可以写“PERFECT!”
    if score > 90:
        cv2.putText(canvas, "PERFECT!",
                    (W+pad, txt_y+80), FONT, 1.2, (0,180,0), 3, cv2.LINE_AA)
    elif score > 85:
        cv2.putText(canvas, "GOOD",
                    (W+pad, txt_y+80), FONT, 1.2, (0,255,255), 3, cv2.LINE_AA)
    elif score > 75:
        cv2.putText(canvas, "NOT BAD",
                    (W+pad, txt_y+80), FONT, 1.2, (0,165,255), 3, cv2.LINE_AA)
    else:
        cv2.putText(canvas, "WRONG!",
                    (W+pad, txt_y+80), FONT, 1.2, (0,0,255), 3, cv2.LINE_AA)

    # 5) 右下小窗口：完美骨架示意
    win_w, win_h = PANEL_W - pad*2, H//2 - pad*2
    x0, y0 = W + pad, H - pad - win_h
    # 背景与边框
    # cv2.rectangle(canvas, (x0,y0), (x0+win_w, y0+win_h),
    #               (0,0,0), thickness=2, lineType=cv2.LINE_AA)

    roi = canvas[y0:y0+win_h, x0:x0+win_w]
    draw_skeleton_on_frame_custom(
        roi,
        perfect_skeleton,
        width=win_w, height=win_h*0.8,
        scale=1.0, offset_x=0, offset_y=0,
        line_color=(0,200,0), point_color=(255,0,0),
        inverse=False, point_radius=3, line_thickness=2
    )

    return canvas

def test():
    # action = "BodyWeightSquats", "JumpingJack", "PushUps"
    video_dir = "test_video"
    json_dir = "test_skeleton_json"
    output_dir = "result_videos"

    # video_to_skeleton(video_dir=video_dir, output_dir=json_dir, output_video_dir='test_skeleton')

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
        first_transform_params = None
        for seg in range(num_actions):
            start_frame = seg * frames_per_action
            end_frame = (seg + 1) * frames_per_action
            original_seq = original_seq_total[start_frame:end_frame]

            orig_norm_seq, transform_params = normalize_pose_and_store_params(original_seq, action)

            if seg == 0:
                first_transform_params = transform_params

            score = compute_action_angle_score(original_seq, perfect_seq, action, base_weight=50.0)

            total_score += score

            projected_original_seq = project_with_stored_params(orig_norm_seq, transform_params)
            projected_original_seq = smooth_sequence(projected_original_seq, window_size=5)

            projected_perfect_seq = project_with_stored_params(perfect_seq, first_transform_params)
            projected_perfect_seq = smooth_sequence(projected_perfect_seq, window_size=5)

            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            seg_frames = []
            for i in range(frames_per_action):
                ret, frame = cap.read()
                if not ret:
                    break

                draw_skeleton_on_frame_custom(
                    frame,
                    projected_original_seq[i],
                    width, height,
                    scale=1.0, offset_x=0, offset_y=0,
                    line_color=(255, 255, 255), point_color=(0, 0, 255)
                )

                beautified = beautify_frame(
                    frame,
                    action=action,
                    count=seg + 1,
                    total=num_actions,
                    score=score,
                    avg_score=round(total_score / (seg + 1), 2),
                    perfect_skeleton=projected_perfect_seq[i]
                )
                seg_frames.append(beautified)

            cap.release()
            output_frames.extend(seg_frames)

        panel_w = width // 2

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_video = os.path.join(output_dir,
                                    os.path.splitext(video_filename)[0] + "_result_overlay.mp4")
        out = cv2.VideoWriter(output_video, fourcc, fps, (width + panel_w, height))
        if not out.isOpened():
            print("ERROR: Cannot open VideoWriter with size", (width + panel_w, height))
            return

        for frame in output_frames:
            out.write(frame)
        out.release()

        print(f"Combined overlay video saved to: {output_video}")


if __name__ == "__main__":
    test()