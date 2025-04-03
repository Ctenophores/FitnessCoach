# import os
# import json
# import numpy as np
# import cv2
# from typing import List
#
#
# #############################
# # 1) Helper: 读取一段动作骨骼序列 (T, 33, 4)
# #############################
# def load_skeleton_sequence(json_path):
#     """
#     从 poseTrackingVideo.py 生成的 .json 中读取所有帧的 pose
#     返回 shape=(T, 33, 4) 的 np.array
#     """
#     with open(json_path, 'r') as f:
#         data = json.load(f)
#     frames = []
#     for frame_data in data:
#         pose = frame_data.get('pose', [])
#         if not pose:
#             pose = [[0, 0, 0, 0]] * 33
#         arr = np.array(pose, dtype=np.float32).reshape(33, 4)
#         frames.append(arr)
#     if len(frames) == 0:
#         return None
#     return np.stack(frames, axis=0)  # (T, 33, 4)
#
#
# #############################
# # 2) Helper: 归一化 (单帧)
# #############################
# def normalize_pose_single_frame(pose_33x4):
#     """
#     以骨盆 index=23 为中心平移, 再用(23->11)或(23->25)距离做缩放 (2D为例)
#     pose_33x4 shape=(33,4)
#     返回相同shape
#     """
#     pose = pose_33x4.copy()
#     center = pose[23, :2]  # 只用 x,y
#     pose[:, :2] -= center
#
#     # 以 (23->11) 为尺度
#     dist = np.linalg.norm(pose[11, :2] - pose[23, :2])
#     if dist < 1e-5:
#         dist = 1e-5
#     pose[:, :2] /= dist
#     return pose
#
#
# #############################
# # 3) Helper: 时间插值 (对T帧插值到 new_len 帧)
# #############################
# def time_resample(sequence, new_len=100):
#     """
#     sequence shape=(T,33,4)
#     在时间维度上做线性插值到 new_len 帧
#     返回 shape=(new_len,33,4)
#     """
#     import math
#     from scipy.interpolate import interp1d
#
#     T = sequence.shape[0]
#     if T == 0:
#         return sequence
#     x_old = np.linspace(0, 1, T)
#     x_new = np.linspace(0, 1, new_len)
#
#     # 对每个 (33,4) 维度分别插值
#     seq_new = []
#     for i in range(33):
#         for j in range(4):
#             f = interp1d(x_old, sequence[:, i, j], kind='linear', fill_value='extrapolate')
#             # shape (new_len,)
#             col_new = f(x_new)
#             if j == 0:
#                 # 第一次时先开一个 (new_len, 4) 的2D array
#                 tmp = np.zeros((new_len, 4), dtype=np.float32)
#                 tmp[:, 0] = col_new
#             else:
#                 tmp[:, j] = col_new
#         # tmp shape=(new_len,4)
#         # 存一下
#         if i == 0:
#             # 第一次, shape=(new_len,33,4)
#             big_tmp = np.zeros((new_len, 33, 4), dtype=np.float32)
#             big_tmp[:, i, :] = tmp
#         else:
#             big_tmp[:, i, :] = tmp
#     return big_tmp
#
#
# #############################
# # 步骤 2: 构建“完美动作” (平均后的多帧骨骼)
# #############################
# def build_perfect_action(json_paths: List[str], target_len=100):
#     """
#     1) 对多个标准动作的 .json
#     2) 各自加载 (T_i, 33,4)
#     3) 对每帧做 normalize_pose_single_frame
#     4) 时间插值到 target_len
#     5) 再对所有插值结果做加权平均
#     返回 shape=(target_len, 33,4)
#     """
#     sequences = []
#     for p in json_paths:
#         seq = load_skeleton_sequence(p)
#         if seq is None or seq.shape[0] < 2:
#             continue
#
#         # 对每帧做 normalize
#         for i in range(seq.shape[0]):
#             seq[i] = normalize_pose_single_frame(seq[i])
#
#         # 时间插值
#         seq_resampled = time_resample(seq, new_len=target_len)
#         sequences.append(seq_resampled)
#
#     if not sequences:
#         return None
#
#     # 叠加
#     all_data = np.stack(sequences, axis=0)  # shape=(N, target_len, 33,4)
#     perfect_seq = np.mean(all_data, axis=0)  # shape=(target_len, 33,4)
#     return perfect_seq
#
#
# #############################
# # 计算用户动作分数
# #############################
# def compute_action_score(user_seq, perfect_seq):
#     """
#     假设二者形状相同 (T,33,4),
#     做简单关节差/角度差并输出 0~100 的分数
#     这里演示最简单的2D位置差
#     """
#     if user_seq.shape != perfect_seq.shape:
#         print("Error: shape mismatch!", user_seq.shape, perfect_seq.shape)
#         return 0.0
#     # 只比较 (x,y)
#     diff = user_seq[:, :, :2] - perfect_seq[:, :, :2]
#     dist_per_frame = np.linalg.norm(diff, axis=-1)  # shape=(T,33)
#     # 平均
#     avg_dist = np.mean(dist_per_frame)
#     # 简单线性映射
#     score = max(0, 100 - avg_dist * 50)
#     return round(score, 2)
#
#
# #############################
# # 步骤 3&4: 对用户动作分段 & 对齐 & 可视化
# #############################
# def evaluate_user_video(user_json, perfect_seq, segment_starts_ends=None):
#     """
#     1) 从 user_json 读取整段骨骼 (M,33,4)
#     2) 根据 segment_starts_ends 分割出多次动作
#     3) 对每次动作做 normalize + 时间插值 => (T,33,4)
#     4) 与 perfect_seq 做比较 => 打分
#     5) 可视化(可选)
#
#     segment_starts_ends: e.g. [(10,60), (80,130), (150,210)], 表示用户做了 3 次动作
#     如果为空, 只视为1段(整个视频)
#     """
#     user_seq = load_skeleton_sequence(user_json)
#     if user_seq is None:
#         print("No user sequence loaded.")
#         return
#
#     # 这里假设 target_len=perfect_seq.shape[0]
#     target_len = perfect_seq.shape[0]
#
#     if not segment_starts_ends:
#         # 全视频当做1段
#         segment_starts_ends = [(0, user_seq.shape[0])]
#
#     for i, (s, e) in enumerate(segment_starts_ends):
#         if e > user_seq.shape[0]:
#             e = user_seq.shape[0]
#         sub_seq = user_seq[s:e]  # shape=(L,33,4)
#         # normalize
#         for f in range(sub_seq.shape[0]):
#             sub_seq[f] = normalize_pose_single_frame(sub_seq[f])
#         # 时间插值
#         sub_seq_resampled = time_resample(sub_seq, new_len=target_len)
#         # compute score
#         score = compute_action_score(sub_seq_resampled, perfect_seq)
#         print(f"Segment {i}: frames[{s}:{e}] => Score: {score}")
#
#     # 如果想做可视化(把两段骨骼叠加画出来), 可以再写一个 draw_sequence_on_video
#
#
# #############################
# # Demo: 主流程
# #############################
# def main():
#     """
#     演示如何:
#     1. 选择多个标准动作的 JSON
#     2. 得到 perfect_action_seq (步骤2)
#     3. 给用户输入的一段视频 => 取其 JSON => 分段 => 与 perfect_action_seq 做对比(步骤3&4)
#     """
#     # 1) 多个标准 JSON
#     standard_jsons = [
#         'perfect_skeleton_data/BodyWeightSquats/perfect1.json',
#         'perfect_skeleton_data/BodyWeightSquats/perfect2.json',
#         'perfect_skeleton_data/BodyWeightSquats/perfect4.json',
#         'perfect_skeleton_data/BodyWeightSquats/perfect5.json',
#     ]
#     # 构建完美动作
#     perfect_seq = build_perfect_action(standard_jsons, target_len=100)
#     if perfect_seq is None:
#         print("No perfect action built. Check your standard JSON.")
#         return
#
#     # 2) 用户 JSON
#     user_json = 'perfect_skeleton_data/BodyWeightSquats/perfect3.json'
#
#     # 打开视频文件
#     cap = cv2.VideoCapture('PerfectAction/BodyWeightSquats/perfect3.mp4')
#
#     # 通过 CAP_PROP_FRAME_COUNT 属性获取总帧数（最大帧数）
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#
#     # 假设我们手动知道用户做了3次深蹲, 分段大约是:
#     segment_starts_ends = [(0, total_frames)]
#     # 其实要看真实帧数来定; 可做更智能的自动分段
#
#     # 3) 评估打分
#     evaluate_user_video(user_json, perfect_seq, segment_starts_ends=segment_starts_ends)
#
#
# if __name__ == "__main__":
#     main()

import os
import json
import numpy as np
import cv2
from typing import List
from scipy.interpolate import interp1d
import mediapipe as mp

POSE_CONNECTIONS = mp.solutions.holistic.POSE_CONNECTIONS  # 用于画线

#############################
# A) 读取并构建完美动作骨骼序列
#############################

def load_skeleton_sequence(json_path):
    """
    从 poseTrackingVideo.py 生成的 .json 中读取所有帧的 pose
    返回 shape=(T, 33, 4) 的 np.array
    """
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

# def normalize_pose_single_frame(pose_33x4):
#     """
#     以骨盆 index=23 为中心平移, 再用(23->11)距离做缩放 (2D为例)，
#     只操作 (x,y)，不处理 z/visibility。
#     pose_33x4 shape=(33,4)
#     返回相同shape
#     """
#     pose = pose_33x4.copy()
#     center = pose[23, :2]  # 只用 x,y
#     pose[:, :2] -= center
#
#     dist = np.linalg.norm(pose[11, :2] - pose[23, :2])
#     if dist < 1e-5:
#         dist = 1e-5
#     pose[:, :2] /= dist
#     return pose

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

        z_axis = np.cross(x_axis, y_axis)  # 身体朝向
        z_axis /= np.linalg.norm(z_axis) + 1e-8

        y_axis = np.cross(z_axis, x_axis)

        R = np.stack([x_axis, y_axis, z_axis], axis=1)  # shape (3, 3)

        xyz = pose[:, :3] - origin
        xyz_rot = xyz @ R

        scale = np.linalg.norm(xyz_rot[11] - xyz_rot[23])
        scale = max(scale, 1e-5)
        xyz_rot /= scale

        # 保存旋转后结果
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

def compute_action_standard_score(original_seq, projected_seq, weight=10.0):
    diff = original_seq[:, :, :3] - projected_seq[:, :, :3]  # (T,33,3)
    distances = np.linalg.norm(diff, axis=-1)  # (T,33)
    avg_distance = np.mean(distances)
    score = max(0, 100 - weight * avg_distance)
    return round(score, 2)

def main():
    standard_jsons = [
        'perfect_skeleton_data/BodyWeightSquats/perfect1.json',
        'perfect_skeleton_data/BodyWeightSquats/perfect2.json',
        'perfect_skeleton_data/BodyWeightSquats/perfect3.json',
        #'perfect_skeleton_data/BodyWeightSquats/perfect4.json',
        'perfect_skeleton_data/BodyWeightSquats/perfect5.json',
    ]
    original_video = "PerfectAction/BodyWeightSquats/perfect5.mp4"
    original_seq = load_skeleton_sequence('perfect_skeleton_data/BodyWeightSquats/perfect5.json')

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

    # 2. 将标准化后的骨架映射回原始空间
    projected_seq = project_perfect_to_original(perfect_seq, original_seq)

    score = compute_action_standard_score(original_seq, projected_seq, weight=10.0)
    print(f"Action Standard Score: {score}")

    # 3. 可视化叠加
    output_video = "overlaid_skeleton5.mp4"
    overlay_perfect_skeleton_on_video(projected_seq,
                                      video_path=original_video,
                                      output_path=output_video,
                                      scale=0.6,
                                      offset_x=150,
                                      offset_y=80)  # scale 可设置为 1，因为是原始单位

if __name__ == "__main__":
    main()
