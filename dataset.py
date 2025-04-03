import json
import os
import random
import shutil
import numpy as np
import trimesh
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence


class FitnessDataset(Dataset):
    def __init__(self, root_dir, split = 'train', transform = None, augment = False):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.augment = augment

        self.files = []
        self.labels = []
        self.label_map = self._build_file_list_and_label_map()
        print(f"[{self.split}] Loaded {len(self.files)} samples.")

    def _build_file_list_and_label_map(self):
        df = pd.read_excel(os.path.join(self.root_dir, "labels.xlsx"))
        label_names = sorted(df["action_label"].unique())
        label_map = {label: idx for idx, label in enumerate(label_names)}

        label_dict = {}
        for _, row in df.iterrows():
            key = str(row["filename"]).strip()  # e.g., "v_PushUps_g22_c04"
            label_dict[key] = (label_map[row["action_label"]], row["rep_count"])
    
        # List the JSON files in the correct split folder (train or test)
        split_folder = os.path.join(self.root_dir, self.split)
        file_list = [
            os.path.join(root, f)
            for root, _, files in os.walk(split_folder)
            for f in files if f.endswith(".json")
        ]
    
        for f in file_list:
            # Remove extension to match the key in the label dictionary
            # key = os.path.splitext(f)[0]
            key = os.path.splitext(os.path.basename(f))[0]
            if key in label_dict:
                # self.files.append(os.path.join(split_folder, f))
                self.labels.append(label_dict[key])
                self.files.append(f)
            else:
                print("Warning: No label found for file", f)

        return label_map


    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        file_path = self.files[idx]
        label = self.labels[idx]

        with open(file_path, 'r') as f:
            data = json.load(f)

        sequence = []
        for frame in data:
            pose = frame['pose'] if frame['pose'] else [[0, 0, 0, 0]] * 33
            flat_pose = [v for lm in pose for v in lm]  # 33 x 4 = 132
            sequence.append(flat_pose)

        sequence = np.array(sequence, dtype=np.float32)
        sequence = normalize_pose(sequence)
            

        if self.augment:
            sequence = RandomTransform()(sequence)

        if self.transform:
            sequence = self.transform(sequence)

        return torch.tensor(sequence, dtype=torch.float32), label


#####################
# Collate Function for PackedSequence
#####################
def collate_fn(batch):
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(seq) for seq in sequences], dtype=torch.long)

    # Pad sequences
    padded = pad_sequence(sequences, batch_first=True)  # (B, T, F)

    # Sort by length (descending)
    lengths, perm_idx = lengths.sort(0, descending=True)
    padded = padded[perm_idx]
    action_labels = [lbl[0] for lbl in labels]
    rep_counts = [lbl[1] for lbl in labels]

    action_labels = torch.tensor(action_labels)[perm_idx]
    rep_counts = torch.tensor(rep_counts, dtype=torch.float32)[perm_idx]

    # Pack
    packed_input = pack_padded_sequence(padded, lengths.cpu(), batch_first=True, enforce_sorted=True)

    return packed_input, (action_labels, rep_counts)

#####################
# Augmentation Functions
#####################

def add_noise(sequence, noise_level=0.01):
    noise = np.random.normal(0, noise_level, sequence.shape)
    return sequence + noise

def time_warp(sequence, rate=1.1):
    from scipy.interpolate import interp1d
    x = np.arange(sequence.shape[0])
    f = interp1d(x, sequence, axis=0, kind='linear', fill_value='extrapolate')
    new_len = int(sequence.shape[0] * rate)
    new_x = np.linspace(0, sequence.shape[0] - 1, new_len)
    return f(new_x)

def horizontal_flip(sequence):
    flipped = np.copy(sequence)
    flipped[:, 0::4] = 1.0 - flipped[:, 0::4]
    return flipped

def rotate_around_y(sequence, angle_deg=None, z_scale=320.0):
    sequence = np.array(sequence, dtype=np.float32)
    sequence = sequence.reshape(sequence.shape[0], 33, 4)

    if angle_deg is None:
        angle_deg = random.uniform(-30, 30)  # Random rotation angle in degrees
    angle_rad = np.radians(angle_deg)

    cos_theta = np.cos(angle_rad)
    sin_theta = np.sin(angle_rad)
    Ry = np.array([[cos_theta, 0, sin_theta],
                   [0,         1, 0        ],
                   [-sin_theta, 0, cos_theta]])

    # Use point 23 (mid-hip) as rotation center
    center = sequence[:, 23, :3].copy()

    # Scale z to match x/y range (assuming 320px image width)
    sequence[:, :, 2] *= z_scale
    center[:, 2] *= z_scale

    xyz = sequence[:, :, :3]
    centered_xyz = xyz - center[:, np.newaxis, :]
    rotated_xyz = np.einsum('ij,tkj->tki', Ry, centered_xyz)
    sequence[:, :, :3] = rotated_xyz + center[:, np.newaxis, :]

    # Rescale z back
    sequence[:, :, 2] /= z_scale

    return sequence.reshape(sequence.shape[0], -1)

class RandomTransform:
    def __call__(self, sequence):
        if random.random() < 0.5:
            sequence = add_noise(sequence)
        if random.random() < 0.5:
            sequence = horizontal_flip(sequence)
        if random.random() < 0.3:
            sequence = time_warp(sequence, rate=random.uniform(0.8, 1.2))
        if random.random() < 0.3:
            sequence = rotate_around_y(sequence)
        return sequence
    
#####################
# Pose Normalization
#####################
def normalize_pose(sequence):
    sequence = np.array(sequence, dtype=np.float32)
    sequence = sequence.reshape(sequence.shape[0], 33, 4)

    center = sequence[:, 23, :3]  # use point 23 (mid-hip) as center
    sequence[:, :, :3] -= center[:, np.newaxis, :]

    scale = np.linalg.norm(sequence[:, 11, :2] - sequence[:, 23, :2], axis=1, keepdims=True)
    scale = np.clip(scale, a_min=1e-5, a_max=None)
    sequence[:, :, :2] /= scale[:, np.newaxis]

    return sequence.reshape(sequence.shape[0], -1)

#####################
# DataLoader Preparation
#####################
def get_dataloaders(data_root='split_data', batch_size=8):
    train_dataset = FitnessDataset(data_root, split='train', augment=True)
    val_dataset = FitnessDataset(data_root, split='test', augment=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return train_loader, val_loader


def split_dataset_files(source_dir='skeleton_data', output_root='split_data', train_ratio=0.8):
    os.makedirs(output_root, exist_ok=True)
    train_dir = os.path.join(output_root, 'train')
    test_dir = os.path.join(output_root, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    label_dirs = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]

    for label in label_dirs:
        all_files = [f for f in os.listdir(os.path.join(source_dir, label)) if f.endswith('.json')]
        random.shuffle(all_files)
        split_point = int(len(all_files) * train_ratio)
        train_files = all_files[:split_point]
        test_files = all_files[split_point:]

        os.makedirs(os.path.join(train_dir, label), exist_ok=True)
        os.makedirs(os.path.join(test_dir, label), exist_ok=True)

        for f in train_files:
            shutil.copy(os.path.join(source_dir, label, f), os.path.join(train_dir, label, f))
        for f in test_files:
            shutil.copy(os.path.join(source_dir, label, f), os.path.join(test_dir, label, f))

        print(f"Copied {len(train_files)} files to {os.path.join(train_dir, label)}")
        print(f"Copied {len(test_files)} files to {os.path.join(test_dir, label)}")
