import json
import os
import random
import shutil
import numpy as np
import trimesh
import torch
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

    def _build_file_list_and_label_map(self):
        dir = os.path.join(self.root_dir, self.split)
        label_names = sorted(os.listdir(dir))
        label_map = {label: idx for idx, label in enumerate(label_names)}
        for label in label_names:
            class_dir = os.path.join(dir, label)
            for f in os.listdir(class_dir):
                if f.endswith('.json'):
                    self.files.append(os.path.join(class_dir, f))
                    self.labels.append(label_map[label])
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
    labels = torch.tensor(labels)[perm_idx]

    # Pack
    packed_input = pack_padded_sequence(padded, lengths.cpu(), batch_first=True, enforce_sorted=True)

    return packed_input, labels

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

class RandomTransform:
    def __call__(self, sequence):
        if random.random() < 0.5:
            sequence = add_noise(sequence)
        if random.random() < 0.5:
            sequence = horizontal_flip(sequence)
        if random.random() < 0.3:
            sequence = time_warp(sequence, rate=random.uniform(0.8, 1.2))
        return sequence
    

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
