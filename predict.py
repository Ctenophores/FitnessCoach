import json
import os
import sys
import torch
import numpy as np
from model import GNNBiLSTMModel  # Replace with your actual model class
from dataset import normalize_pose


def segment_predictions(pred_seq):
    segments = []
    if len(pred_seq) == 0:
        return segments

    current_action = pred_seq[0]
    start_frame = 0
    for t in range(1, len(pred_seq)):
        if pred_seq[t] != current_action:
            segments.append((current_action, start_frame, t - 1))
            current_action = pred_seq[t]
            start_frame = t
    segments.append((current_action, start_frame, len(pred_seq) - 1))
    return segments

def main(json_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the JSON file
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    sequence = []
    for frame in data:
        # If the frame doesn't have a pose, use a default value.
        pose = frame.get('pose', [[0, 0, 0, 0]] * 33)
        # Flatten the list-of-lists: 33 joints * 4 features = 132 features.
        flat_pose = [v for joint in pose for v in joint]
        sequence.append(flat_pose)
    
    sequence = np.array(sequence, dtype=np.float32)
    # Normalize the pose as per training preprocessing.
    sequence = normalize_pose(sequence)  # shape: [T, 132]
    T = sequence.shape[0]
    
    # The model expects input of shape [B, T, 33, in_features] where in_features = 4.
    sequence_tensor = torch.tensor(sequence, dtype=torch.float32).view(1, T, 33, 4)
    sequence_tensor = sequence_tensor.to(device)
    
    # Define the model hyperparameters (must match training).
    in_features = 4       # Each joint has 4 features.
    gcn_hidden = 32       # Example value used during training.
    lstm_hidden = 64      # Example value used during training.
    num_actions = 3       # Number of action classes (e.g., PushUps, Squats, JumpingJacks).
    
    # Instantiate the model and load the checkpoint.
    model = GNNBiLSTMModel(in_features, gcn_hidden, lstm_hidden, num_actions)
    model = model.to(device)
    
    checkpoint_path = r"C:\Users\yeeli\Desktop\Lab\Final\FitnessCoach\FitnessCoach\checkpoints\model_epoch_901.pt"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        print("Checkpoint not found. Using untrained model.")
    
    num_joints = 33
    adj = torch.eye(num_joints, dtype=torch.float32).to(device)
    # Evaluate the model on the sequence.
    model.eval()
    with torch.no_grad():
        # If your model expects an adjacency matrix, replace None with the proper matrix.
        action_logits, _ = model(sequence_tensor, adj)
        # Assuming action_logits is of shape [1, T, num_actions], get per-frame predictions.
        predicted_actions = torch.argmax(action_logits, dim=-1)  # shape: [1, T]
        predicted_actions = predicted_actions.squeeze(0).cpu().numpy()  # shape: [T]
        predicted_actions = np.atleast_1d(predicted_actions)
    
    # Segment the predictions into contiguous intervals.
    segments = segment_predictions(predicted_actions)
    
    # Define a mapping from numeric labels to action names.
    # This mapping should match how your dataset was constructed.
    label_mapping = {0: "JumpingJacks", 1: "PushUps", 2: "Squats"}
    
    # Print the segmentation results.
    print("Predicted Action Segments:")
    for seg in segments:
        action_idx, start, end = seg
        action_name = label_mapping.get(action_idx, str(action_idx))
        duration = end - start + 1
        print(f"  Action: {action_name} (Label {action_idx}) from Frame {start} to {end} (Duration: {duration} frames)")
    
    # Count how many times each action appears.
    action_counts = {}
    for seg in segments:
        action_idx = seg[0]
        action_name = label_mapping.get(action_idx, str(action_idx))
        action_counts[action_name] = action_counts.get(action_name, 0) + 1
    print("\nAction Occurrences:")
    for action_name, count in action_counts.items():
        print(f"  {action_name}: {count} time(s)")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict_json.py <path_to_json>")
    else:
        json_path = sys.argv[1]
        main(json_path)