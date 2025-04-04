import json
import os
import sys
import torch
import numpy as np
from model import GNNBiLSTMModel  # Replace with your actual model class
from dataset import normalize_pose
from model import build_mediapipe_adjacency
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


def segment_reps_from_peaks(peaks, total_frames):
    segments = []
    if len(peaks) == 0:
        return segments
    if len(peaks) == 1:
        # If only one rep is detected, define a small window around it.
        start = max(0, peaks[0] - 5)
        end = min(total_frames - 1, peaks[0] + 5)
        segments.append((start, end))
        return segments
    
    # First rep:
    first_start = max(0, int(peaks[0] - (peaks[1] - peaks[0]) / 2))
    first_end = int((peaks[0] + peaks[1]) / 2)
    segments.append((first_start, first_end))
    
    # Intermediate reps:
    for i in range(1, len(peaks) - 1):
        start = int((peaks[i - 1] + peaks[i]) / 2) + 1
        end = int((peaks[i] + peaks[i + 1]) / 2)
        segments.append((start, end))
    
    # Last rep:
    last_start = int((peaks[-2] + peaks[-1]) / 2) + 1
    last_end = min(total_frames - 1, int(peaks[-1] + (peaks[-1] - peaks[-2]) / 2))
    segments.append((last_start, last_end))
    
    return segments

def main(json_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the JSON file
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    sequence = []
    for frame in data:
        pose = frame.get('pose', None)
        # Ensure we have exactly 33 joints; if not, fill with default values.
        if pose is None or len(pose) != 33:
            pose = [[0, 0, 0, 0]] * 33
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
    
    checkpoint_path = r"C:\Users\yeeli\Desktop\Lab\Final\FitnessCoach\FitnessCoach\checkpoints\model_epoch_1000.pt"
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
    adj = build_mediapipe_adjacency(num_joints).float().to(device)
    # Evaluate the model on the sequence.
    model.eval()
    with torch.no_grad():
        action_logits, rep_count_pred, frame_scores = model(sequence_tensor, adj)
        
        # For overall action classification, use the final time step:
        final_action_logits = action_logits
        predicted_action = torch.argmax(final_action_logits, dim=-1).item()
        
        # Get the predicted rep count
        predicted_rep_count = int(round(rep_count_pred.squeeze().item()))
        
        # Process frame_scores if desired.
        # For instance, you might threshold them to mark rep-phase frames:
        frame_scores = frame_scores.cpu().numpy()  # shape: [T]
        phase_prob = 1 / (1 + np.exp(-frame_scores))
        phase_prob = np.squeeze(phase_prob)  # ensure 1-D array

        # Detect candidate peaks with a lower threshold to capture more candidates.
        candidate_peaks, properties = find_peaks(phase_prob, height=0.02, distance=5)

        # Get the predicted rep count (an integer).
        predicted_rep_count = int(round(rep_count_pred.squeeze().item()))

        # If there are more candidate peaks than the predicted count, select the top ones.
        if len(candidate_peaks) > predicted_rep_count and predicted_rep_count > 0:
            # Sort candidate peaks by their peak height in descending order.
            peak_heights = properties['peak_heights']
            # Get indices that would sort peaks by height.
            sorted_idx = np.argsort(peak_heights)[::-1]
            # Select the top predicted_rep_count peaks.
            selected_peaks = candidate_peaks[sorted_idx][:predicted_rep_count]
            # Then sort the selected peaks in ascending order (so they are in time order).
            selected_peaks = np.sort(selected_peaks)
        else:
            # If fewer peaks are detected, use them as is.
            selected_peaks = candidate_peaks

        print("Selected peaks:", selected_peaks)

        # Now, define rep segments based on these peaks.
        def segment_reps_from_selected_peaks(peaks, total_frames):
            segments = []
            if len(peaks) == 0:
                return segments
            if len(peaks) == 1:
                start = max(0, peaks[0] - 5)
                end = min(total_frames - 1, peaks[0] + 5)
                segments.append((start, end))
                return segments
    
            # For the first rep, we might start at frame 0 or use a midpoint.
            first_start = max(0, int(peaks[0] - (peaks[1] - peaks[0]) / 2))
            first_end = int((peaks[0] + peaks[1]) / 2)
            segments.append((first_start, first_end))
    
            # For intermediate reps:
            for i in range(1, len(peaks) - 1):
                start = int((peaks[i - 1] + peaks[i]) / 2) + 1
                end = int((peaks[i] + peaks[i + 1]) / 2)
                segments.append((start, end))
    
            # For the last rep:
            last_start = int((peaks[-2] + peaks[-1]) / 2) + 1
            last_end = min(total_frames - 1, int(peaks[-1] + (peaks[-1] - peaks[-2]) / 2))
            segments.append((last_start, last_end))
            return segments

        rep_segments = segment_reps_from_selected_peaks(selected_peaks, T)
    
    label_mapping = {0: "JumpingJacks", 1: "PushUps", 2: "Squats"}
    
    print("\nPredicted Overall Action:")
    print(f"  {label_mapping.get(predicted_action, str(predicted_action))} (Label {predicted_action})")
    print(f"Predicted Rep Count: {predicted_rep_count}\n")
    
    if rep_segments:
        print("Detected Rep Segments:")
        for i, seg in enumerate(rep_segments, 1):
            start, end = seg
            print(f"  Rep {i}: Frames {start} to {end} (Duration: {end - start + 1} frames)")
    else:
        print("No rep segments detected based on phase peaks.")
    
    # Optionally, plot frame phase scores.
    plt.figure(figsize=(10, 4))
    plt.plot(phase_prob, label="Phase Probability")
    plt.plot(selected_peaks, phase_prob[selected_peaks], "ro", label="Detected Peaks")
    plt.title("Frame-level Phase Probability")
    plt.xlabel("Frame")
    plt.ylabel("Probability")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict_json.py <path_to_json>")
    else:
        json_path = sys.argv[1]
        main(json_path)