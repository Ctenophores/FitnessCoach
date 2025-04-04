import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from dataset import get_dataloaders, collate_fn

from model import GNNBiLSTMModel, build_mediapipe_adjacency
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Hyperparameters
    in_features = 4      # (x, y, z, vis) or whatever your input
    gcn_hidden = 32
    lstm_hidden = 64
    num_classes = 3      # e.g. squat=0, pushup=1, lunge=2, etc.
    save_freq = 100
    eval_freq = 20
    

    # Initialize model
    model = GNNBiLSTMModel(in_features, gcn_hidden, lstm_hidden, num_classes).to(device)

    # Build adjacency
    adj = build_mediapipe_adjacency(num_joints=33).float().to(device)
    
    # Optimizer & loss
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    criterion_rep = nn.MSELoss()
    criterion_phase = nn.BCEWithLogitsLoss()
    
    # Get data loaders from your dataset.py
    train_loader, val_loader = get_dataloaders(data_root='split_data', batch_size=8)
    
    # Training loop
    epochs = 1000
    train_losses = []
    val_losses = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_samples = 0
        
        for packed_input, (action_labels, rep_counts, _) in train_loader:
            action_labels = action_labels.to(device)
            rep_counts = rep_counts.to(device)
            if rep_counts.dim() == 1:
                rep_counts = rep_counts.unsqueeze(-1)
            # We need to handle the PackedSequence. Let’s see how to do that:
            # 1) Unpack to get padded => shape [B, T, F]
            padded, lengths = torch.nn.utils.rnn.pad_packed_sequence(
                packed_input, batch_first=True
            )
            # padded => [B, T, F=132 if 33*4]
            # lengths => [B]
            
            # Reshape => [B, T, 33, 4] for GCN
            B, T, F = padded.shape
            padded = padded.view(B, T, 33, in_features).to(device)
            
            # Forward
            action_logits, rep_count_pred, frame_scores = model(padded, adj)
            
            # If you have 1 label per entire sequence, pick final step
            # shape => [B, num_classes]
            final_action_logits = action_logits
            final_rep_count_pred = rep_count_pred

            frame_labels = build_frame_labels(lengths, rep_counts, device)  # shape [B, T]
            
            # If you have 1 label per sequence => labels shape [B]
            # Cross-entropy
            alpha = 3.0
            beta = 1.0

            loss_action = criterion(final_action_logits, action_labels)
            loss_rep = criterion_rep(final_rep_count_pred, rep_counts)

            rep_phase_gt = torch.zeros(B, T, device=device)
            for b in range(B):
                r = int(round(rep_counts[b].item()))
                if r > 0:
                    # Evenly split T frames into r segments
                    divisions = np.linspace(0, T, r + 1, endpoint=True)
                    # For each segment, compute midpoint index
                    midpoints = [(divisions[i] + divisions[i+1]) / 2 for i in range(r)]
                    midpoints = [min(T - 1, max(0, int(round(mp)))) for mp in midpoints]
                    for idx in midpoints:
                        rep_phase_gt[b, idx] = 1
            # phase_logits is [B, T, 2] and ground truth is [B, T]
            loss_phase = criterion_phase(frame_scores.view(B * T), rep_phase_gt.view(B * T))

            loss = loss_action + alpha * loss_rep + beta * loss_phase
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            batch_size = B
            total_loss += loss.item() * batch_size
            total_samples += batch_size
        
        avg_loss = total_loss / total_samples
        train_losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_loss:.4f}")

        if epoch % eval_freq == 0:
            avg_val_loss, acc, mae, rmse, int_acc = evaluate(
                model, val_loader, adj, in_features, device, criterion, criterion_rep, alpha=alpha, epoch=epoch
            )
            val_losses.append(avg_val_loss)
            print(f"Epoch {epoch+1}/{epochs} "
                  f"Val Loss: {avg_val_loss:.4f} | Val Acc: {acc:.2f}% | "
                  f"Rep MAE: {mae:.2f}, RMSE: {rmse:.2f}, Int Acc: {int_acc*100:.2f}%")
            plot_loss(val_losses, save_path="val_loss_curve.png")

        if (epoch+1) % save_freq == 0:
            # avg_val_loss, acc, mae, rmse, int_acc = evaluate(
            #     model, val_loader, adj, in_features, device, criterion, criterion_rep
            # )
            # val_losses.append(avg_val_loss)
            # print(f"Epoch {epoch+1}/{epochs} "
            #       f"Val Loss: {avg_val_loss:.4f} | Val Acc: {acc:.2f}% | "
            #       f"Rep MAE: {mae:.2f}, RMSE: {rmse:.2f}, Int Acc: {int_acc*100:.2f}%")
            plot_loss(train_losses)
            save_path = f"checkpoints/model_epoch_{epoch+1}.pt"
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f"Saved model to {save_path}")

    
    # You could do a val loop, etc.
def evaluate(model, val_loader, adj, in_features, device, criterion, criterion_rep, alpha, epoch=None):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    rep_preds_all = []
    rep_trues_all = []

    with torch.no_grad():
        for packed_input, (action_labels, rep_counts, filenames) in val_loader:
            action_labels = action_labels.to(device)
            rep_counts = rep_counts.to(device)
            padded, lengths = torch.nn.utils.rnn.pad_packed_sequence(packed_input, batch_first=True)
            B, T, F = padded.shape
            padded = padded.view(B, T, 33, in_features).to(device)

            action_logits, rep_count_pred, frame_scores = model(padded, adj)


            if epoch is not None:
                os.makedirs("frame_score_plots", exist_ok=True)
                # filenames = batch_data[2] if len(batch_data) > 2 else [f"sample{i}.json" for i in range(frame_scores.shape[0])]

                for i in range(min(4, frame_scores.shape[0])):  # 最多画4张
                    scores = torch.sigmoid(frame_scores[i]).detach().cpu().numpy()
                    peaks, _ = find_peaks(scores, height=0.5, distance=10)
                    file_stem = filenames[i].replace('.json', '')
                    plt.figure(figsize=(8, 3))
                    plt.plot(scores, label="Frame Scores")
                    plt.plot(peaks, scores[peaks], "ro", label="Detected Peaks")

                    T = len(scores)
                    R = int(rep_counts[i].item())
                    if R > 0:
                        boundaries = np.linspace(0, T, R + 1, dtype=int)  # 分成 R 段
                        gt_peaks = np.array([(boundaries[i] + boundaries[i+1]) // 2 for i in range(R)])
                        plt.plot(gt_peaks, scores[gt_peaks], "g^", label="GT Peaks")

                    plt.title(f"{file_stem} - Epoch {epoch}")
                    plt.xlabel("Frame")
                    plt.ylabel("Score")
                    plt.legend()
                    plt.grid(True)
                    plt.tight_layout()
                    plt.savefig(f"frame_score_plots/{file_stem}_epoch{epoch}.png")
                    plt.close()

            loss_action = criterion(action_logits, action_labels)
            loss_rep = criterion_rep(rep_count_pred, rep_counts)
            loss = loss_action + alpha * loss_rep

            val_loss += loss.item() * B
            total += B
            pred_labels = torch.argmax(action_logits, dim=1)
            correct += (pred_labels == action_labels).sum().item()

            rep_preds_all.extend(rep_count_pred.squeeze().cpu().numpy())
            rep_trues_all.extend(rep_counts.cpu().numpy())

    avg_val_loss = val_loss / total
    accuracy = correct / total * 100
    rep_preds_all = np.array(rep_preds_all)
    rep_trues_all = np.array(rep_trues_all)
    mae = np.mean(np.abs(rep_preds_all - rep_trues_all))
    rmse = np.sqrt(np.mean((rep_preds_all - rep_trues_all) ** 2))
    int_acc = np.mean(np.round(rep_preds_all) == rep_trues_all)

    return avg_val_loss, accuracy, mae, rmse, int_acc

def build_frame_labels(lengths, rep_counts, device, sigma=5.0):
    B = len(lengths)
    max_len = max(lengths).item()
    frame_labels = torch.zeros(B, max_len, device=device)

    for i in range(B):
        T = lengths[i].item()
        R = int(rep_counts[i].item())
        if R < 1:
            continue
        centers = torch.linspace(0, T - 1, R)

        for c in centers:
            x = torch.arange(T, device=device)
            frame_labels[i, :T] += torch.exp(-0.5 * ((x - c) / sigma) ** 2)

        frame_labels[i, :T] /= frame_labels[i, :T].max() + 1e-8

    return frame_labels
def plot_loss(train_losses, save_path="loss_curve.png"):
    plt.clf() 
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.grid(True)
    plt.legend()
    plt.savefig(save_path)
    print(f"Saved loss curve to {save_path}")

if __name__ == "__main__":
    main()