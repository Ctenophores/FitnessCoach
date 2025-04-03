import os
import torch
import torch.nn as nn
import torch.optim as optim

from dataset import get_dataloaders, collate_fn

from model import GNNBiLSTMModel, build_mediapipe_adjacency
import matplotlib.pyplot as plt

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Hyperparameters
    in_features = 4      # (x, y, z, vis) or whatever your input
    gcn_hidden = 32
    lstm_hidden = 64
    num_classes = 3      # e.g. squat=0, pushup=1, lunge=2, etc.
    save_freq = 100

    # Initialize model
    model = GNNBiLSTMModel(in_features, gcn_hidden, lstm_hidden, num_classes).to(device)

    # Build adjacency
    adj = build_mediapipe_adjacency(num_joints=33).float().to(device)
    
    # Optimizer & loss
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    criterion_rep = nn.MSELoss()
    
    # Get data loaders from your dataset.py
    train_loader, val_loader = get_dataloaders(data_root='split_data', batch_size=8)
    
    # Training loop
    epochs = 1000
    train_losses = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_samples = 0
        
        for packed_input, (action_labels, rep_counts) in train_loader:
            action_labels = action_labels.to(device)
            rep_counts = rep_counts.to(device)
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
            action_logits, rep_count_pred = model(padded, adj)
            
            # If you have 1 label per entire sequence, pick final step
            # shape => [B, num_classes]
            final_action_logits = action_logits
            final_rep_count_pred = rep_count_pred
            
            # If you have 1 label per sequence => labels shape [B]
            # Cross-entropy
            alpha = 1.0
            loss_action = criterion(final_action_logits, action_labels)
            loss_rep = criterion_rep(final_rep_count_pred, rep_counts)
            loss = loss_action + alpha * loss_rep
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            batch_size = B
            total_loss += loss.item() * batch_size
            total_samples += batch_size
        
        avg_loss = total_loss / total_samples
        train_losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_loss:.4f}")

        if (epoch) % save_freq == 0:
            plot_loss(train_losses)
            save_path = f"checkpoints/model_epoch_{epoch+1}.pt"
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f"Saved model to {save_path}")



    
    # You could do a val loop, etc.

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