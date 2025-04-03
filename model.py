import torch
import torch.nn as nn
import torch.nn.functional as F


def build_mediapipe_adjacency(num_joints=33):
    connections = [
        # Torso & Arms
        (11, 13),  # left_shoulder -> left_elbow
        (13, 15),  # left_elbow -> left_wrist
        (12, 14),  # right_shoulder -> right_elbow
        (14, 16),  # right_elbow -> right_wrist
        (11, 12),  # left_shoulder -> right_shoulder
        (23, 24),  # left_hip -> right_hip
        (11, 23),  # left_shoulder -> left_hip
        (12, 24),  # right_shoulder -> right_hip

        # Legs
        (23, 25),  # left_hip -> left_knee
        (25, 27),  # left_knee -> left_ankle
        (24, 26),  # right_hip -> right_knee
        (26, 28),  # right_knee -> right_ankle

        # Feet
        (27, 29),  # left_ankle -> left_heel
        (29, 31),  # left_heel -> left_foot_index
        (28, 30),  # right_ankle -> right_heel
        (30, 32),  # right_heel -> right_foot_index

        # Hands/fingers (mainly wrist -> index/pinky/thumb)
        (15, 17),  # left_wrist -> left_pinky
        (15, 19),  # left_wrist -> left_index
        (15, 21),  # left_wrist -> left_thumb
        (16, 18),  # right_wrist -> right_pinky
        (16, 20),  # right_wrist -> right_index
        (16, 22),  # right_wrist -> right_thumb
    ]
    
    A = torch.zeros(num_joints, num_joints)
    for (i, j) in connections:
        A[i, j] = 1
        A[j, i] = 1  # undirected
    
    return A

class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
    
    def forward(self, x, adj):
        # Summation of neighbor features:
        x_t = x.permute(0, 2, 1)                # => [B, in_features, J]
        neighbor_sum = torch.matmul(x_t, adj)   # => [B, in_features, J]
        neighbor_sum = neighbor_sum.permute(0, 2, 1)  # => [B, J, in_features]
        
        out = self.linear(neighbor_sum)
        out = F.relu(out)
        return out

class GNNBiLSTMModel(nn.Module):
    def __init__(self, in_features, gcn_hidden, lstm_hidden, num_actions):
        super().__init__()
        
        # Two GCN layers
        self.gcn1 = GCNLayer(in_features, gcn_hidden)
        self.gcn2 = GCNLayer(gcn_hidden, gcn_hidden)
        
        # Bi-LSTM for temporal encoding
        # (bidirectional => output = 2*lstm_hidden per frame)
        self.lstm = nn.LSTM(
            input_size=gcn_hidden,
            hidden_size=lstm_hidden,
            batch_first=True,
            bidirectional=True
        )
        
        # Classifier for frame-level predictions
        self.classifier = nn.Linear(2 * lstm_hidden, num_actions)

        self.regressor = nn.Linear(2*lstm_hidden, 1)
    
    def forward(self, x, adj):
        B, T, J, F = x.shape
        
        # (1) Flatten so GCN can process each frame independently
        x_reshaped = x.reshape(B * T, J, F)  # => [B*T, J, F]
        
        # (2) GCN layers
        out = self.gcn1(x_reshaped, adj)   # => [B*T, J, gcn_hidden]
        out = self.gcn2(out, adj)         # => [B*T, J, gcn_hidden]
        
        # (3) Pool over joints => single embedding per frame
        out = out.mean(dim=1)             # => [B*T, gcn_hidden]
        
        # (4) Reshape to [B, T, gcn_hidden]
        out = out.view(B, T, -1)
        
        # (5) Bi-LSTM => [B, T, 2*lstm_hidden]
        lstm_out, _ = self.lstm(out)
        
        # 6) Use final time step => [B, 2*lstm_hidden]
        final_feat = lstm_out[:, -1, :]

        # 7) Heads:
        class_logits = self.classifier(final_feat)  # => [B, num_actions]
        rep_pred     = self.regressor(final_feat)   # => [B, 1]
        
        return class_logits, rep_pred
    
