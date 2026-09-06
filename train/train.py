"""
PyTorch Training Script for Violence Detection.

Trains the Temporal Transformer classifier on pre-extracted DINOv2 features.
Logs metrics (accuracy, loss, ROC AUC) per epoch to CSV and checkpoints the
model after every epoch.
Uses PyTorch Dataset and DataLoader for efficient batching.
"""

import csv
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    FEATURE_OUTPUT_DIR, TRAIN_OUTPUT_DIR,
    FEATURE_DIM, SEQUENCE_LENGTH, BATCH_SIZE,
    NUM_EPOCHS, INITIAL_LR, LR_DECAY_RATE,
    RANDOM_SEED,
)
from models.temporal_extractor import TemporalTransformer

# Reproducibility
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)


class ViolenceFeatureDataset(Dataset):
    """
    PyTorch Dataset that extracts sliding windows of length `seq_len` from 
    continuous feature arrays, ensuring no window crosses a video boundary.
    """
    def __init__(self, features, labels, new_idx, seq_len=SEQUENCE_LENGTH):
        self.features = features
        self.labels = labels
        self.seq_len = seq_len
        
        # Create a boolean array where True means a new video starts
        is_boundary = np.zeros(len(features), dtype=bool)
        if len(new_idx) > 0:
            is_boundary[new_idx] = True
            
        # Pre-compute valid starting indices
        # An index `i` is valid if `[i, i + seq_len)` does not contain any boundary
        # i.e., no True values in `is_boundary[i+1 : i+seq_len]`
        self.valid_indices = []
        for i in range(len(features) - seq_len + 1):
            if not np.any(is_boundary[i+1 : i + seq_len]):
                self.valid_indices.append(i)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        start_idx = self.valid_indices[idx]
        end_idx = start_idx + self.seq_len
        
        # Extract features (seq_len, 768)
        x = self.features[start_idx : end_idx]
        x_tensor = torch.from_numpy(x).float()
        
        # Target: last frame's label → [violence, non-violence]
        last_label = self.labels[end_idx - 1]
        y_tensor = torch.tensor([last_label, 1 - last_label], dtype=torch.float32)
        
        return x_tensor, y_tensor


def load_features():
    """Load pre-extracted DINOv2 features and labels."""
    train_features = np.load(os.path.join(FEATURE_OUTPUT_DIR, "train.npy"), allow_pickle=True)
    train_labels = np.load(os.path.join(FEATURE_OUTPUT_DIR, "train_lbl.npy"), allow_pickle=True)
    train_new_idx = np.load(os.path.join(FEATURE_OUTPUT_DIR, "train_images_new_idx.npy"), allow_pickle=True)

    test_features = np.load(os.path.join(FEATURE_OUTPUT_DIR, "test.npy"), allow_pickle=True)
    test_labels = np.load(os.path.join(FEATURE_OUTPUT_DIR, "test_lbl.npy"), allow_pickle=True)
    test_new_idx = np.load(os.path.join(FEATURE_OUTPUT_DIR, "test_images_new_idx.npy"), allow_pickle=True)

    return train_features, train_labels, train_new_idx, test_features, test_labels, test_new_idx


def train_one_epoch(model, optimizer, criterion, dataloader, device):
    """Train for one epoch using DataLoader."""
    model.train()
    total_loss = 0.0
    total_correct = 0
    count = 0

    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        
        optimizer.zero_grad()
        output = model(batch_x)  # (batch_size, 2)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_x.size(0)
        
        # batch_y is [violence_prob, non_violence_prob], true label is 1 if index 0 is 1.0
        predicted = (output[:, 0] > 0.5).int()
        labels = batch_y[:, 0].int()
        
        total_correct += (predicted == labels).sum().item()
        count += batch_x.size(0)

    avg_loss = total_loss / max(count, 1)
    avg_acc = total_correct / max(count, 1)
    return avg_loss, avg_acc


@torch.no_grad()
def evaluate(model, criterion, dataloader, device):
    """Evaluate model on test DataLoader."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    count = 0
    all_outputs = []
    all_labels = []

    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        
        output = model(batch_x)
        loss = criterion(output, batch_y)

        total_loss += loss.item() * batch_x.size(0)
        
        predicted = (output[:, 0] > 0.5).int()
        labels = batch_y[:, 0].int()
        
        total_correct += (predicted == labels).sum().item()
        count += batch_x.size(0)

        all_outputs.extend(output[:, 0].cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / max(count, 1)
    avg_acc = total_correct / max(count, 1)

    # ROC AUC
    all_labels_np = np.array(all_labels)
    all_outputs_np = np.array(all_outputs)
    try:
        roc_auc = roc_auc_score(all_labels_np, all_outputs_np)
    except ValueError:
        roc_auc = 0.0

    return avg_loss, avg_acc, roc_auc, all_labels_np, all_outputs_np


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    print("Loading features...")
    train_features, train_labels, train_new_idx, test_features, test_labels, test_new_idx = load_features()
    print(f"Train features: {train_features.shape}, Test features: {test_features.shape}")

    # Create Datasets and DataLoaders
    print("Building datasets...")
    train_dataset = ViolenceFeatureDataset(train_features, train_labels, train_new_idx)
    test_dataset = ViolenceFeatureDataset(test_features, test_labels, test_new_idx)
    
    if len(train_dataset) == 0:
        print("Error: Train dataset is empty. Check your SEQUENCE_LENGTH or data preprocessing.")
        return

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"Train batches: {len(train_loader)} | Test batches: {len(test_loader)}")

    # Model
    model = TemporalTransformer().to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}")

    # Optimizer & Loss
    optimizer = torch.optim.SGD(model.parameters(), lr=INITIAL_LR)
    criterion = nn.BCELoss()

    # Output directory
    os.makedirs(os.path.join(TRAIN_OUTPUT_DIR, "models"), exist_ok=True)

    # CSV log
    csv_path = os.path.join(TRAIN_OUTPUT_DIR, "training_log.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Train_Acc", "Train_Loss", "Test_Acc", "Test_Loss", "Test_ROC_AUC", "LR"])

    train_losses, test_losses = [], []

    checkpoint_path = os.path.join(TRAIN_OUTPUT_DIR, "models", "last_model.pt")

    for epoch in range(1, NUM_EPOCHS + 1):
        current_lr = INITIAL_LR * (LR_DECAY_RATE ** (epoch - 1))
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

        start_time = time.time()

        # Train
        train_loss, train_acc = train_one_epoch(
            model, optimizer, criterion, train_loader, device
        )

        # Evaluate
        test_loss, test_acc, test_roc, test_labels_used, test_outputs = evaluate(
            model, criterion, test_loader, device
        )

        epoch_time = time.time() - start_time

        print(
            f"Epoch {epoch}/{NUM_EPOCHS} | "
            f"Train Acc: {train_acc:.4f} Loss: {train_loss:.4f} | "
            f"Test Acc: {test_acc:.4f} Loss: {test_loss:.4f} ROC: {test_roc:.4f} | "
            f"LR: {current_lr:.6f} | Time: {epoch_time:.1f}s"
        )

        # Confusion matrix
        predicted_labels = [1 if x > 0.5 else 0 for x in test_outputs]
        cm = confusion_matrix(test_labels_used, predicted_labels)
        print(f"  Confusion Matrix: {cm.tolist()}")

        # Checkpoint after every epoch
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "test_loss": test_loss,
            "test_roc": test_roc,
        }, checkpoint_path)

        # Log to CSV
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_acc, train_loss, test_acc, test_loss, test_roc, current_lr])

        train_losses.append(train_loss)
        test_losses.append(test_loss)

        # Plot ROC curve for last epoch
        if epoch == NUM_EPOCHS:
            fpr, tpr, _ = roc_curve(test_labels_used, test_outputs)
            plt.figure()
            plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC AUC = {test_roc:.4f}")
            plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("ROC Curve")
            plt.legend(loc="lower right")
            plt.savefig(os.path.join(TRAIN_OUTPUT_DIR, "roc_curve.png"), dpi=150)
            plt.close()

    # Plot loss curves
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(test_losses, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training / Test Loss")
    plt.legend()
    plt.savefig(os.path.join(TRAIN_OUTPUT_DIR, "loss_curve.png"), dpi=150)
    plt.close()

    print(f"\nTraining complete! Checkpoints saved to {TRAIN_OUTPUT_DIR}/models/")
    print(f"Logs saved to {csv_path}")


if __name__ == "__main__":
    main()
