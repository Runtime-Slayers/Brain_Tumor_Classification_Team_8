"""
=============================================================================
BRAIN TUMOR CLASSIFICATION — TEAM 8
File: brain_tumor_train.py
Role: Model Training, Optimization & Convergence Monitoring
=============================================================================

Author      : Team 8
Project     : Brain Tumor Classification using Deep Learning & Explainable AI

Description :
    Full training loop for the Attention-Gated ResNet-34 model.
    Implements:
        - Weighted Cross-Entropy Loss (class imbalance compensation)
        - Label Smoothing (ε=0.1) to prevent over-confident boundary predictions
        - AdamW optimizer with decoupled weight decay (L2 regularization)
        - Cosine Annealing Warm Restarts learning rate schedule
        - Early Stopping with configurable patience
        - Best-model checkpoint saving (validation-loss triggered)
        - Training/validation loss and accuracy curve generation
=============================================================================
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')   # Non-interactive backend for server/headless environments

from brain_tumor_dataset import get_dataloaders
from brain_tumor_model   import BrainTumorClassifier


# ─────────────────────────────────────────────────────────────────────────────
# HYPERPARAMETER CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

CONFIG = {
    # Dataset
    "dataset_dir"   : "Dataset",    # Root dataset directory
    "batch_size"    : 32,           # SGD mini-batch size
    "use_clahe"     : True,         # Apply CLAHE preprocessing

    # Model
    "num_classes"   : 4,            # Glioma, Meningioma, Pituitary, No Tumor
    "dropout_p"     : 0.4,          # Monte Carlo Dropout probability

    # Training
    "num_epochs"    : 60,           # Max training epochs (early stopping may terminate earlier)
    "patience"      : 10,           # Early stopping patience (epochs without val improvement)
    "label_smoothing": 0.1,         # Label smoothing ε — prevents over-confident logits

    # Optimizer (AdamW)
    "learning_rate" : 1e-3,         # Initial learning rate α
    "weight_decay"  : 1e-4,         # L2 regularization coefficient λ

    # LR Scheduler (Cosine Annealing Warm Restarts)
    "T_0"           : 10,           # Epochs per first restart cycle
    "T_mult"        : 2,            # Geometric growth factor for subsequent cycles

    # Outputs
    "checkpoint_path"    : "best_model.pth",
    "training_plot_path" : "training_curves.png",
}


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Execute one full training epoch over the training DataLoader.

    Sets model to training mode (enables BatchNorm statistics updates and
    Dropout stochasticity). Iterates over all mini-batches, computes weighted
    cross-entropy loss with label smoothing, backpropagates gradients, and
    updates model parameters via AdamW optimizer step.

    Args:
        model      : BrainTumorClassifier instance.
        dataloader : PyTorch DataLoader for training split.
        criterion  : Weighted Cross-Entropy loss function.
        optimizer  : AdamW optimizer instance.
        device     : torch.device ('cuda' or 'cpu').

    Returns:
        tuple: (epoch_loss [float], epoch_accuracy [float])
    """
    model.train()                   # Activates BatchNorm updates + Dropout

    running_loss  = 0.0
    correct_preds = 0
    total_samples = 0

    for batch_idx, (images, labels) in enumerate(dataloader):
        # Transfer batch tensors to compute device (GPU/CPU)
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Zero accumulated gradients from previous iteration
        optimizer.zero_grad()

        # Forward pass through Attention-Gated ResNet-34
        logits = model(images)                          # [B, num_classes]

        # Compute weighted Cross-Entropy loss with label smoothing
        loss = criterion(logits, labels)

        # Backpropagation — compute analytical gradients via autograd engine
        loss.backward()

        # Gradient clipping: prevents exploding gradients during early training
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        # AdamW parameter update step
        optimizer.step()

        # Accumulate statistics
        running_loss  += loss.item() * images.size(0)
        _, predictions = logits.max(dim=1)
        correct_preds  += predictions.eq(labels).sum().item()
        total_samples  += labels.size(0)

    epoch_loss = running_loss  / total_samples
    epoch_acc  = correct_preds / total_samples
    return epoch_loss, epoch_acc


def validate_epoch(model, dataloader, criterion, device):
    """
    Evaluate model performance on the validation/test DataLoader.

    Sets model to evaluation mode (freezes BatchNorm and disables Dropout).
    No gradient computation performed (torch.no_grad() context).

    Args:
        model      : BrainTumorClassifier instance.
        dataloader : PyTorch DataLoader for test/validation split.
        criterion  : Loss function (same as training).
        device     : torch.device.

    Returns:
        tuple: (epoch_loss [float], epoch_accuracy [float])
    """
    model.eval()                    # Disables Dropout; freezes BN running stats

    running_loss  = 0.0
    correct_preds = 0
    total_samples = 0

    with torch.no_grad():           # Suppress gradient tape for memory efficiency
        for images, labels in dataloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(images)
            loss   = criterion(logits, labels)

            running_loss  += loss.item() * images.size(0)
            _, predictions = logits.max(dim=1)
            correct_preds  += predictions.eq(labels).sum().item()
            total_samples  += labels.size(0)

    epoch_loss = running_loss  / total_samples
    epoch_acc  = correct_preds / total_samples
    return epoch_loss, epoch_acc


def plot_training_curves(train_losses, val_losses, train_accs, val_accs, output_path):
    """
    Save training and validation loss / accuracy curves as a PNG figure.

    The convergence plot is essential for diagnosing:
        - Overfitting: val loss diverges upward while train loss continues falling.
        - Underfitting: both curves remain high; model lacks sufficient capacity.
        - Correct fit: both curves converge in parallel with minimal divergence.

    Args:
        train_losses (list): Per-epoch training loss values.
        val_losses   (list): Per-epoch validation loss values.
        train_accs   (list): Per-epoch training accuracy values.
        val_accs     (list): Per-epoch validation accuracy values.
        output_path  (str) : File path to save the figure (PNG).
    """
    epochs = range(1, len(train_losses) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Attention-Gated ResNet-34 — Training Convergence", fontsize=14, fontweight='bold')

    # Loss subplot
    axes[0].plot(epochs, train_losses, 'b-o', markersize=3, label='Training Loss')
    axes[0].plot(epochs, val_losses,   'r-o', markersize=3, label='Validation Loss')
    axes[0].set_title("Loss vs. Epochs")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy subplot
    axes[1].plot(epochs, train_accs, 'b-o', markersize=3, label='Training Accuracy')
    axes[1].plot(epochs, val_accs,   'r-o', markersize=3, label='Validation Accuracy')
    axes[1].set_title("Accuracy vs. Epochs")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Training] Convergence curves saved → {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN TRAINING PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def train_model(config: dict = CONFIG):
    """
    Full end-to-end training pipeline for the Brain Tumor Classifier.

    Pipeline Stages:
        1. Device detection (CUDA GPU preferred; CPU fallback).
        2. Dataset loading with CLAHE preprocessing and class weight computation.
        3. Model instantiation with pre-trained ResNet-34 backbone.
        4. Loss, optimizer, and LR scheduler configuration.
        5. Epoch loop: train → validate → early stopping check → checkpoint save.
        6. Training curve generation.

    Args:
        config (dict): Hyperparameter configuration dictionary (default: CONFIG above).
    """
    # ── Device Detection ───────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*65}")
    print(f"  Brain Tumor Classifier — Training Pipeline — Team 8")
    print(f"{'='*65}")
    print(f"[Setup] Compute device   : {device}")
    if device.type == 'cuda':
        print(f"[Setup] GPU              : {torch.cuda.get_device_name(0)}")
        print(f"[Setup] VRAM available   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── Data Loading ───────────────────────────────────────────────────────
    train_loader, test_loader, classes, class_weights = get_dataloaders(
        root_dir   = config["dataset_dir"],
        batch_size = config["batch_size"],
        use_clahe  = config["use_clahe"]
    )

    # ── Model Initialization ───────────────────────────────────────────────
    model = BrainTumorClassifier(
        num_classes = config["num_classes"],
        dropout_p   = config["dropout_p"]
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[Model] Parameters       : {total_params:,}")

    # ── Loss Function: Weighted Cross-Entropy + Label Smoothing ───────────
    # class_weights tensor is moved to the same device as the model.
    # label_smoothing=0.1 softens one-hot targets to [0.025, ..., 0.925, ..., 0.025]
    # preventing the network from pushing logit margins to extremes on noisy scans.
    criterion = nn.CrossEntropyLoss(
        weight          = class_weights.to(device),
        label_smoothing = config["label_smoothing"]
    )

    # ── AdamW Optimizer ────────────────────────────────────────────────────
    # AdamW decouples weight decay from adaptive gradient scaling, providing
    # superior L2 regularization behaviour compared to standard Adam.
    optimizer = optim.AdamW(
        model.parameters(),
        lr           = config["learning_rate"],
        weight_decay = config["weight_decay"]
    )

    # ── Cosine Annealing LR Scheduler ──────────────────────────────────────
    # Decays learning rate following a cosine curve, periodically restarting
    # from a higher rate (warm restarts) to escape local minima.
    # T_0=10: first restart after 10 epochs; T_mult=2: doubles cycle length each restart.
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=config["T_0"], T_mult=config["T_mult"]
    )

    # ── Training Loop ──────────────────────────────────────────────────────
    train_losses, val_losses = [], []
    train_accs,   val_accs   = [], []

    best_val_loss      = float('inf')
    epochs_no_improve  = 0

    print(f"\n[Training] Launching {config['num_epochs']}-epoch training loop...\n")
    print(f"{'Epoch':>6} | {'Train Loss':>10} | {'Train Acc':>9} | {'Val Loss':>8} | {'Val Acc':>7} | {'LR':>10}")
    print("-" * 65)

    for epoch in range(1, config["num_epochs"] + 1):

        # Forward + backward pass over all training batches
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Evaluation pass on held-out test split (no augmentation, no dropout)
        val_loss, val_acc = validate_epoch(model, test_loader, criterion, device)

        # Step LR scheduler (cosine decay with periodic warm restarts)
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # Record metrics for convergence curve plotting
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        # Print epoch summary
        print(f"{epoch:>6} | {train_loss:>10.4f} | {train_acc:>8.4f}% | {val_loss:>8.4f} | {val_acc:>6.4f}% | {current_lr:>10.6f}")

        # ── Early Stopping & Checkpoint Logic ──────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss     = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), config["checkpoint_path"])
            print(f"         *** Best model saved (val_loss={best_val_loss:.4f}) ***")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= config["patience"]:
                print(f"\n[Training] Early stopping triggered at epoch {epoch}.")
                print(f"[Training] No improvement in {config['patience']} consecutive epochs.")
                break

    # ── Post-Training ──────────────────────────────────────────────────────
    plot_training_curves(
        train_losses, val_losses, train_accs, val_accs,
        output_path=config["training_plot_path"]
    )

    print(f"\n[Training] Complete. Best validation loss  : {best_val_loss:.4f}")
    print(f"[Training] Best model checkpoint saved at  : {config['checkpoint_path']}")
    print(f"[Training] Convergence curves saved at     : {config['training_plot_path']}")
    print(f"{'='*65}\n")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRYPOINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    train_model(config=CONFIG)
