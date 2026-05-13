"""
Exercise 4: CNN on MNIST with PyTorch
======================================
Trains a two-layer CNN on MNIST from scratch.
Logs train/val loss per epoch, prints test accuracy,
and shows a confusion matrix.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix, classification_report

torch.manual_seed(42)
np.random.seed(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "./data"
BATCH_SIZE = 64
EPOCHS = 5
LR = 1e-3
VAL_SPLIT = 0.2


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def get_dataloaders():
    """
    Download MNIST, split train into 80/20 train/val,
    return train_loader, val_loader, test_loader.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        # Normalise to zero mean, unit variance using MNIST statistics
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    full_train = datasets.MNIST(DATA_DIR, train=True,  download=True, transform=transform)
    test_set   = datasets.MNIST(DATA_DIR, train=False, download=True, transform=transform)

    n_val = int(len(full_train) * VAL_SPLIT)
    n_train = len(full_train) - n_val
    train_set, val_set = random_split(
        full_train, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------

class SimpleCNN(nn.Module):
    """
    Architecture:
      Conv2d(1, 32, 3)   -> 28x28 -> 26x26 feature maps (no padding)
      ReLU
      MaxPool2d(2, 2)    -> 13x13
      Conv2d(32, 64, 3)  -> 13x13 -> 11x11
      ReLU
      MaxPool2d(2, 2)    -> 5x5
      Flatten            -> 64 * 5 * 5 = 1600
      Linear(1600, 128)
      ReLU
      Linear(128, 10)    -> logits for 10 digit classes
    """

    def __init__(self):
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),   # (B, 1, 28, 28) -> (B, 32, 26, 26)
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                # -> (B, 32, 13, 13)
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3),  # -> (B, 64, 11, 11)
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                # -> (B, 64, 5, 5)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),                      # -> (B, 1600)
            nn.Linear(1600, 128),
            nn.ReLU(),
            nn.Linear(128, 10),                # -> (B, 10) raw logits
        )

    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.classifier(x)
        return x


# ---------------------------------------------------------------------------
# Training / evaluation utilities
# ---------------------------------------------------------------------------

def run_epoch(model, loader, criterion, optimizer=None):
    """
    One forward pass over the loader.
    If optimizer is provided, perform backprop + update (training mode).
    Returns mean loss and accuracy.
    """
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss, correct, total = 0.0, 0, 0

    ctx = torch.no_grad() if not training else torch.enable_grad()
    with ctx:
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            logits = model(images)
            loss = criterion(logits, labels)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(labels)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += len(labels)

    return total_loss / total, correct / total


def collect_predictions(model, loader):
    """Run inference and collect all true labels + predicted labels."""
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            preds = model(images).argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    return np.array(all_labels), np.array(all_preds)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Using device: {DEVICE}")

    train_loader, val_loader, test_loader = get_dataloaders()

    model = SimpleCNN().to(DEVICE)
    criterion = nn.CrossEntropyLoss()           # combines log-softmax + NLL loss
    optimizer = optim.Adam(model.parameters(), lr=LR)

    print(f"\nTraining for {EPOCHS} epochs (batch_size={BATCH_SIZE}):")
    print(f"{'Epoch':>6} | {'Train Loss':>11} | {'Train Acc':>10} | {'Val Loss':>9} | {'Val Acc':>8}")
    print("-" * 60)

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc     = run_epoch(model, val_loader,   criterion)
        print(f"{epoch:>6} | {train_loss:>11.4f} | {train_acc:>9.4f} | {val_loss:>9.4f} | {val_acc:>8.4f}")

    # Final test evaluation
    true_labels, pred_labels = collect_predictions(model, test_loader)
    test_acc = np.mean(true_labels == pred_labels)

    print(f"\nFinal test accuracy: {test_acc:.4f}")

    print("\nClassification Report:")
    print(classification_report(true_labels, pred_labels,
                                 target_names=[str(i) for i in range(10)]))

    print("Confusion Matrix (rows=true, cols=predicted):")
    cm = confusion_matrix(true_labels, pred_labels)
    # Pretty-print with digit labels
    header = "    " + "  ".join(f"{i:3d}" for i in range(10))
    print(header)
    for i, row in enumerate(cm):
        print(f"{i:3d} " + "  ".join(f"{v:3d}" for v in row))


if __name__ == "__main__":
    main()
