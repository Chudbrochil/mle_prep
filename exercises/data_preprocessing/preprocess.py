import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize_batch(images):
    """
    Normalize a batch of images using ImageNet channel statistics.
    Expects images of shape (B, C, H, W) with values in [0, 1].
    """
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    return (images - mean) / std


def convert_labels(labels, num_classes=10):
    """Return labels in the format expected by nn.CrossEntropyLoss."""
    return F.one_hot(labels.long(), num_classes=num_classes)


def augment(images):
    """Randomly flip images horizontally."""
    if torch.rand(1).item() > 0.5:
        images = torch.flip(images, dims=[0])
    return images


def main():
    torch.manual_seed(0)
    images = torch.rand(8, 3, 32, 32)
    labels = torch.randint(0, 10, (8,))

    images = augment(images)
    images = normalize_batch(images)
    labels = convert_labels(labels)

    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(3 * 32 * 32, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    outputs = model(images)
    loss = criterion(outputs, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Loss: {loss.item():.4f}")
    print("Preprocessing pipeline complete.")


if __name__ == "__main__":
    main()
