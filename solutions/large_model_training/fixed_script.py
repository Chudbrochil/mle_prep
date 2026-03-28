import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split


class LargeModel(nn.Module):
    def __init__(self, input_dim=512, num_classes=50):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def train_epoch(model, loader, criterion, optimizer):
    model.train()
    batch_losses = []

    for X, y in loader:
        optimizer.zero_grad()           # Fix 2: zero_grad BEFORE forward pass
        outputs = model(X)              # Fix 1: remove .detach() — grad must flow through outputs
        loss = criterion(outputs, y)
        batch_losses.append(loss.item())

        loss.backward()
        optimizer.step()

    return sum(batch_losses) / len(batch_losses)


def validate(model, loader, criterion):
    model.eval()                        # Fix 3: eval mode
    correct = 0
    total_loss = 0.0

    with torch.no_grad():               # Fix 3: no gradient tracking during validation
        for X, y in loader:
            outputs = model(X)
            total_loss += criterion(outputs, y).item()
            correct += (outputs.argmax(1) == y).sum().item()

    return total_loss / len(loader), correct / len(loader.dataset)


def main():
    torch.manual_seed(0)

    X = torch.randn(1000, 512)
    y = torch.randint(0, 50, (1000,))
    dataset = TensorDataset(X, y)
    train_set, val_set = random_split(dataset, [800, 200])

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=64)

    model = LargeModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(5):
        train_loss = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = validate(model, val_loader, criterion)
        print(f"Epoch {epoch + 1}: train={train_loss:.4f}, val={val_loss:.4f}, acc={val_acc:.4f}")

    print("Training complete.")


if __name__ == "__main__":
    main()
