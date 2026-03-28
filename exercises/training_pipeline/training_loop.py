import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split


class MLP(nn.Module):
    def __init__(self, input_dim=20, hidden_dim=64, num_classes=5):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


def train_epoch(model, loader, criterion, optimizer):
    model.train()
    epoch_losses = []

    for X, y in loader:
        outputs = model(X)
        loss = criterion(outputs, y)
        epoch_losses.append(loss)

        loss.backward()
        optimizer.zero_grad()
        optimizer.step()

    return sum(epoch_losses) / len(epoch_losses)


def evaluate(model, loader, criterion):
    total_loss = 0.0
    correct = 0
    total = 0

    for X, y in loader:
        outputs = model(X)
        total_loss += criterion(outputs, y).item()
        preds = outputs.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    return total_loss / len(loader), correct / total


def main():
    torch.manual_seed(42)

    X = torch.randn(500, 20)
    y = torch.randint(0, 5, (500,))
    dataset = TensorDataset(X, y)
    train_set, val_set = random_split(dataset, [400, 100])

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32)

    model = MLP()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(5):
        train_loss = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = evaluate(model, val_loader, criterion)
        print(f"Epoch {epoch + 1}: train={train_loss:.4f}, val={val_loss:.4f}, acc={val_acc:.4f}")

    print("Training complete.")


if __name__ == "__main__":
    main()
