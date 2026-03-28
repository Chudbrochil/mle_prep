import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Fix 1: preserve batch dimension
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def main():
    batch_size = 8
    images = torch.randn(batch_size, 1, 28, 28)
    labels = torch.randint(0, 10, (batch_size,))

    model = SimpleNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    outputs = model(images)
    loss = criterion(outputs, labels)  # Fix 2: labels must be 1D for CrossEntropyLoss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Loss: {loss.item():.4f}")
    print("Training step complete.")


if __name__ == "__main__":
    main()
