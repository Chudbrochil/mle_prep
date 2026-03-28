import torch
import torch.nn as nn
import torch.optim as optim


class CNN(nn.Module):
    """
    CNN for 32x32 RGB images, 10 output classes.
    3 conv+pool blocks: spatial dims go 32 -> 16 -> 8 -> 4.
    """
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.softmax(self.fc3(x), dim=1)


class SmallCNN(nn.Module):
    """A smaller CNN variant for quick tests."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(32, -1)
        return self.fc2(torch.relu(self.fc1(x)))


def main():
    data = torch.randn(8, 3, 32, 32)
    targets = torch.randint(0, 10, (8,))
    criterion = nn.CrossEntropyLoss()

    print("Testing CNN...")
    model = CNN()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    outputs = model(data)
    loss = criterion(outputs, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"CNN loss: {loss.item():.4f}")

    print("\nTesting SmallCNN...")
    model2 = SmallCNN()
    optimizer2 = optim.Adam(model2.parameters(), lr=0.001)
    outputs2 = model2(data)
    loss2 = criterion(outputs2, targets)
    optimizer2.zero_grad()
    loss2.backward()
    optimizer2.step()
    print(f"SmallCNN loss: {loss2.item():.4f}")

    print("\nAll models working correctly.")


if __name__ == "__main__":
    main()
