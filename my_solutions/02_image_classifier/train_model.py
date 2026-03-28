import torch
import torch.nn as nn
import torch.nn.functional as F

# Error#1: RuntimeError: shape '[784]' is invalid for input of size 6272
# Error#2: RuntimeError: 0D or 1D target tensor expected, multi-target not supported

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        print(x.shape)
        x = x.view(x.size(0), 28 * 28) # Bug#1: We weren't multiplying by batch size.
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
    print(labels)

    loss = criterion(outputs, labels) # Bug#2: We didn't need to unsqueeze the labels. Just use them.

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Loss: {loss.item():.4f}")
    print("Training step complete.")


if __name__ == "__main__":
    main()
