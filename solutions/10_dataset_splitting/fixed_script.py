import csv
import random
import torch
import torch.nn as nn


def load_dataset(filepath):
    features = []
    labels = []
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)  # Fix 1: remove delimiter='\t', file is comma-separated
        for row in reader:
            features.append([
                float(row['age']),
                float(row['salary']),
                float(row['years_exp']),
            ])
            labels.append(int(row['promoted']))  # Fix 2: convert to int
    return features, labels


def split_dataset(features, labels, train_ratio=0.7):
    # Fix 3: shuffle before splitting to avoid class imbalance in splits
    combined = list(zip(features, labels))
    random.shuffle(combined)
    features, labels = zip(*combined)

    n = len(features)
    split_idx = int(n * train_ratio)
    train_X = list(features[:split_idx])
    train_y = list(labels[:split_idx])
    test_X = list(features[split_idx:])
    test_y = list(labels[split_idx:])
    return train_X, train_y, test_X, test_y


class LogisticRegression(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 2)

    def forward(self, x):
        return self.linear(x)


def train_and_evaluate(train_X, train_y, test_X, test_y):
    X_train = torch.tensor(train_X, dtype=torch.float32)
    y_train = torch.tensor(train_y)
    X_test = torch.tensor(test_X, dtype=torch.float32)
    y_test = torch.tensor(test_y)

    model = LogisticRegression(input_dim=3)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        preds = model(X_test).argmax(dim=1)
        accuracy = (preds == y_test).float().mean().item()

    print(f"Train size: {len(train_X)}, Test size: {len(test_X)}")
    print(f"Train label distribution: {sum(train_y)} promoted out of {len(train_y)}")
    print(f"Test label distribution:  {sum(test_y)} promoted out of {len(test_y)}")
    print(f"Test accuracy: {accuracy:.4f}")


def main():
    random.seed(42)
    features, labels = load_dataset('employees.csv')
    train_X, train_y, test_X, test_y = split_dataset(features, labels)
    train_and_evaluate(train_X, train_y, test_X, test_y)


if __name__ == "__main__":
    main()
