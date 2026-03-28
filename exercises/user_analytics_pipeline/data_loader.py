import json
import numpy as np


def load_training_data():
    with open('train_data.json', 'r') as f:
        return json.load(f)


def load_validation_data():
    with open('validation_data.json', 'r') as f:
        return json.load(f)


def extract_features_and_labels(data):
    features = []
    labels = []
    for sample in data:
        features.append([sample['age'], sample['income']])
        labels.append(1 if sample['approved'] else 0)
    return features, labels


def main():
    train_data = load_training_data()
    val_data = load_validation_data()

    train_features, train_labels = extract_features_and_labels(train_data)
    val_features, val_labels = extract_features_and_labels(val_data)

    all_features = train_features + val_features
    all_labels = train_labels + val_labels

    features = np.array(all_features, dtype=str)
    labels = np.array(all_labels)

    print(f"Loaded {len(features)} samples")
    print(f"Age:    mean={np.mean(features[:, 0]):.1f}, std={np.std(features[:, 0]):.1f}")
    print(f"Income: mean={np.mean(features[:, 1]):,.0f}")
    print(f"Approval rate: {np.mean(labels) * 100:.1f}%")


if __name__ == "__main__":
    main()
