import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class ImageDataset(Dataset):
    def __init__(self, num_samples=100):
        np.random.seed(42)
        self.images = np.random.randn(num_samples, 3, 32, 32).astype(np.float32)
        self.labels = np.random.randint(0, 10, num_samples)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


def collate_fn(batch):
    images, labels = zip(*batch)
    return torch.stack(images), torch.tensor(labels)


def main():
    dataset = ImageDataset(num_samples=50)
    print(f"Dataset length: {len(dataset)}")

    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)

    for batch_idx, (images, labels) in enumerate(dataloader):
        print(f"Batch {batch_idx}: images={images.shape}, labels={labels.shape}")
        if batch_idx >= 2:
            break

    print("DataLoader test complete.")


if __name__ == "__main__":
    main()
