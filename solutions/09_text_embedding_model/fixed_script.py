import torch
import torch.nn as nn


VOCAB = {
    '<PAD>': 0,
    '<UNK>': 1,
    'the':   2, 'a':       3, 'is':      4, 'this':    5,
    'good':  6, 'bad':     7, 'great':   8, 'terrible':9,
    'love':  10,'hate':    11,'movie':   12,'film':     13,
    'was':   14,'not':     15,'very':    16,'really':   17,
    'amazing':18,'awful':  19,
}


class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128, num_classes=3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)  # Fix 1: full vocab size
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)
        return self.classifier(hidden[-1])


def tokenize(text, max_len=20):
    tokens = text.lower().split()
    indices = []
    for t in tokens:
        if t in VOCAB:
            indices.append(VOCAB[t])
        else:
            indices.append(VOCAB['<UNK>'])  # Fix 2: use the designated UNK token index
    indices = indices[:max_len]
    indices += [VOCAB['<PAD>']] * (max_len - len(indices))
    return indices


def main():
    texts = [
        "this film was really amazing",
        "i dislike bad movies",
        "the movie is not good",
    ]
    labels = [1, 0, 2]

    sequences = [tokenize(t) for t in texts]
    batch = torch.tensor(sequences)
    targets = torch.tensor(labels)

    vocab_size = len(VOCAB)
    model = TextClassifier(vocab_size=vocab_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    outputs = model(batch)
    loss = criterion(outputs, targets)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Loss: {loss.item():.4f}")
    print("Training step complete.")


if __name__ == "__main__":
    main()
