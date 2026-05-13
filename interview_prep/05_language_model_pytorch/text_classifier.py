"""
Exercise 5: Text Classification with PyTorch (Sentiment)
=========================================================
Bag-of-embeddings classifier for binary sentiment analysis.
Uses a synthesized IMDB-style dataset (~2000 samples) so the file
runs without any external dataset download dependency.

Model: Embedding -> mean pooling -> Linear -> ReLU -> Linear
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.metrics import classification_report

torch.manual_seed(42)
np.random.seed(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
MAX_LEN = 30       # pad/truncate every sequence to this many tokens
EMBED_DIM = 64
HIDDEN_DIM = 32
BATCH_SIZE = 32
EPOCHS = 5
LR = 1e-3
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15


# ---------------------------------------------------------------------------
# Synthetic dataset
# ---------------------------------------------------------------------------

POSITIVE_TEMPLATES = [
    "this movie was absolutely {adj} and {adj}",
    "i {verb} this film it was {adj}",
    "what a {adj} experience truly {adj} storytelling",
    "the acting was {adj} and the plot was {adj}",
    "a {adj} film that left me feeling {adj}",
    "i would {verb} this to everyone it is {adj}",
    "one of the most {adj} films i have ever seen",
    "brilliantly directed and {adj} performances throughout",
    "the {adj} cinematography made this film {adj}",
    "an {adj} and {adj} masterpiece of cinema",
]

NEGATIVE_TEMPLATES = [
    "this movie was absolutely {adj} and {adj}",
    "i {verb} this film it was {adj}",
    "what a {adj} waste of time truly {adj} storytelling",
    "the acting was {adj} and the plot was {adj}",
    "a {adj} film that left me feeling {adj}",
    "i would never {verb} this to anyone it is {adj}",
    "one of the most {adj} films i have ever seen",
    "poorly directed with {adj} performances throughout",
    "the {adj} cinematography made this film {adj}",
    "a {adj} and {adj} disaster of a movie",
]

POS_ADJ = ["amazing", "wonderful", "fantastic", "brilliant", "superb",
           "excellent", "outstanding", "incredible", "moving", "captivating"]
NEG_ADJ = ["terrible", "awful", "horrible", "dreadful", "boring",
           "disappointing", "mediocre", "painful", "dull", "forgettable"]
POS_VERB = ["loved", "enjoyed", "adored", "appreciated", "recommended"]
NEG_VERB = ["hated", "despised", "regretted", "disliked", "avoided"]


def generate_sentence(template, adj_list, verb_list, rng):
    """Fill a template with random words from the given word lists."""
    sentence = template
    while "{adj}" in sentence:
        sentence = sentence.replace("{adj}", rng.choice(adj_list), 1)
    while "{verb}" in sentence:
        sentence = sentence.replace("{verb}", rng.choice(verb_list), 1)
    return sentence


def build_synthetic_dataset(n=2000, seed=42):
    """
    Generate n synthetic sentiment sentences (~50/50 split).
    Returns list of (text, label) tuples where label in {0, 1}.
    """
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n // 2):
        tmpl = rng.choice(POSITIVE_TEMPLATES)
        text = generate_sentence(tmpl, POS_ADJ, POS_VERB, rng)
        samples.append((text, 1))
    for _ in range(n // 2):
        tmpl = rng.choice(NEGATIVE_TEMPLATES)
        text = generate_sentence(tmpl, NEG_ADJ, NEG_VERB, rng)
        samples.append((text, 0))
    rng.shuffle(samples)
    return samples


# ---------------------------------------------------------------------------
# Tokenisation and vocabulary
# ---------------------------------------------------------------------------

def whitespace_tokenize(text):
    """Simple whitespace tokenizer — split on spaces."""
    return text.lower().split()


def build_vocab(texts, min_freq=1):
    """
    Build a word -> index mapping from training texts.
    Index 0 is reserved for <PAD>, index 1 for <UNK>.
    """
    from collections import Counter
    counter = Counter()
    for text in texts:
        counter.update(whitespace_tokenize(text))

    vocab = {"<PAD>": 0, "<UNK>": 1}
    for word, freq in counter.items():
        if freq >= min_freq:
            vocab[word] = len(vocab)
    return vocab


def encode(text, vocab, max_len):
    """
    Convert a raw text string into a fixed-length integer sequence.
    - Tokens not in vocab map to <UNK> (index 1).
    - Sequences shorter than max_len are right-padded with <PAD> (index 0).
    - Sequences longer than max_len are truncated.
    """
    tokens = whitespace_tokenize(text)
    ids = [vocab.get(t, 1) for t in tokens]  # 1 = <UNK>
    # Truncate
    ids = ids[:max_len]
    # Pad
    ids += [0] * (max_len - len(ids))         # 0 = <PAD>
    return ids


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class SentimentDataset(Dataset):
    """
    Wraps a list of (encoded_sequence, label) pairs as a PyTorch Dataset.
    """

    def __init__(self, encoded_pairs):
        self.sequences = torch.tensor([x for x, _ in encoded_pairs], dtype=torch.long)
        self.labels    = torch.tensor([y for _, y in encoded_pairs], dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class BagOfEmbeddingsClassifier(nn.Module):
    """
    Bag-of-embeddings (mean pooling) text classifier.

    Architecture:
      Embedding(vocab_size, embed_dim)
        -> Lookup an embed_dim vector for each token in the sequence.
      Mean pooling over sequence length
        -> Collapse (B, L, embed_dim) to (B, embed_dim).
        -> This loses word order but is fast and often effective.
      Linear(embed_dim, hidden_dim) + ReLU
        -> Non-linear hidden layer.
      Linear(hidden_dim, 2)
        -> Raw logits for positive / negative class.

    Why mean pooling over LSTM?
    Simpler, faster, more explainable under interview pressure,
    and still competitive on short texts with strong word-level signal.
    """

    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes=2, pad_idx=0):
        super().__init__()
        # padding_idx=0 means the PAD token's embedding is always zero
        # and does not receive gradient updates
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x: (B, L) — integer token indices
        embedded = self.embedding(x)         # (B, L, embed_dim)
        pooled   = embedded.mean(dim=1)      # (B, embed_dim) — mean over sequence
        hidden   = self.relu(self.fc1(pooled))  # (B, hidden_dim)
        logits   = self.fc2(hidden)          # (B, 2)
        return logits


# ---------------------------------------------------------------------------
# Training / evaluation loop
# ---------------------------------------------------------------------------

def run_epoch(model, loader, criterion, optimizer=None):
    """One pass over the loader; returns (mean_loss, accuracy)."""
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss, correct, total = 0.0, 0, 0
    ctx = torch.enable_grad() if training else torch.no_grad()

    with ctx:
        for seqs, labels in loader:
            seqs, labels = seqs.to(DEVICE), labels.to(DEVICE)
            logits = model(seqs)
            loss = criterion(logits, labels)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(labels)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += len(labels)

    return total_loss / total, correct / total


def collect_predictions(model, loader):
    model.eval()
    all_true, all_pred = [], []
    with torch.no_grad():
        for seqs, labels in loader:
            seqs = seqs.to(DEVICE)
            preds = model(seqs).argmax(dim=1).cpu().numpy()
            all_pred.extend(preds)
            all_true.extend(labels.numpy())
    return np.array(all_true), np.array(all_pred)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Using device: {DEVICE}")

    # Build raw dataset
    raw_data = build_synthetic_dataset(n=2000, seed=42)
    print(f"Dataset size: {len(raw_data)} samples")

    # Train/val/test split (indices)
    n = len(raw_data)
    n_test = int(n * TEST_SPLIT)
    n_val  = int(n * VAL_SPLIT)
    n_train = n - n_val - n_test

    train_raw = raw_data[:n_train]
    val_raw   = raw_data[n_train:n_train + n_val]
    test_raw  = raw_data[n_train + n_val:]

    # Build vocabulary from training texts only (no leakage)
    train_texts = [text for text, _ in train_raw]
    vocab = build_vocab(train_texts)
    print(f"Vocabulary size: {len(vocab)}")

    # Encode all splits
    def encode_split(split):
        return [(encode(text, vocab, MAX_LEN), label) for text, label in split]

    train_ds = SentimentDataset(encode_split(train_raw))
    val_ds   = SentimentDataset(encode_split(val_raw))
    test_ds  = SentimentDataset(encode_split(test_raw))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

    # Instantiate model
    model = BagOfEmbeddingsClassifier(
        vocab_size=len(vocab),
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    print(f"\nTraining for {EPOCHS} epochs:")
    print(f"{'Epoch':>6} | {'Train Loss':>11} | {'Train Acc':>10} | {'Val Loss':>9} | {'Val Acc':>8}")
    print("-" * 60)

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer)
        va_loss, va_acc = run_epoch(model, val_loader,   criterion)
        print(f"{epoch:>6} | {tr_loss:>11.4f} | {tr_acc:>9.4f} | {va_loss:>9.4f} | {va_acc:>8.4f}")

    # Final test evaluation
    true_labels, pred_labels = collect_predictions(model, test_loader)
    test_acc = np.mean(true_labels == pred_labels)

    print(f"\nFinal test accuracy: {test_acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(true_labels, pred_labels,
                                 target_names=["negative", "positive"]))


if __name__ == "__main__":
    main()
