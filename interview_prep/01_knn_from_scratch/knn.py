"""
Exercise 1: K-Nearest Neighbors from Scratch
==============================================
Implements KNN using only NumPy. Iris dataset loaded via sklearn (data only).
Compares custom KNN accuracy against sklearn's KNeighborsClassifier.
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier

np.random.seed(42)


# ---------------------------------------------------------------------------
# Distance and voting helpers
# ---------------------------------------------------------------------------

def euclidean_distance(a, b):
    """
    Euclidean distance between two 1-D vectors.
    ||a - b||_2 = sqrt(sum((a_i - b_i)^2))
    Using squared differences avoids sqrt for comparisons, but we compute
    the true distance here for clarity.
    """
    return np.sqrt(np.sum((a - b) ** 2))


def majority_vote(labels):
    """
    Return the most common label in `labels`.
    np.bincount counts occurrences of each non-negative integer;
    argmax returns the index (== label) with the highest count.
    """
    counts = np.bincount(labels.astype(int))
    return np.argmax(counts)


# ---------------------------------------------------------------------------
# KNN Classifier
# ---------------------------------------------------------------------------

class KNNClassifier:
    """K-Nearest Neighbors classifier using Euclidean distance."""

    def fit(self, X, y):
        """
        KNN has no training phase — we just memorise the training set.
        X: (n_samples, n_features)  y: (n_samples,)
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X, k=3):
        """
        For each query point x in X:
          1. Compute distances to all training points.
          2. Select the k nearest neighbours.
          3. Return the majority class label.
        """
        predictions = []
        for x in X:
            # Compute distance from x to every training point
            distances = np.array([euclidean_distance(x, x_train)
                                   for x_train in self.X_train])

            # argsort returns indices that sort distances ascending
            k_nearest_indices = np.argsort(distances)[:k]

            # Gather the labels of the k nearest neighbours
            k_nearest_labels = self.y_train[k_nearest_indices]

            # Majority vote among k neighbours
            predictions.append(majority_vote(k_nearest_labels))

        return np.array(predictions)


# ---------------------------------------------------------------------------
# K selection via k-fold cross-validation
# ---------------------------------------------------------------------------

def kfold_split(n_samples, n_folds):
    """
    Yield (train_indices, val_indices) for each fold.
    Pure NumPy implementation of k-fold splitting.
    """
    indices = np.arange(n_samples)
    fold_sizes = np.full(n_folds, n_samples // n_folds)
    # Distribute the remainder across the first folds
    fold_sizes[: n_samples % n_folds] += 1

    current = 0
    folds = []
    for size in fold_sizes:
        folds.append(indices[current: current + size])
        current += size

    for fold_idx in range(n_folds):
        val_idx = folds[fold_idx]
        train_idx = np.concatenate([folds[i] for i in range(n_folds)
                                    if i != fold_idx])
        yield train_idx, val_idx


def select_k(X, y, k_values=None, n_folds=5):
    """
    Run k-fold CV for each candidate k value and return the k with
    the highest mean validation accuracy.
    """
    if k_values is None:
        k_values = [1, 3, 5, 7, 9, 11]

    # Shuffle before splitting to avoid class ordering artefacts
    shuffled = np.random.permutation(len(y))
    X, y = X[shuffled], y[shuffled]

    best_k, best_score = None, -np.inf

    print(f"\nK selection via {n_folds}-fold cross-validation:")
    print(f"{'k':>4}  {'mean_acc':>10}  {'std_acc':>10}")
    print("-" * 30)

    for k in k_values:
        fold_accuracies = []
        clf = KNNClassifier()

        for train_idx, val_idx in kfold_split(len(y), n_folds):
            clf.fit(X[train_idx], y[train_idx])
            preds = clf.predict(X[val_idx], k=k)
            acc = np.mean(preds == y[val_idx])
            fold_accuracies.append(acc)

        mean_acc = np.mean(fold_accuracies)
        std_acc = np.std(fold_accuracies)
        print(f"{k:>4}  {mean_acc:>10.4f}  {std_acc:>10.4f}")

        if mean_acc > best_score:
            best_score = mean_acc
            best_k = k

    print(f"\nBest k = {best_k}  (mean CV accuracy = {best_score:.4f})")
    return best_k


# ---------------------------------------------------------------------------
# Train/test split (NumPy only)
# ---------------------------------------------------------------------------

def train_test_split_numpy(X, y, test_size=0.2):
    """Random shuffle then split at (1-test_size)*n_samples."""
    n = len(y)
    indices = np.random.permutation(n)
    split = int(n * (1 - test_size))
    return (X[indices[:split]], X[indices[split:]],
            y[indices[:split]], y[indices[split:]])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Load Iris (sklearn used only for data access)
    iris = load_iris()
    X, y = iris.data, iris.target

    X_train, X_test, y_train, y_test = train_test_split_numpy(X, y, test_size=0.2)

    # Select best k via cross-validation on training set
    best_k = select_k(X_train, y_train,
                      k_values=[1, 3, 5, 7, 9, 11], n_folds=5)

    # Evaluate custom KNN
    clf = KNNClassifier()
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test, k=best_k)
    custom_acc = np.mean(preds == y_test)

    # Evaluate sklearn KNN (same k for fair comparison)
    sk_clf = KNeighborsClassifier(n_neighbors=best_k)
    sk_clf.fit(X_train, y_train)
    sk_preds = sk_clf.predict(X_test)
    sk_acc = np.mean(sk_preds == y_test)

    print("\n" + "=" * 40)
    print(f"Test accuracy  — Custom KNN (k={best_k}): {custom_acc:.4f}")
    print(f"Test accuracy  — sklearn KNN (k={best_k}): {sk_acc:.4f}")
    print("=" * 40)


if __name__ == "__main__":
    main()
