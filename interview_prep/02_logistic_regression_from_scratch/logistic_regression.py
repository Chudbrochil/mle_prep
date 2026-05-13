"""
Exercise 2: Logistic Regression from Scratch
=============================================
Implements binary logistic regression using only NumPy.
Uses the breast cancer dataset (binary labels) from sklearn.
Compares custom implementation against sklearn's LogisticRegression.
"""

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression as SklearnLR
from sklearn.preprocessing import StandardScaler

np.random.seed(42)


# ---------------------------------------------------------------------------
# Core math functions
# ---------------------------------------------------------------------------

def sigmoid(z):
    """
    Sigmoid (logistic) function: sigma(z) = 1 / (1 + exp(-z)).

    Maps any real number to (0, 1), which we interpret as P(y=1 | x).
    Numerically stable: for large negative z, exp(-z) dominates and
    sigma -> 0; for large positive z, exp(-z) -> 0 and sigma -> 1.
    """
    return 1.0 / (1.0 + np.exp(-z))


def binary_cross_entropy(y, y_hat):
    """
    Binary cross-entropy loss (log-loss):
      L = -mean[ y * log(y_hat) + (1 - y) * log(1 - y_hat) ]

    This is the negative log-likelihood under a Bernoulli model.
    We clip y_hat away from 0 and 1 to avoid log(0) = -inf.
    """
    eps = 1e-9  # numerical stability: keep y_hat in (eps, 1-eps)
    y_hat = np.clip(y_hat, eps, 1 - eps)
    return -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))


# ---------------------------------------------------------------------------
# Logistic Regression Classifier
# ---------------------------------------------------------------------------

class LogisticRegression:
    """
    Binary logistic regression trained with batch gradient descent.

    Forward pass:   z = X @ w + b          (linear combination)
                    y_hat = sigmoid(z)      (probability estimate)
    Loss:           L = binary_cross_entropy(y, y_hat)
    Backward pass:  dL/dw = X.T @ (y_hat - y) / n
                    dL/db = mean(y_hat - y)
    Update:         w -= lr * dL/dw
                    b -= lr * dL/db
    """

    def __init__(self):
        self.w = None
        self.b = None
        self.loss_history = []

    def fit(self, X, y, lr=0.1, epochs=1000):
        """
        Gradient descent training loop.

        Parameters
        ----------
        X : ndarray, shape (n, d)
        y : ndarray, shape (n,)  — binary labels {0, 1}
        lr : learning rate (step size for gradient descent)
        epochs : number of full passes over the dataset
        """
        n, d = X.shape

        # Initialise weights to zero (common choice; symmetry breaking
        # isn't needed for logistic regression unlike neural nets)
        self.w = np.zeros(d)
        self.b = 0.0

        for epoch in range(1, epochs + 1):
            # ---- Forward pass ------------------------------------------------
            z = X @ self.w + self.b          # linear logit: (n,)
            y_hat = sigmoid(z)               # predicted probabilities: (n,)

            # ---- Loss --------------------------------------------------------
            loss = binary_cross_entropy(y, y_hat)
            self.loss_history.append(loss)

            # ---- Backward pass -----------------------------------------------
            # dL/d(y_hat): from chain rule through cross-entropy
            # dL/dz = y_hat - y  (analytical result combining sigmoid + BCE)
            # This elegant form comes from:
            #   dL/dy_hat = -(y/y_hat - (1-y)/(1-y_hat))   [BCE gradient]
            #   dy_hat/dz  = y_hat * (1 - y_hat)             [sigmoid derivative]
            #   dL/dz      = dL/dy_hat * dy_hat/dz = y_hat - y
            error = y_hat - y                # dL/dz shape: (n,)

            # dL/dw = X.T @ dL/dz / n
            # Chain rule: loss averages over n samples, so divide by n.
            # X.T @ error sums d(loss_i)/dw for each sample i.
            dw = X.T @ error / n             # shape: (d,)

            # dL/db = mean(dL/dz) — bias gradient is just the mean error
            # because d(z)/db = 1 for every sample.
            db = error.mean()                # scalar

            # ---- Parameter update --------------------------------------------
            # Gradient descent: move opposite to gradient direction
            self.w -= lr * dw
            self.b -= lr * db

            # Print progress every 100 epochs
            if epoch % 100 == 0:
                print(f"  Epoch {epoch:>5} | loss = {loss:.6f}")

    def predict_proba(self, X):
        """Return predicted probability P(y=1 | x) for each row of X."""
        return sigmoid(X @ self.w + self.b)

    def predict(self, X, threshold=0.5):
        """Return binary predictions by thresholding probabilities."""
        return (self.predict_proba(X) >= threshold).astype(int)


# ---------------------------------------------------------------------------
# Train/test split and feature scaling (NumPy only)
# ---------------------------------------------------------------------------

def train_test_split_numpy(X, y, test_size=0.2):
    n = len(y)
    idx = np.random.permutation(n)
    cut = int(n * (1 - test_size))
    return X[idx[:cut]], X[idx[cut:]], y[idx[:cut]], y[idx[cut:]]


def standardize(X_train, X_test):
    """
    Z-score normalisation: x' = (x - mean) / std.
    Parameters estimated on train set only (no data leakage).
    """
    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0) + 1e-8   # avoid division by zero
    return (X_train - mu) / sigma, (X_test - mu) / sigma


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Load breast cancer dataset (binary: malignant=1, benign=0)
    data = load_breast_cancer()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split_numpy(X, y, test_size=0.2)

    # Feature standardisation is critical for gradient descent convergence
    X_train_s, X_test_s = standardize(X_train, X_test)

    print("Training custom Logistic Regression (gradient descent):")
    print("-" * 55)
    clf = LogisticRegression()
    clf.fit(X_train_s, y_train, lr=0.1, epochs=1000)

    custom_preds = clf.predict(X_test_s)
    custom_acc = np.mean(custom_preds == y_test)

    # sklearn comparison (uses L-BFGS by default, more sophisticated optimiser)
    sk_clf = SklearnLR(max_iter=1000, random_state=42)
    # sklearn's own scaler for fair comparison
    scaler = StandardScaler()
    X_train_sk = scaler.fit_transform(X_train)
    X_test_sk = scaler.transform(X_test)
    sk_clf.fit(X_train_sk, y_train)
    sk_preds = sk_clf.predict(X_test_sk)
    sk_acc = np.mean(sk_preds == y_test)

    print("\n" + "=" * 45)
    print(f"Test accuracy — Custom LR  : {custom_acc:.4f}")
    print(f"Test accuracy — sklearn LR : {sk_acc:.4f}")
    print("=" * 45)
    print(f"\nFinal training loss: {clf.loss_history[-1]:.6f}")


if __name__ == "__main__":
    main()
