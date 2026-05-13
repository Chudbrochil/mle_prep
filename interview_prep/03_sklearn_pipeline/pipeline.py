"""
Exercise 3: sklearn Pipeline with Messy Tabular Data
=====================================================
Simulates a realistic messy dataset, builds a full sklearn preprocessing
Pipeline + ColumnTransformer, compares three classifiers, tunes the best
with RandomizedSearchCV, and evaluates with multiple metrics.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, f1_score,
                              precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import (RandomizedSearchCV, StratifiedKFold,
                                     cross_val_score, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

np.random.seed(42)


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

def make_messy_dataset(n=600, missing_num=0.15, missing_cat=0.10, pos_rate=0.30):
    """
    Generate a synthetic tabular dataset with:
    - Numeric features + controlled missingness
    - Categorical features + controlled missingness
    - Class imbalance (~70/30)
    """
    n_pos = int(n * pos_rate)
    n_neg = n - n_pos

    # --- Numeric features (signal + noise) ---
    # Positive class has slightly higher feature values
    num_feat_pos = np.random.randn(n_pos, 4) + np.array([1.0, 0.5, -0.5, 0.8])
    num_feat_neg = np.random.randn(n_neg, 4)
    num_feat = np.vstack([num_feat_pos, num_feat_neg])

    # --- Categorical feature ---
    cats = ["alpha", "beta", "gamma", "delta"]
    # Positive class skewed toward 'alpha', negative toward 'delta'
    cat_pos = np.random.choice(["alpha", "beta"], size=n_pos, p=[0.6, 0.4])
    cat_neg = np.random.choice(["gamma", "delta"], size=n_neg, p=[0.5, 0.5])
    cat_feat = np.concatenate([cat_pos, cat_neg])

    # --- Target ---
    y = np.array([1] * n_pos + [0] * n_neg)

    # --- Build DataFrame ---
    df = pd.DataFrame(
        num_feat,
        columns=["feat_num_1", "feat_num_2", "feat_num_3", "feat_num_4"]
    )
    df["feat_cat"] = cat_feat
    df["target"] = y

    # Shuffle rows
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # --- Inject missing values ---
    for col in ["feat_num_1", "feat_num_2", "feat_num_3", "feat_num_4"]:
        mask = np.random.rand(n) < missing_num
        df.loc[mask, col] = np.nan

    mask_cat = np.random.rand(n) < missing_cat
    df.loc[mask_cat, "feat_cat"] = np.nan

    return df


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def build_preprocessor(numeric_cols, categorical_cols):
    """
    ColumnTransformer that applies:
      - Numeric pipeline: median imputation -> StandardScaler
      - Categorical pipeline: most-frequent imputation -> OneHotEncoder
    """
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_cols),
        ("cat", categorical_pipeline, categorical_cols),
    ])

    return preprocessor


# ---------------------------------------------------------------------------
# Model comparison
# ---------------------------------------------------------------------------

def evaluate_models(X_train, y_train, preprocessor):
    """
    Run 5-fold stratified CV for three classifiers and return results table.
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    }

    results = {}
    for name, model in models.items():
        pipe = Pipeline([("prep", preprocessor), ("clf", model)])
        scores = cross_val_score(pipe, X_train, y_train,
                                 cv=cv, scoring="roc_auc", n_jobs=-1)
        results[name] = {
            "mean_auc": scores.mean(),
            "std_auc": scores.std(),
        }

    return results, models


def print_comparison_table(results):
    print("\nModel Comparison (5-fold stratified CV, ROC-AUC):")
    print(f"{'Model':<25} {'Mean AUC':>10} {'Std AUC':>10}")
    print("-" * 48)
    for name, r in sorted(results.items(), key=lambda x: -x[1]["mean_auc"]):
        print(f"{name:<25} {r['mean_auc']:>10.4f} {r['std_auc']:>10.4f}")


# ---------------------------------------------------------------------------
# Hyperparameter tuning
# ---------------------------------------------------------------------------

def tune_best_model(X_train, y_train, preprocessor):
    """
    Tune GradientBoostingClassifier (usually best on tabular data)
    with RandomizedSearchCV.
    """
    param_dist = {
        "clf__n_estimators": [50, 100, 200, 300],
        "clf__learning_rate": [0.01, 0.05, 0.1, 0.2],
        "clf__max_depth": [2, 3, 4, 5],
        "clf__subsample": [0.7, 0.8, 1.0],
        "clf__min_samples_leaf": [1, 3, 5],
    }

    pipe = Pipeline([
        ("prep", preprocessor),
        ("clf", GradientBoostingClassifier(random_state=42)),
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    search = RandomizedSearchCV(
        pipe,
        param_distributions=param_dist,
        n_iter=20,
        scoring="roc_auc",
        cv=cv,
        random_state=42,
        n_jobs=-1,
        verbose=0,
    )
    search.fit(X_train, y_train)
    print(f"\nBest CV AUC (RandomizedSearchCV): {search.best_score_:.4f}")
    print(f"Best params: {search.best_params_}")
    return search.best_estimator_


# ---------------------------------------------------------------------------
# Final evaluation
# ---------------------------------------------------------------------------

def full_evaluation(model, X_test, y_test):
    preds = model.predict(X_test)
    probas = model.predict_proba(X_test)[:, 1]

    print("\n" + "=" * 50)
    print("Final Test-Set Evaluation (best tuned model)")
    print("=" * 50)
    print(f"  Accuracy  : {accuracy_score(y_test, preds):.4f}")
    print(f"  Precision : {precision_score(y_test, preds):.4f}")
    print(f"  Recall    : {recall_score(y_test, preds):.4f}")
    print(f"  F1        : {f1_score(y_test, preds):.4f}")
    print(f"  ROC-AUC   : {roc_auc_score(y_test, probas):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, preds))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    df = make_messy_dataset(n=600)

    print("Dataset shape:", df.shape)
    print(f"Class balance: {df['target'].value_counts(normalize=True).to_dict()}")
    print(f"Missing values:\n{df.isnull().sum()}")

    numeric_cols = ["feat_num_1", "feat_num_2", "feat_num_3", "feat_num_4"]
    categorical_cols = ["feat_cat"]

    X = df[numeric_cols + categorical_cols]
    y = df["target"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    preprocessor = build_preprocessor(numeric_cols, categorical_cols)

    # Step 1: compare models
    print("\nComparing models with cross-validation...")
    results, _ = evaluate_models(X_train, y_train, preprocessor)
    print_comparison_table(results)

    # Step 2: tune best model
    print("\nTuning GradientBoostingClassifier with RandomizedSearchCV...")
    best_model = tune_best_model(X_train, y_train, preprocessor)

    # Step 3: final evaluation on held-out test set
    full_evaluation(best_model, X_test, y_test)


if __name__ == "__main__":
    main()
