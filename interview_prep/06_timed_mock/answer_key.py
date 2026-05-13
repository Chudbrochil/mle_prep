"""
STOP — open this only after your 30 minutes are up.
====================================================
This file contains:
  1. Script to generate (and save) mock_dataset.csv reproducibly.
  2. A full end-to-end solution with EDA, feature engineering,
     preprocessing pipeline, model comparison, and evaluation.
  3. A debrief section with a scoring rubric.
"""

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, f1_score, roc_auc_score,
                              accuracy_score, precision_score, recall_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

np.random.seed(42)

CSV_PATH = Path(__file__).parent / "mock_dataset.csv"


# ===========================================================================
# SECTION 1 — Generate and save mock_dataset.csv
# ===========================================================================

def generate_dataset(n=300, seed=42):
    """
    Generate a realistic tabular dataset with meaningful correlations
    between features and the binary target 'purchased'.

    Correlations built in:
      - Higher income  -> more likely to purchase
      - Higher credit_score -> more likely to purchase
      - job_category 'tech' or 'finance' -> more likely to purchase
      - 'retail' and 'other' -> less likely to purchase
    """
    rng = np.random.default_rng(seed)

    # --- Age: integer 22–65, ~8% missing ---
    age = rng.integers(22, 65, size=n).astype(float)
    age[rng.random(n) < 0.08] = np.nan

    # --- Income: right-skewed, ~10% missing ---
    income = rng.lognormal(mean=10.8, sigma=0.6, size=n)  # ~$50k median
    income[rng.random(n) < 0.10] = np.nan

    # --- job_category: 5 categories, ~7% missing ---
    job_probs = [0.30, 0.20, 0.20, 0.20, 0.10]
    job_cats = ["tech", "finance", "healthcare", "retail", "other"]
    job_category = rng.choice(job_cats, size=n, p=job_probs).astype(object)
    job_category[rng.random(n) < 0.07] = None

    # --- Years experience: 0–40, correlated with age ---
    age_filled = np.where(np.isnan(age), rng.integers(30, 45, size=n), age)
    years_experience = np.clip(
        (age_filled - 22) * rng.uniform(0.5, 1.0, size=n) + rng.normal(0, 2, n),
        0, 40
    )

    # --- num_dependents: 0–5, integer ---
    num_dependents = rng.integers(0, 6, size=n)

    # --- has_loan: binary ---
    has_loan = rng.binomial(1, 0.45, size=n)

    # --- credit_score: 300–850, ~12% missing ---
    credit_score = np.clip(rng.normal(680, 80, size=n), 300, 850)
    credit_score[rng.random(n) < 0.12] = np.nan

    # --- Irrelevant noise columns ---
    irrelevant_col_1 = rng.normal(0, 1, size=n)
    irrelevant_col_2 = rng.uniform(0, 100, size=n)

    # --- Build target 'purchased' with meaningful correlations ---
    # Compute a logit score from signal features
    income_filled = np.where(np.isnan(income), np.nanmedian(income), income)
    credit_filled = np.where(np.isnan(credit_score), np.nanmedian(credit_score), credit_score)

    job_score = np.zeros(n)
    for i, j in enumerate(job_category):
        if j == "tech":       job_score[i] =  1.0
        elif j == "finance":  job_score[i] =  0.5
        elif j == "healthcare": job_score[i] = 0.0
        elif j == "retail":   job_score[i] = -0.5
        elif j == "other":    job_score[i] = -0.5
        # None -> 0 (neutral)

    # Normalise income and credit to comparable scale
    income_norm = (income_filled - income_filled.mean()) / income_filled.std()
    credit_norm = (credit_filled - credit_filled.mean()) / credit_filled.std()

    logit = (0.8 * income_norm
             + 0.7 * credit_norm
             + 0.6 * job_score
             - 0.3 * has_loan
             + rng.normal(0, 0.5, size=n)
             - 0.5)   # intercept to get ~40% positive rate

    prob = 1 / (1 + np.exp(-logit))
    purchased = rng.binomial(1, prob)

    df = pd.DataFrame({
        "age":              age,
        "income":           income,
        "job_category":     job_category,
        "years_experience": years_experience,
        "num_dependents":   num_dependents,
        "has_loan":         has_loan,
        "credit_score":     credit_score,
        "irrelevant_col_1": irrelevant_col_1,
        "irrelevant_col_2": irrelevant_col_2,
        "purchased":        purchased,
    })

    return df


def save_dataset():
    df = generate_dataset(n=300, seed=42)
    df.to_csv(CSV_PATH, index=False)
    print(f"Saved dataset to {CSV_PATH}")
    print(f"  Shape: {df.shape}")
    print(f"  Target distribution: {df['purchased'].value_counts(normalize=True).round(3).to_dict()}")
    return df


# ===========================================================================
# SECTION 2 — Full solution
# ===========================================================================

def run_eda(df):
    """Exploratory data analysis: distributions, missingness, correlations."""
    print("\n" + "=" * 60)
    print("EDA")
    print("=" * 60)

    print("\n--- Shape and dtypes ---")
    print(df.dtypes)

    print("\n--- Missing values ---")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(1)
    print(pd.DataFrame({"count": missing, "pct": missing_pct})[missing > 0])

    print("\n--- Target distribution ---")
    print(df["purchased"].value_counts(normalize=True).round(3))

    print("\n--- Numeric summaries ---")
    print(df.describe().round(2))

    print("\n--- Correlation with target (numeric features) ---")
    numeric_cols = ["age", "income", "years_experience",
                    "num_dependents", "credit_score",
                    "irrelevant_col_1", "irrelevant_col_2"]
    corrs = df[numeric_cols + ["purchased"]].corr()["purchased"].drop("purchased")
    print(corrs.sort_values(ascending=False).round(3))

    print("\n--- Purchase rate by job_category ---")
    print(df.groupby("job_category")["purchased"].mean().sort_values(ascending=False).round(3))


def feature_engineering(df):
    """
    Feature engineering steps:
      1. income_per_year: income / max(years_experience, 1) — interaction feature
      2. senior_employee: years_experience > 10
      3. job_tier: encode job prestige ordinally (tech=2, finance=2, healthcare=1, retail=0, other=0)
    Drop irrelevant noise columns.
    """
    df = df.copy()

    # Interaction: income relative to experience (use median fill for missing)
    income_filled = df["income"].fillna(df["income"].median())
    exp_safe = df["years_experience"].clip(lower=1)
    df["income_per_year"] = income_filled / exp_safe

    # Binary feature
    df["senior_employee"] = (df["years_experience"] > 10).astype(int)

    # Ordinal job tier
    tier_map = {"tech": 2, "finance": 2, "healthcare": 1, "retail": 0, "other": 0}
    df["job_tier"] = df["job_category"].map(tier_map)  # NaN if missing

    # Drop noise columns — a strong candidate should justify dropping them
    # after confirming near-zero correlation with target
    df = df.drop(columns=["irrelevant_col_1", "irrelevant_col_2"])

    return df


def build_pipeline(numeric_cols, categorical_cols):
    """Full sklearn Pipeline with ColumnTransformer."""
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocessor = ColumnTransformer([
        ("num", num_pipe, numeric_cols),
        ("cat", cat_pipe, categorical_cols),
    ])
    return preprocessor


def compare_models(X_train, y_train, preprocessor):
    """Compare LR and GBM via stratified 5-fold CV (ROC-AUC)."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    models = {
        "LogisticRegression":   LogisticRegression(max_iter=1000, random_state=42),
        "GradientBoosting":     GradientBoostingClassifier(n_estimators=100, random_state=42),
        "RandomForest":         RandomForestClassifier(n_estimators=100, random_state=42),
    }

    print("\n--- Model comparison (5-fold CV, ROC-AUC) ---")
    print(f"{'Model':<25} {'Mean AUC':>10} {'Std':>8}")
    print("-" * 45)

    results = {}
    for name, clf in models.items():
        pipe = Pipeline([("prep", preprocessor), ("clf", clf)])
        scores = cross_val_score(pipe, X_train, y_train,
                                 cv=cv, scoring="roc_auc", n_jobs=-1)
        results[name] = (scores.mean(), scores.std(), clf)
        print(f"{name:<25} {scores.mean():>10.4f} {scores.std():>8.4f}")

    best_name = max(results, key=lambda k: results[k][0])
    print(f"\nBest model: {best_name}")
    return results, best_name


def evaluate_on_test(model, X_test, y_test):
    """Full evaluation on test set."""
    preds = model.predict(X_test)
    probas = model.predict_proba(X_test)[:, 1]

    print("\n--- Test set evaluation ---")
    print(f"  Accuracy  : {accuracy_score(y_test, preds):.4f}")
    print(f"  Precision : {precision_score(y_test, preds):.4f}")
    print(f"  Recall    : {recall_score(y_test, preds):.4f}")
    print(f"  F1        : {f1_score(y_test, preds):.4f}")
    print(f"  ROC-AUC   : {roc_auc_score(y_test, probas):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, preds))


def feature_importance(model, feature_names):
    """Print feature importances from the best model if available."""
    clf = model.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
        # Get feature names after ColumnTransformer
        prep = model.named_steps["prep"]
        try:
            transformed_names = prep.get_feature_names_out()
        except Exception:
            transformed_names = [f"f{i}" for i in range(len(importances))]
        idx = np.argsort(importances)[::-1]
        print("\n--- Feature importances (top 10) ---")
        for i in idx[:10]:
            print(f"  {transformed_names[i]:<35} {importances[i]:.4f}")


def full_solution():
    print("\n" + "=" * 60)
    print("FULL SOLUTION")
    print("=" * 60)

    # Load dataset
    df = pd.read_csv(CSV_PATH)

    # EDA
    run_eda(df)

    # Feature engineering
    df = feature_engineering(df)

    # Define feature sets
    numeric_cols = ["age", "income", "years_experience",
                    "num_dependents", "has_loan", "credit_score",
                    "income_per_year", "senior_employee", "job_tier"]
    categorical_cols = ["job_category"]

    X = df[numeric_cols + categorical_cols]
    y = df["purchased"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    preprocessor = build_pipeline(numeric_cols, categorical_cols)

    # Model comparison
    results, best_name = compare_models(X_train, y_train, preprocessor)

    # Train best model on full training set
    _, _, best_clf = results[best_name]
    best_pipe = Pipeline([("prep", preprocessor), ("clf", best_clf)])
    best_pipe.fit(X_train, y_train)

    # Test evaluation
    evaluate_on_test(best_pipe, X_test, y_test)

    # Feature importance
    feature_importance(best_pipe, numeric_cols + categorical_cols)


# ===========================================================================
# SECTION 3 — Debrief and scoring rubric
# ===========================================================================

def debrief():
    print("\n" + "=" * 60)
    print("DEBRIEF — What a strong submission covers")
    print("=" * 60)

    rubric = [
        ("Handling missingness", [
            "STRONG: Chose median for numeric (robust to outliers), "
            "mode for categorical, justified the choice.",
            "WEAK: Dropped rows with NaN, or used mean without justification.",
        ]),
        ("Encoding strategy", [
            "STRONG: OneHotEncoded nominal job_category; considered ordinal "
            "encoding if category has a natural order; used handle_unknown='ignore'.",
            "WEAK: Label-encoded a nominal variable (implies false ordinal relationship).",
        ]),
        ("Feature selection rationale", [
            "STRONG: Explicitly dropped irrelevant_col_1/2 after checking "
            "near-zero correlation with target, explained reasoning.",
            "WEAK: Included all columns without inspection, or dropped features "
            "without justification.",
        ]),
        ("Interaction features", [
            "STRONG: Created income_per_year (income / experience) capturing "
            "productivity; noted it's a domain-motivated feature.",
            "WEAK: No interaction features attempted.",
        ]),
        ("Evaluation thoroughness", [
            "STRONG: Reported accuracy, precision, recall, F1, ROC-AUC; "
            "noted class imbalance and chose F1/AUC as primary metrics.",
            "WEAK: Reported only accuracy; didn't account for class imbalance.",
        ]),
        ("Model selection", [
            "STRONG: Tried at least 2 models, used stratified CV for comparison, "
            "selected best model based on held-out AUC not training performance.",
            "WEAK: Trained one model on full data and tested on the same data (leakage).",
        ]),
        ("Pipeline hygiene", [
            "STRONG: All preprocessing fit on train set only; used Pipeline to "
            "prevent leakage; used StratifiedKFold for imbalanced data.",
            "WEAK: Scaled or imputed before splitting (target leakage).",
        ]),
    ]

    for category, points in rubric:
        print(f"\n  [{category}]")
        for point in points:
            print(f"    - {point}")

    print("\n" + "-" * 60)
    print("Common mistakes to avoid:")
    mistakes = [
        "Fitting the scaler on the full dataset before train/test split.",
        "Using accuracy as the sole metric on an imbalanced dataset.",
        "Forgetting handle_unknown='ignore' in OneHotEncoder (test set may have unseen categories).",
        "Not checking feature importance — a key part of a real interview answer.",
        "Skipping EDA and going straight to modelling.",
    ]
    for m in mistakes:
        print(f"  * {m}")
    print()


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    # Step 1: Generate and save the CSV
    save_dataset()

    # Step 2: Full solution
    full_solution()

    # Step 3: Debrief
    debrief()
