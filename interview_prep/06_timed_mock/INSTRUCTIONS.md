# 30-Minute Mock Interview: End-to-End ML

## Rules

- **Do not open `answer_key.py` until your 30 minutes are up.**
- Work in a new Python file (e.g. `my_solution.py`) in this directory.
- You may use any library in the standard ML stack (pandas, numpy, sklearn, etc.).
- Treat this as a real interview: think out loud (comments), justify choices.

---

## Start your timer now.

---

## Dataset

`mock_dataset.csv` — 300 rows, binary classification target: `purchased` (1 = yes, 0 = no).

| Column | Type | Notes |
|---|---|---|
| `age` | int | Some missing |
| `income` | float | Some missing |
| `job_category` | string | tech / finance / healthcare / retail / other — some missing |
| `years_experience` | float | Complete |
| `num_dependents` | int | Complete |
| `has_loan` | binary 0/1 | Complete |
| `credit_score` | float | Some missing |
| `irrelevant_col_1` | float | Random noise |
| `irrelevant_col_2` | float | Random noise |
| `purchased` | binary 0/1 | **Target** |

---

## Checklist

Work through these steps in order. Move quickly — 30 minutes goes fast.

### 1. Exploratory Data Analysis (EDA)
- [ ] Print shape, dtypes, and head
- [ ] Check missing value counts and percentages per column
- [ ] Look at the target distribution (is it balanced?)
- [ ] Compute correlations between numeric features and the target
- [ ] Check purchase rate broken down by `job_category`
- [ ] Note any obvious outliers or skewed distributions

### 2. Feature Engineering
- [ ] Decide how to handle missing values (justify median vs. mean vs. mode)
- [ ] Encode `job_category` (consider: OneHot vs. ordinal vs. target encoding — pick one and justify)
- [ ] Consider at least one interaction feature or derived feature (e.g., income relative to experience)
- [ ] Identify and explicitly drop features that appear to be noise — explain why
- [ ] Think about whether any numeric features should be log-transformed (skewness)

### 3. Preprocessing Pipeline
- [ ] Build a sklearn `Pipeline` (or `ColumnTransformer`) so no preprocessing leaks into the test set
- [ ] Fit all transformers on training data only
- [ ] Use `StratifiedKFold` when splitting or doing cross-validation (class imbalance matters)

### 4. Model Selection — try at least 2
- [ ] Logistic Regression (interpretable baseline)
- [ ] A tree-based model (RandomForest or GradientBoosting)
- [ ] Use cross-validation to compare, not just a single train/test split
- [ ] Report CV scores before picking a winner

### 5. Evaluation — not just accuracy
- [ ] Accuracy
- [ ] Precision, Recall, F1
- [ ] ROC-AUC
- [ ] Classification report on the held-out test set
- [ ] (Bonus) Feature importance from the best model

---

## When Time Is Up

Open `answer_key.py` and compare your approach to the solution. Pay attention to:

- Did you handle missingness the same way? Why might the choices differ?
- Did you catch that `irrelevant_col_1` and `irrelevant_col_2` are noise?
- Did you report enough evaluation metrics for an imbalanced dataset?
- Did you create any interaction features?

Read the debrief section at the bottom of `answer_key.py` for the scoring rubric.
