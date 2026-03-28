# Dataset Splitting

A logistic regression pipeline that loads employee data from CSV, splits it into train/test sets, trains a model, and reports accuracy.

## Expected Output
When working correctly:
1. Load 60 employee records from `employees.csv`
2. Split into train/test sets (70/30)
3. Train a logistic regression model to predict promotions
4. Report ~50% test accuracy or better (this is a simple dataset)

## Files
- `split_dataset.py` - Main pipeline script
- `employees.csv` - Employee dataset with features: age, salary, years_exp, department, promoted

## Usage
```bash
python split_dataset.py
```
