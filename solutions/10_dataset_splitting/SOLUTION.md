# Solution: Dataset Splitting

## Bugs and Fixes

### Bug 1 — Wrong CSV delimiter
`csv.DictReader(f, delimiter='\t')` reads `employees.csv` as if it were tab-separated.
Since the file uses commas, every row is parsed as a single key containing all column names
joined together (e.g., `'age,salary,years_exp,department,promoted'`).
Accessing `row['age']` then raises a `KeyError`.

**Error:** `KeyError: 'age'`

**Fix:** Remove `delimiter='\t'` — `DictReader` defaults to comma, which matches the file.

---

### Bug 2 — Labels loaded as strings
After fixing Bug 1, `row['promoted']` returns the string `'0'` or `'1'`.
`torch.tensor(['0', '1', ...])` cannot infer a numeric dtype from strings.

**Error:** `RuntimeError: could not determine the desired value type` (or similar)

**Fix:** `labels.append(int(row['promoted']))` — convert to integer before building tensors.

---

### Bug 3 — No shuffle before split (data leakage via ordering)
`employees.csv` is sorted: all non-promoted employees first, all promoted employees second.
Without shuffling, the training set (first 70%) contains only class 0, and the test set
contains mostly class 1. The model never sees class 1 examples during training.

**No crash** — but test accuracy will be ~0% because the model learned to always predict 0.

**Fix:** Shuffle `(features, labels)` pairs together before slicing:
```python
combined = list(zip(features, labels))
random.shuffle(combined)
features, labels = zip(*combined)
```
Always shuffle paired data together to avoid reordering label/feature alignment.
