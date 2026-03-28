# Solution: User Analytics Pipeline

## Bugs and Fixes

### Bug 1 — Wrong filename
`load_training_data()` opens `train_data.json` but the file is named `training_data.json`.

**Error:** `FileNotFoundError: [Errno 2] No such file or directory: 'train_data.json'`

**Fix:** Change to `open('training_data.json', 'r')`.

---

### Bug 2 — Missing key in validation data
`extract_features_and_labels()` always accesses `sample['approved']`, but one record in
`validation_data.json` uses `'status'` instead of `'approved'`.

**Error:** `KeyError: 'approved'`

**Fix:** Use `.get()` with a fallback:
```python
approved = sample.get('approved', sample.get('status') == 'approved')
```

---

### Bug 3 — Wrong numpy dtype
`np.array(all_features, dtype=str)` creates a string array. Calling `np.mean()` on it fails.

**Error:** `TypeError: ufunc 'add' did not contain a loop with signature matching types (dtype('<U32'), dtype('<U32')) -> None`

**Fix:** Change to `dtype=float`.

## Debugging approach
Run the script, read the first traceback, fix, repeat. Each bug surfaces only after the previous one is fixed — a classic cascading error pattern common in data loading code.
