# Solution: Custom DataLoader

## Bugs and Fixes

### Bug 1 — Missing `__len__` method
`DataLoader` calls `len(dataset)` to determine how many batches to create. Without `__len__`,
this raises a `TypeError`.

**Error:** `TypeError: object of type 'ImageDataset' has no len()`

**Fix:** Add:
```python
def __len__(self):
    return len(self.images)
```

---

### Bug 2 — `__getitem__` returns numpy arrays, not tensors
After fixing `__len__`, the DataLoader iterates and calls `collate_fn`. Inside `collate_fn`,
`torch.stack(images)` expects a list of tensors but receives numpy arrays.

**Error:** `TypeError: expected Tensor as element 0 in argument 0, but got numpy.ndarray`

**Fix:** Convert in `__getitem__`:
```python
image = torch.from_numpy(self.images[idx])
label = int(self.labels[idx])
return image, label
```

## Key concept
PyTorch's `Dataset` contract requires both `__len__` and `__getitem__`, and `__getitem__`
should return tensors (or types your collate function explicitly handles).
