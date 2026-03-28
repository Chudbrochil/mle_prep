# Solution: Data Preprocessing Pipeline

## Bugs and Fixes

### Bug 1 — Broadcasting mismatch in normalization
`mean` and `std` have shape `(3,)`. When subtracting from `images` of shape `(B, C, H, W)`,
PyTorch aligns from the right: it tries to broadcast `(3,)` over the last dimension (width=32),
which fails because 3 ≠ 32.

**Error:** `RuntimeError: The size of tensor a (32) must match the size of tensor b (3) at non-singleton dimension 3`

**Fix:** Reshape to `(1, 3, 1, 1)` so it broadcasts over the channel dimension:
```python
mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
```

---

### Bug 2 — One-hot labels passed to CrossEntropyLoss
`F.one_hot(labels, num_classes=10)` returns an `int64` tensor of shape `(B, 10)`.
`nn.CrossEntropyLoss` expects either `(B,)` integer class indices or `(B, C)` *float* soft labels.
Passing `(B, C)` int64 triggers a dtype error.

**Error:** `RuntimeError: expected scalar type Float but found Long`

**Fix:** Return `labels.long()` directly — CrossEntropyLoss handles the one-hot internally.

---

### Bug 3 — Horizontal flip on wrong dimension
`torch.flip(images, dims=[0])` reverses the *batch* dimension (shuffles image order).
For a horizontal flip, flip the *width* dimension.

**No crash** — images appear to transform, but pixels are not actually mirrored.

**Fix:** `torch.flip(images, dims=[3])` — dim 3 is width in `(B, C, H, W)`.
