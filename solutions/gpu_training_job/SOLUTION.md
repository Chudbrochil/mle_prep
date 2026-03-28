# Solution: GPU Training Job — Device & Dtype

## Bugs and Fixes

### Bug 1 — Wrong input dtype (float64 vs float32)
`torch.randn(..., dtype=torch.float64)` creates a double-precision tensor.
PyTorch model weights default to float32. You can't mix them in a forward pass.

**Error:** `RuntimeError: expected scalar type Float but found Double`

**Fix:** Remove `dtype=torch.float64` — let it default to `torch.float32`.

---

### Bug 2 — Model and data not moved to device
When CUDA is available, `model`, `images`, and `labels` all stay on CPU.
The forward pass fails because they're on different devices.

**Error (CUDA only):** `RuntimeError: Expected all tensors to be on the same device`

**Fix:**
```python
model = MNISTClassifier().to(device)
images = torch.randn(32, 1, 28, 28).to(device)
labels = torch.randint(0, 10, (32,)).to(device)
```

## Key concept
Both bugs are "wrong device/dtype" errors — the most common class of runtime errors
in PyTorch. Always match dtypes and devices between model and data.
`nn.CrossEntropyLoss` does not need `.to(device)` since it has no parameters.
