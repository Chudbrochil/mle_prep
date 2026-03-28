# Solution: Image Classifier — Tensor Shapes

## Bugs and Fixes

### Bug 1 — View ignores batch dimension
`x.view(28 * 28)` tries to flatten the entire batch into a single vector of 784 elements,
but the input has shape `(batch_size, 1, 28, 28)` = 6272 total elements (for batch_size=8).

**Error:** `RuntimeError: shape '[784]' is invalid for input of size 6272`

**Fix:** `x.view(x.size(0), -1)` — keep batch dimension, flatten the rest.

---

### Bug 2 — Labels have wrong shape for CrossEntropyLoss
`labels.unsqueeze(1)` turns labels from `(8,)` into `(8, 1)`. `nn.CrossEntropyLoss` expects
a 1D target tensor of class indices, not a 2D tensor.

**Error:** `RuntimeError: 1D target tensor expected, multi-target not supported`

**Fix:** Remove `.unsqueeze(1)` — pass `labels` directly.

## Key concept
`nn.CrossEntropyLoss` takes:
- `input`: `(N, C)` — raw logits for each class
- `target`: `(N,)` — integer class index per sample
