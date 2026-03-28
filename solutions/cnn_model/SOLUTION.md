# Solution: CNN Model Architecture

## Bugs and Fixes

### Bug 1 — Wrong padding for 5×5 kernel
A 5×5 conv with `padding=1` on a 32×32 input produces output size `(32 + 2×1 - 5) + 1 = 30`.
After 3 MaxPool(2,2) layers the spatial dims cascade: 30→15→7→3.
The FC layer expects input `128×4×4 = 2048` but receives `128×3×3 = 1152`.

**Error:** `RuntimeError: mat1 and mat2 shapes cannot be multiplied (8x1152 vs 2048x256)`

**Fix:** Use `padding=2` for a 5×5 kernel to maintain spatial size: `(32 + 2×2 - 5) + 1 = 32`.
After 3 MaxPool(2,2): 32→16→8→4. FC input is `128×4×4 = 2048`. ✓

**Rule of thumb:** For a kernel of size `k`, use `padding = k // 2` to preserve spatial dims.

---

### Bug 2 — Softmax before CrossEntropyLoss
`torch.softmax(self.fc3(x), dim=1)` applies softmax to the final layer outputs.
`nn.CrossEntropyLoss` internally applies `log_softmax`, so you end up with
`log(softmax(logits))` instead of `log_softmax(logits)` — the loss is computed on
already-normalized probabilities, producing wrong gradients and unstable training.

**No crash** — the script runs but the model trains poorly.

**Fix:** Return raw logits: `return self.fc3(x)`. Let `CrossEntropyLoss` handle the softmax.

---

### Bug 3 — Hardcoded batch size in `view()`
`x.view(32, -1)` assumes a batch size of exactly 32. With 8 samples,
the total elements (8×32×8×8 = 16384) can't reshape to (32, 512) in a way that
matches the FC layer's expected input of 2048.

**Error:** `RuntimeError: shape '[32, -1]' is invalid for input of size 16384`
(or a mat-multiply shape error on the next line)

**Fix:** `x.view(x.size(0), -1)` — always use the actual batch size.
