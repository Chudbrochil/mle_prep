# Solution: Training Pipeline

## Bugs and Fixes

### Bug 1 — Wrong loss function
`nn.MSELoss` is for regression (requires float targets matching the output shape).
For multi-class classification with integer class labels, use `nn.CrossEntropyLoss`.

**Error:** `RuntimeError: expected scalar type Long but found Float`
(MSELoss tries to compute element-wise squares but the target dtype doesn't match)

**Fix:** `criterion = nn.CrossEntropyLoss()`

---

### Bug 2 — Storing loss tensors in list
`epoch_losses.append(loss)` keeps the full computation graph alive for every batch.
At the end of the epoch, `sum(epoch_losses)` returns a tensor, and formatting it
with `:.4f` fails.

**Error:** `TypeError: unsupported format character` (when printing `train_loss:.4f`)

**Fix:** `epoch_losses.append(loss.item())` — Python float, no graph attached.

---

### Bug 3 — `optimizer.zero_grad()` called after `loss.backward()`
Calling `zero_grad` after `backward` clears the gradients that were just computed,
so `optimizer.step()` updates weights using zeroed gradients. Every step is a no-op.

**No crash** — loss stays high and the model never converges.

**Fix:** Move `optimizer.zero_grad()` to *before* the forward pass.

---

### Bug 4 — Missing `model.eval()` and `torch.no_grad()` in `evaluate()`
In train mode, dropout randomly zeros activations, making validation metrics
non-deterministic. Without `torch.no_grad()`, PyTorch builds a computation graph
for every validation batch, wasting memory.

**No crash** — but validation results are noisy and memory usage is higher than needed.

**Fix:**
```python
model.eval()
with torch.no_grad():
    ...
```
