# Solution: Large Model Training Loop

## Bugs and Fixes

### Bug 1 — `.detach()` breaks gradient flow
`model(X).detach()` severs the connection between the model's computation graph and the
loss. When `loss.backward()` is called, there is nothing to differentiate through.

**Error:** `RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn`

**Fix:** Remove `.detach()` — `outputs = model(X)`. The gradient must flow from the loss
back through the outputs and into the model's parameters.

---

### Bug 2 — `optimizer.zero_grad()` called after `loss.backward()`
`zero_grad()` is called *after* `backward()`, which clears the gradients that were just
computed. Then `optimizer.step()` updates weights using all-zero gradients. The model
makes zero progress — you can see this because the loss is identical every epoch.

**No crash** — but the loss never decreases no matter how long you train.

**Fix:** Move `optimizer.zero_grad()` to *before* the forward pass.

---

### Bug 3 — Missing `model.eval()` and `torch.no_grad()` in `validate()`
In train mode, dropout randomly zeros activations during validation, giving inconsistent
results. Without `torch.no_grad()`, PyTorch tracks gradients for every validation batch,
building a computation graph that's never used but wastes memory.

**No crash** — but validation metrics are noisy and memory usage is higher than needed.

**Fix:**
```python
model.eval()
with torch.no_grad():
    ...
```
