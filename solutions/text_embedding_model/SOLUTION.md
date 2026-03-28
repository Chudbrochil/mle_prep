# Solution: Text Embedding Model

## Bugs and Fixes

### Bug 1 — Embedding table too small
`nn.Embedding(vocab_size - 5, embed_dim)` creates an embedding for only 15 tokens
(indices 0–14) while the vocabulary has 20 tokens (indices 0–19).
Tokens like `'really'` (index 17) or `'amazing'` (index 18) immediately trigger an error.

**Error:** `IndexError: index out of range in self` (from the embedding lookup)

**Fix:** `nn.Embedding(vocab_size, embed_dim)` — the embedding table must cover all valid indices.

---

### Bug 2 — OOV tokens use index `len(VOCAB)` instead of `<UNK>`
For out-of-vocabulary words, `indices.append(len(VOCAB))` appends index 20.
After fixing Bug 1, the embedding has indices 0–19, so index 20 is still out of range.

**Error:** `IndexError: index out of range in self`

**Fix:** `indices.append(VOCAB['<UNK>'])` — use index 1, the designated unknown token.

## Key concept
The embedding table size must equal your vocabulary size exactly. Reserve a special
`<UNK>` token at a known index for any word not in your vocabulary.
