# MLE Interview Prep — Debugging Exercises

Debugging exercises designed to match the Scale AI MLE interview format: a 60-minute
live coding session where you run broken scripts and fix them.

**Rules (mirror the real interview):**
- No AI tools
- Google and Stack Overflow are fine
- Think out loud and explain your reasoning
- Aim for 5–10 minutes per exercise

---

## Exercises

| # | Directory | Focus | Time target |
|---|-----------|-------|-------------|
| 1 | [01_user_analytics_pipeline](exercises/01_user_analytics_pipeline/) | JSON data loading | 5–7 min |
| 2 | [02_image_classifier](exercises/02_image_classifier/) | Tensor shapes | 5–7 min |
| 3 | [03_gpu_training_job](exercises/03_gpu_training_job/) | Dtype & device placement | 6–8 min |
| 4 | [04_custom_dataloader](exercises/04_custom_dataloader/) | PyTorch Dataset/DataLoader | 8–10 min |
| 5 | [05_large_model_training](exercises/05_large_model_training/) | Training loop correctness | 7–9 min |
| 6 | [06_data_preprocessing](exercises/06_data_preprocessing/) | Normalization pipeline | 5–7 min |
| 7 | [07_cnn_model](exercises/07_cnn_model/) | CNN architecture | 8–10 min |
| 8 | [08_training_pipeline](exercises/08_training_pipeline/) | End-to-end training | 8–10 min |
| 9 | [09_text_embedding_model](exercises/09_text_embedding_model/) | Embeddings & vocab | 6–8 min |
| 10 | [10_dataset_splitting](exercises/10_dataset_splitting/) | CSV loading & train/test split | 6–8 min |

---

## How to work through an exercise

```bash
cd exercises/01_user_analytics_pipeline
python data_loader.py        # run it — it will fail
# read the traceback, find the bug, fix it, re-run
# repeat until the script exits cleanly
```

Each exercise is self-contained. The script fails with a real Python traceback.
Fix bugs one at a time by following the error messages.

Check your work after finishing:
```bash
cd ../../evaluation
python self_check.py --exercise 1
python self_check.py --all
```

---

## After you've attempted an exercise

Solutions live in [solutions/](solutions/). Each has:
- `fixed_script.py` — the corrected version
- `SOLUTION.md` — explains each bug and its fix

Only look at these after you've genuinely attempted the exercise.

---

## Debugging tips

1. Read the full traceback — the last line tells you the error type, the lines above show where
2. Check tensor shapes with `print(tensor.shape)` or `breakpoint()`
3. Check dtypes with `tensor.dtype` and devices with `tensor.device`
4. Fix one error at a time — re-run after each fix
5. When stuck: look at what the script is *supposed* to print (described in each README)
