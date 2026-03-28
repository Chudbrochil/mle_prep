#!/usr/bin/env python3
"""
Self-evaluation tool for MLE debugging exercises.
Run your fixed script; this checks that it exits cleanly and prints the expected output.

Usage:
  python self_check.py --exercise 1     # check a single exercise
  python self_check.py --all            # check all exercises
  python self_check.py --list           # list exercises
"""

import sys
import subprocess
import argparse
from pathlib import Path


EXERCISES = {
    1:  ("user_analytics_pipeline", "data_loader.py"),
    2:  ("image_classifier",        "train_model.py"),
    3:  ("gpu_training_job",        "train.py"),
    4:  ("custom_dataloader",       "dataset.py"),
    5:  ("large_model_training",    "train_large_model.py"),
    6:  ("data_preprocessing",      "preprocess.py"),
    7:  ("cnn_model",               "model.py"),
    8:  ("training_pipeline",       "training_loop.py"),
    9:  ("text_embedding_model",    "embedding_model.py"),
    10: ("dataset_splitting",       "split_dataset.py"),
}

# Strings that must appear in stdout for the exercise to be considered solved.
SUCCESS_PATTERNS = {
    1:  ["Loaded", "samples", "Approval rate"],
    2:  ["Training step complete"],
    3:  ["Training step complete"],
    4:  ["DataLoader test complete"],
    5:  ["Training complete"],
    6:  ["Preprocessing pipeline complete"],
    7:  ["All models working correctly"],
    8:  ["Training complete"],
    9:  ["Training step complete"],
    10: ["Test accuracy", "Train size"],
}

HINTS = {
    1: {
        "FileNotFoundError": "The filename in the script doesn't match the actual file on disk.",
        "KeyError":          "Not all records share the same keys — check validation_data.json.",
        "TypeError":         "Check the dtype passed to np.array(). Can you do math on strings?",
    },
    2: {
        "size":     "A tensor is the wrong shape for this operation. Print .shape before and after.",
        "view":     "view() must preserve the total number of elements. Include the batch dimension.",
        "target":   "CrossEntropyLoss expects 1D integer targets, not 2D.",
    },
    3: {
        "Double":   "Model weights are float32 by default. Match input dtype.",
        "Float":    "Model weights are float32 by default. Match input dtype.",
        "device":   "Move both the model and data to the same device with .to(device).",
    },
    4: {
        "no len":   "Implement __len__ in your Dataset subclass.",
        "numpy":    "Return tensors from __getitem__, not numpy arrays.",
        "stack":    "torch.stack() requires tensors. Convert in __getitem__ with torch.from_numpy().",
    },
    5: {
        "format":   "You can't format a tensor with :.4f. Use loss.item() to get a Python float.",
        "grad":     "optimizer.zero_grad() must come before loss.backward(), not after.",
    },
    6: {
        "size of tensor": "mean/std shape doesn't broadcast with (B,C,H,W). Reshape to (1,C,1,1).",
        "scalar type":    "CrossEntropyLoss expects integer class indices, not one-hot tensors.",
    },
    7: {
        "mat1 and mat2":  "FC layer input size doesn't match. Recalculate spatial dims after conv+pool.",
        "padding":        "For a 5x5 kernel, padding=2 preserves spatial size. padding=1 shrinks it.",
        "view":           "Don't hardcode batch size in view(). Use x.size(0) instead.",
    },
    8: {
        "scalar type Long": "MSELoss needs float targets. Use CrossEntropyLoss for classification.",
        "format":           "sum(tensor_list) returns a tensor. Use loss.item() when appending.",
        "zero_grad":        "zero_grad() must be called before backward(), not after.",
    },
    9: {
        "index out of range": "Embedding table is smaller than vocab size, or OOV index is too large.",
        "UNK":                "Use VOCAB['<UNK>'] for unknown tokens, not len(VOCAB).",
    },
    10: {
        "KeyError":     "Wrong delimiter? The CSV uses commas, not tabs.",
        "value type":   "CSV values are strings. Convert labels to int before making tensors.",
        "accuracy":     "Check label distribution in train vs test — did you shuffle before splitting?",
    },
}


class ExerciseChecker:
    def __init__(self):
        self.repo_root = Path(__file__).parent.parent
        self.exercises_dir = self.repo_root / "exercises"

    def check(self, num: int) -> bool:
        if num not in EXERCISES:
            print(f"Exercise {num} not found. Valid range: 1–{max(EXERCISES)}.")
            return False

        folder, script = EXERCISES[num]
        exercise_dir = self.exercises_dir / folder
        script_path = exercise_dir / script

        print(f"Exercise {num:02d}: {folder}/{script}")
        print("-" * 50)

        if not script_path.exists():
            print(f"  MISSING: {script_path}")
            return False

        try:
            result = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=exercise_dir,
                capture_output=True,
                text=True,
                timeout=60,
            )
        except subprocess.TimeoutExpired:
            print("  TIMEOUT: script ran for >60s (infinite loop?)")
            return False

        if result.returncode != 0:
            print("  FAIL: script exited with an error\n")
            print(result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr)
            self._hint(num, result.stderr)
            return False

        # Check expected output
        missing = [p for p in SUCCESS_PATTERNS.get(num, []) if p not in result.stdout]
        if missing:
            print("  PARTIAL: script ran but expected output not found")
            print(f"  Missing: {missing}")
            print("\nActual output:")
            print(result.stdout)
            return False

        print("  PASS")
        print(result.stdout)
        return True

    def _hint(self, num: int, stderr: str):
        hints = HINTS.get(num, {})
        for key, message in hints.items():
            if key.lower() in stderr.lower():
                print(f"\n  Hint: {message}")
                return

    def check_all(self):
        results = {}
        for num in sorted(EXERCISES):
            results[num] = self.check(num)
            print()

        passed = sum(results.values())
        total = len(results)
        print("=" * 50)
        print(f"Results: {passed}/{total} exercises passing")
        for num, ok in results.items():
            folder = EXERCISES[num][0]
            status = "PASS" if ok else "FAIL"
            print(f"  {num:02d}. {folder}: {status}")

    def list_exercises(self):
        print(f"{'#':>3}  {'Directory':<30} {'Script'}")
        print("-" * 55)
        for num, (folder, script) in sorted(EXERCISES.items()):
            print(f"  {num:>2}. {folder:<30} {script}")


def main():
    parser = argparse.ArgumentParser(description="Check MLE exercise solutions")
    parser.add_argument("--exercise", "-e", type=int, metavar="N", help="Exercise number (1–10)")
    parser.add_argument("--all",      "-a", action="store_true",    help="Check all exercises")
    parser.add_argument("--list",     "-l", action="store_true",    help="List exercises")
    args = parser.parse_args()

    checker = ExerciseChecker()

    if args.list:
        checker.list_exercises()
    elif args.all:
        checker.check_all()
    elif args.exercise:
        checker.check(args.exercise)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
