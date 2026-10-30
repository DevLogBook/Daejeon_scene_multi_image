"""Evaluate GRM on synthetic agricultural pairs with ground-truth homographies."""
from pathlib import Path
from agri_eval_common import main

if __name__ == "__main__":
    main("GRM", "synthetic", Path(__file__).resolve().parent)
