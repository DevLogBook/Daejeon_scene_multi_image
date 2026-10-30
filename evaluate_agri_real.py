"""Evaluate and save all real agricultural pairs, including unannotated pairs."""
from pathlib import Path
from agri_eval_common import main

if __name__ == "__main__":
    main("GRM", "real", Path(__file__).resolve().parent)
