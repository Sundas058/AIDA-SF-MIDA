"""Supervised source-training entry point."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from idm.engine import train_main
if __name__ == "__main__":
    train_main(baseline=True)

