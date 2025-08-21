import sys
import subprocess
from pathlib import Path

TRAIN_SCRIPT = Path("src/train.py")
MODEL_PATH = Path("models/rf_model.pkl")

def test_train_script_runs():
    if MODEL_PATH.exists():
        MODEL_PATH.unlink()

    result = subprocess.run(
        [sys.executable, str(TRAIN_SCRIPT)],  # <-- use current venv python
        capture_output=True,
        text=True
    )
    print(result.stdout)
    print(result.stderr)

    assert result.returncode == 0, "Training script failed to run."
