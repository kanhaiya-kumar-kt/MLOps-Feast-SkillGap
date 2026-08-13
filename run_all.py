"""Run the complete local Feast assignment workflow."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def run(cmd, cwd=ROOT):
    print("\n$", " ".join(map(str, cmd)))
    subprocess.run(cmd, cwd=cwd, check=True)

run([sys.executable, "src/prepare_data.py"])
run(["feast", "-c", "feature_repo", "apply"])
run([sys.executable, "src/historical_features.py"])
run([sys.executable, "src/train_model.py"])
print("\nNow run materialization manually with the current UTC timestamp:")
print("feast -c feature_repo materialize-incremental <UTC_TIMESTAMP>")
print("Then run: python src/online_prediction.py")
