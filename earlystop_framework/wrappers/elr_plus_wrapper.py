import os
import sys
import subprocess

def elr_plus_train(config_path: str):
    """
    Spawn ELR+’s own train.py using the given config.
    """
    # 1) Locate ELR_plus sub-repository where train.py lives
    this_dir = os.path.dirname(__file__)
    elr_plus_repo = os.path.abspath(
        os.path.join(this_dir, "..", "models", "ELR", "ELR_plus")
    )

    # 2) Build the command
    config_abs = os.path.abspath(config_path)
    cmd = [sys.executable, "train.py", "-c", config_abs]
    print(f"[ELR+ Wrapper] Running: {' '.join(cmd)}  (cwd={elr_plus_repo})")

    # 3) Invoke ELR+’s native training script
    result = subprocess.run(cmd, cwd=elr_plus_repo)
    if result.returncode != 0:
        raise RuntimeError(f"ELR+ training failed (exit code {result.returncode})")