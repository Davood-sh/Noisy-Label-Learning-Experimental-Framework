# wrappers/elr_wrapper.py

import os
import sys
import subprocess
import time

def elr_train(config_path: str):
    """
    Spawn ELR's own train.py using the given config.
    """
    # 1) Locate ELR's inner folder where train.py lives
    this_dir = os.path.dirname(__file__)
    elr_repo = os.path.abspath(
        os.path.join(this_dir, "..", "models", "ELR", "ELR")
    )

    # 2) Use an absolute path so ELR can locate the config file
    config_abs = os.path.abspath(config_path)

    # 3) Build the command
    cmd = [sys.executable, "train.py", "-c", config_abs]
    print(f"[ELR Wrapper] Running: {' '.join(cmd)}  (cwd={elr_repo})")

    # 4) Invoke ELR’s native training script
    result = subprocess.run(cmd, cwd=elr_repo)
    if result.returncode != 0:
        raise RuntimeError(f"ELR training failed (exit code {result.returncode})")
