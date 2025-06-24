import os
import sys
import subprocess

def sop_train(config_path: str):
    """
    Spawn SOP’s own train.py using the given config.
    """
    # 1) Locate the SOP repository folder
    this_dir = os.path.dirname(__file__)
    sop_repo = os.path.abspath(
        os.path.join(this_dir, "..", "models", "SOP")
    )

    # 2) Build the command to run SOP's train script
    config_abs = os.path.abspath(config_path)
    cmd = [sys.executable, "train.py", "-c", config_abs]
    print(f"[SOP Wrapper] Running: {' '.join(cmd)}  (cwd={sop_repo})")

    

    # 3) Invoke SOP's native training script
    result = subprocess.run(cmd, cwd=sop_repo)
    if result.returncode != 0:
        raise RuntimeError(f"SOP training failed (exit code {result.returncode})")