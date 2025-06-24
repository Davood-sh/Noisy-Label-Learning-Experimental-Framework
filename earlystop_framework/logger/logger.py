# logger/logger.py

import os
import json
from datetime import datetime

class Logger:
    """
    Collects per-epoch metrics and writes a final JSON file under:
       results/{model}/{dataset}_.../run_{timestamp}.json
    """
    def __init__(self, cfg):
        self.cfg        = cfg
        self.records    = []
        self.model_name = cfg["model"]
        dataset         = cfg.get("dataset", "cifar10")
        noise_pct       = int(100 * cfg.get("noise_percent", 0.0))
        subdir          = f"{dataset}_noise{noise_pct}"
        self.out_dir    = os.path.join("results", self.model_name, subdir)
        os.makedirs(self.out_dir, exist_ok=True)

    def log_epoch(self, epoch, metrics):
        """
        metrics: dict of { 'train_loss': …, 'val_acc': …, … }
        """
        entry = {"epoch": epoch}
        entry.update(metrics)
        self.records.append(entry)

    def save(self, final_metrics=None):
        """
        Writes out a JSON containing:
          - cfg
          - list of per-epoch records
          - final_metrics (e.g. {'test_acc': …})
          - timestamp
        """
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        out = {
            "model":   self.model_name,
            "config":  self.cfg,
            "records": self.records,
            "final":   final_metrics or {},
            "timestamp": timestamp,
        }
        fname = f"run_{timestamp}.json"
        path  = os.path.join(self.out_dir, fname)
        with open(path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[Logger] Saved results to {path}")
