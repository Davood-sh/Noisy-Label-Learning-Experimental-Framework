# Noisy-Label Learning Experimental Framework

A unified, reproducible framework for benchmarking noisy-label learning methods side-by-side on standard vision datasets (CIFAR-10, CIFAR-100, MNIST, etc.). Drop in new algorithms via lightweight wrappers—no core code changes required.

---

## 🚀 Features

### Consistent Data Pipeline  
Apply the same augmentations, normalization, splits, and noise injections across methods.

### Reproducibility  
Single, user-specified seed (e.g. `123`) controls Python, NumPy, and PyTorch RNGs so all methods see identical batches and initial weights.

### Method Wrappers  
Each algorithm (CORES, ELR, ELR⁺, SOP, …) lives in its own subfolder. A small `wrappers/<method>_wrapper.py`:

1. Reads a JSON config.  
2. Calls the method’s official training entrypoint in a subprocess.  
3. Captures key console outputs (accuracy, loss).  
4. Times training vs. inference.  
5. Appends results to standard log files.

### Unified Logging  
Results live under:
results/<method>/<dataset>/<model_type>/run_<timestamp>/


---

## ➕ Open to Extensions

Add any new noisy-label method by writing a tiny wrapper and a JSON config—no need to modify core scripts.  

