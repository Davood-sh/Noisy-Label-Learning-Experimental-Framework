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

## 🛠 Installation

**Clone:**  
```bash
git clone https://github.com/yourorg/noisy-label-framework.git
cd noisy-label-framework
```

**Create Python venv & install:**
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## ⚙️ Usage
**Configure: edit or copy configs/cores_cifar10.json:**

```bash
{
  "model":      "cores",
  "dataset":    "cifar10",
  "noise_type": "symmetric",
  "noise_rate": 0.6,
  "seed":       123,
  "epochs":     50,
  "batch_size": 128,
  "result_dir": "results/cores"
}

```
**Run:**
```bash
python -m scripts/train -c configs/cores_cifar10.json
```

## 📁 Repo Structure 
```bash
.
├── configs/                # JSON configs per experiment
├── models/
│   ├── CORES/              # Official CORES code
│   ├── ELR/                # Official ELR code
│   └── SOP/                # Official SOP code
├── scripts/
│   └── train.py            # Top-level driver
├── wrappers/
│   ├── cores_wrapper.py
│   ├── elr_wrapper.py
│   └── sop_wrapper.py
├── data_loader.py          # Standard dataloader for simple baselines
├── logger/
│   └── Logger.py           # Unified logging class
└── results/                # Populated by runs

```

## ➕ Adding a New Method

1. Clone its official repo under models/.

2. Write wrappers/<new>_wrapper.py:

3. Build the subprocess command to its train script.

4. Time train & eval.

6. Create a JSON config in configs/.

7. Run via scripts/train.py.

## 🔮 Future Work
Support more datasets and methods






