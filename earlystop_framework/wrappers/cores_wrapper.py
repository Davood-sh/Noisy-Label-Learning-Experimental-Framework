# wrappers/cores_wrapper.py

import os
import sys
import json
import subprocess
import time

def cores_train(config_path: str):
    """
    Run CORES Phase 1 and Phase 2 training in sequence using the given JSON config.
    Appends:
      - phase1.txt             (Phase 1 final line)
      - phase2_train.txt       (Phase 2 train line)
      - phase2_test.txt        (Phase 2 test line)
      - phase2_summary.txt     (Phase 2 summary line)
      - timings.txt            (training_time only)
    under <result_dir>/<dataset>/<model>/.
    """
    # 1) load config
    with open(config_path) as f:
        cfg = json.load(f)

    seed = cfg.get('seed', 0)     

    # 2) locate CORES root
    this_dir   = os.path.dirname(__file__)
    cores_root = os.path.abspath(os.path.join(this_dir, '..', 'models', 'CORES'))

    # 3) prepare result folder
    result_dir = cfg.get(
        'result_dir',
        os.path.abspath(os.path.join(this_dir, '..', 'results', 'cores'))
    )
    save_dir = os.path.join(result_dir, cfg['dataset'], cfg.get('model_type','cores'))
    os.makedirs(save_dir, exist_ok=True)

    def append_to(fname, line):
        with open(os.path.join(save_dir, fname), 'a') as f:
            f.write(line.rstrip() + "\n")

    # -------- Phase 1 --------
    phase1 = [
        sys.executable,
        os.path.join(cores_root, 'phase1.py'),
        f"--loss={cfg['loss']}",
        f"--dataset={cfg['dataset']}",
        f"--model={cfg.get('model_type','cores')}",
        f"--noise_type={cfg['noise_type']}",
        f"--noise_rate={cfg['noise_rate']}",
        f"--seed={seed}",
        f"--result_dir={result_dir}",
    ]
    print(f"[CORES Wrapper] Phase 1: {' '.join(phase1)}")
    t0 = time.time()
    r1 = subprocess.run(phase1, cwd=cores_root)
    if r1.returncode != 0:
        raise RuntimeError(f"Phase 1 failed (exit {r1.returncode})")
    t_phase1 = time.time() - t0
    append_to('timings.txt', f"phase1_time: {t_phase1:.1f}s")

    # pick up final line from phase1.txt
    p1f = os.path.join(save_dir, 'phase1.txt')
    if os.path.exists(p1f):
        with open(p1f) as f:
            final = [L for L in f if 'Epoch' in L and 'Train Acc' in L]
        if final:
            append_to('phase1.txt', final[-1])

    # -------- Phase 2 Training --------
    phase2_conf = cfg['phase2_config']
    phase2 = [
        sys.executable,
        os.path.join(cores_root, 'phase2', 'phase2.py'),
        '-c', phase2_conf,
        '--unsupervised',
        f'--random_state={seed}',
        '--save', os.path.join(save_dir, 'phase2_checkpoint.pth'),   
    ]
    print(f"[CORES Wrapper] Phase 2 (train): {' '.join(phase2)}")
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    t1 = time.time()
    p2 = subprocess.Popen(
        phase2, cwd=os.path.join(cores_root,'phase2'),
        env=env, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, bufsize=1
    )

    train_line = test_line = summary_line = None
    for L in p2.stdout:
        print(L, end='')
        if L.startswith("Train At Epoch"):
            train_line = L
        elif L.startswith("Test At Epoch"):
            test_line = L
        elif L.strip().startswith("OrderedDict"):
            summary_line = L

    p2.wait()
    if p2.returncode != 0:
        raise RuntimeError(f"Phase 2 training failed (exit {p2.returncode})")

    t_phase2 = time.time() - t1
    append_to('timings.txt', f"phase2_time: {t_phase2:.1f}s")

    if train_line:
        append_to('phase2_train.txt', train_line)
    if test_line:
        append_to('phase2_test.txt', test_line)
    if summary_line:
        append_to('phase2_summary.txt', summary_line)

    total_train = t_phase1 + t_phase2
    append_to('timings.txt', f"training_time: {total_train:.1f}s")

    print(f"[CORES Wrapper] Done!  training_time={total_train:.1f}s")
