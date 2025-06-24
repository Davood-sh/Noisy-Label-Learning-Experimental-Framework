# scripts/train.py

import sys
import time                           # new
import torch
import torch.nn as nn
import torch.optim as optim

from scripts.parse_config import parse_args, load_config, get_model
from data_loader import get_dataloader
from logger.logger import Logger

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)

def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total   = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            _, preds = out.max(1)
            correct += (preds == y).sum().item()
            total   += y.size(0)
    return 100.0 * correct / total

if __name__ == "__main__":
    # 1) Parse config
    args       = parse_args()
    cfg        = load_config(args.config)
    model_name = cfg.get("model", "").lower()

    # 2) Delegate to wrappers if needed
    if model_name == "elr":
        from wrappers.elr_wrapper import elr_train
        elr_train(args.config); sys.exit(0)
    elif model_name == "elr_plus":
        from wrappers.elr_plus_wrapper import elr_plus_train
        elr_plus_train(args.config); sys.exit(0)
    elif model_name == "sop":
        from wrappers.sop_wrapper import sop_train
        sop_train(args.config); sys.exit(0)
    elif model_name == "cores":
        from wrappers.cores_wrapper import cores_train
        cores_train(args.config); sys.exit(0)

    # 3) Standard pipeline
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader = get_dataloader(cfg)
    model = get_model(cfg).to(device)

    # 4) Loss & optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=cfg.get("lr", 0.1),
        momentum=0.9,
        weight_decay=cfg.get("weight_decay", 5e-4)
    )

    # 5) Logger
    logger = Logger(cfg)

    # 6) Training loop with timing
    epochs = cfg.get("epochs", 50)
    t_train_start = time.time()
    for epoch in range(1, epochs + 1):
        t_ep_start = time.time()
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        t_ep = time.time() - t_ep_start

        val_acc = evaluate(model, val_loader, device)
        print(f"Epoch {epoch:3d}  Train Loss: {train_loss:.4f}  Val Acc: {val_acc:.2f}%  (epoch_time: {t_ep:.1f}s)")

        logger.log_epoch(epoch, {
            "train_loss":  train_loss,
            "val_acc":     val_acc,
            "epoch_time":  t_ep
        })
    training_time = time.time() - t_train_start

    # 7) Final test (inference) timing
    t_inf_start = time.time()
    test_acc = evaluate(model, test_loader, device)
    inference_time = time.time() - t_inf_start

    print(f"Test Accuracy: {test_acc:.2f}%  (inference_time: {inference_time:.1f}s)")

    # 8) Save results (including our two timers)
    logger.save(final_metrics={
        "test_acc":       test_acc,
        "training_time":  round(training_time, 1),
        "inference_time": round(inference_time, 1)
    })
