# -*- coding: utf-8 -*-
import os
import sys
import shutil
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from data.cifar import CIFAR10, CIFAR100
from data.mnist import MNIST
from data.datasets import input_dataset
from models import CNN, ResNet34
from loss import loss_cross_entropy, loss_cores, f_beta
from random import sample
from datetime import datetime
import time   

def parse_args():
    parser = argparse.ArgumentParser(description="CORES Phase 1: Sample Sieve")
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--lr_plan', type=str, default='cyclic', help='base or cyclic')
    parser.add_argument('--loss', type=str, default='cores', help='ce or cores')
    parser.add_argument('--result_dir', type=str, default='results', help='where to save results')
    parser.add_argument('--noise_rate', type=float, default=0.2)
    parser.add_argument('--noise_type', type=str, default='pairflip', help='pairflip, symmetric, instance')
    parser.add_argument('--dataset', type=str, default='cifar10', help='mnist, cifar10, or cifar100')
    parser.add_argument('--model', type=str, default='cnn', help='cnn or resnet')
    parser.add_argument('--n_epoch', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--print_freq', type=int, default=50)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--resume', '-r', action='store_true')
    return parser.parse_args()

def accuracy(logit, target, topk=(1,)):
    output = F.softmax(logit, dim=1)
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def adjust_learning_rate(optimizer, epoch, alpha_plan):
    for param_group in optimizer.param_groups:
        param_group['lr'] = alpha_plan[epoch] / (1 + f_beta(epoch))

def train(epoch, num_classes, train_loader, model, optimizer,
          loss_all, loss_div_all, loss_type, noise_prior=None, args=None):
    print(f'[{datetime.now():%Y-%m-%d %H:%M:%S}] Starting epoch {epoch+1}, beta={f_beta(epoch):.4f}')
    train_total = 0
    train_correct = 0
    num_samples = len(train_loader.dataset)
    v_list = np.zeros(num_samples)
    idx_each_class_noisy = [[] for _ in range(num_classes)]

    if noise_prior is not None and not isinstance(noise_prior, torch.Tensor):
        noise_prior = torch.tensor(noise_prior.astype('float32')).cuda().unsqueeze(0)

    for i, (images, labels, indexes) in enumerate(train_loader):
        ind = indexes.cpu().numpy()
        bs = len(ind)
        images = images.cuda()
        labels = labels.cuda()

        logits = model(images)
        prec, _ = accuracy(logits, labels, topk=(1,5))
        train_total += 1
        train_correct += prec.item()

        if loss_type == 'ce':
            loss = loss_cross_entropy(
                epoch, logits, labels, range(num_classes),
                ind, train_loader.dataset.noise_or_not,
                loss_all, loss_div_all
            )
        else:  # cores
            loss, loss_v = loss_cores(
                epoch, logits, labels, range(num_classes),
                ind, train_loader.dataset.noise_or_not,
                loss_all, loss_div_all, noise_prior=noise_prior
            )
            v_list[ind] = loss_v
            for j in range(bs):
                if loss_v[j] == 0:
                    idx_each_class_noisy[labels[j].item()].append(ind[j])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % args.print_freq == 0:
            print(f"  Iter {i+1}/{len(train_loader)}  Acc: {prec.item():.4f}  Loss: {loss.item():.4f}")

    noise_prior_delta = np.array([len(x) for x in idx_each_class_noisy])
    train_acc = train_correct / train_total
    print(f"Epoch {epoch+1} completed. Train Acc: {train_acc:.4f}")
    print("Noise prior delta per class:", noise_prior_delta)
    return train_acc, noise_prior_delta

def evaluate(test_loader, model, save_dir, args, best_acc=0.0, epoch=0):
    model.eval()
    correct = 0
    total = 0
    for images, labels, _ in test_loader:
        images = images.cuda()
        outputs = F.softmax(model(images), dim=1)
        _, preds = outputs.max(1)
        total += labels.size(0)
        correct += (preds.cpu() == labels).sum().item()
    acc = 100 * correct / total
    if acc > best_acc:
        ckpt = {
            'state_dict': model.state_dict(),
            'epoch': epoch,
            'acc': acc
        }
        os.makedirs(save_dir, exist_ok=True)
        torch.save(ckpt, os.path.join(save_dir, f"best.pth.tar"))
        best_acc = acc
    return acc, best_acc

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # Load dataset
    train_ds, test_ds, num_classes, num_samples = input_dataset(
        args.dataset, args.noise_type, args.noise_rate
    )
    print("train labels:", len(train_ds.train_labels), train_ds.train_labels[:10])

    # Model
    model = CNN(3, num_classes) if args.model == 'cnn' else ResNet34(num_classes)
    model.cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    # Prepare dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=64, shuffle=True,
        num_workers=args.num_workers, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=64, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )

    loss_all = np.zeros((num_samples, args.n_epoch))
    loss_div_all = np.zeros((num_samples, args.n_epoch))

    save_dir = os.path.join(args.result_dir, args.dataset, args.model)
    os.makedirs(save_dir, exist_ok=True)

    # Clean old log
    log_txt = os.path.join(save_dir, f"{args.loss}{args.noise_type}{args.noise_rate}.txt")
    if os.path.exists(log_txt):
        os.remove(log_txt)
    with open(log_txt, "w") as f:
        f.write("epoch train_acc test_acc\n")

    # Learning rate schedule
    alpha_plan = [args.lr] * args.n_epoch

    best_acc = 0.0
    noise_prior = train_ds.noise_prior


    # Timers
    total_train_time = 0.0
    total_infer_time = 0.0


    for epoch in range(args.n_epoch):
        adjust_learning_rate(optimizer, epoch, alpha_plan)

        t0 = time.time()
        train_acc, noise_delta = train(
            epoch, num_classes, train_loader, model, optimizer,
            loss_all, loss_div_all, args.loss, noise_prior=noise_prior, args=args
        )
        epoch_train_time = time.time() - t0
        total_train_time += epoch_train_time
        # --- TRAINING TIMER END ---

        noise_prior = (noise_prior * num_samples - noise_delta) / num_samples

        ti0 = time.time()
        test_acc, best_acc = evaluate(test_loader, model, save_dir, args, best_acc, epoch)

        epoch_infer_time = time.time() - ti0
        total_infer_time += epoch_infer_time
        # --- INFERENCE TIMER END ---

        with open(log_txt, "a") as f:
            f.write(f"{epoch} {train_acc:.4f} {test_acc:.4f}\n")
            f.write(f"\ntraining_time: {total_train_time:.1f}s\n")
            f.write(f"inference_time: {total_infer_time:.1f}s\n")

if __name__ == '__main__':
    # On Windows, protect the entry point for multiprocessing
    from multiprocessing import freeze_support
    freeze_support()
    main()
