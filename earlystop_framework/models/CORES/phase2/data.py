import logging
import os
import copy
import numpy as np
import torch
import torchvision
from PIL import Image

from torch.utils.data import Subset, Dataset, DataLoader
from torchvision.transforms import transforms
from sklearn.model_selection import StratifiedShuffleSplit
from theconf import Config as C

from archive import autoaug_policy, autoaug_paper_cifar10, fa_reduced_cifar10
from augmentations import *
from common import get_logger
from samplers.stratified_sampler import StratifiedSampler
from utils import noisify

logger = get_logger('Unsupervised Data Augmentation')
logger.setLevel(logging.INFO)

class MyCIFAR10(torchvision.datasets.CIFAR10):
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target, index

def get_dataloaders(dataset, batch, batch_unsup, dataroot, with_noise=True, random_state=0, unsup_idx=set()):
    # --- transforms ---
    if 'cifar' in dataset:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010)),
        ])
        transform_valid = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010)),
        ])
    else:
        raise ValueError(f'dataset={dataset}')

    # --- autoaugment policy insert ---
    autoaug = transforms.Compose([])
    aug_cfg = C.get()['aug']
    if isinstance(aug_cfg, list):
        autoaug.transforms.insert(0, Augmentation(aug_cfg))
    else:
        if aug_cfg == 'fa_reduced_cifar10':
            autoaug.transforms.insert(0, Augmentation(fa_reduced_cifar10()))
        elif aug_cfg == 'autoaug_cifar10':
            autoaug.transforms.insert(0, Augmentation(autoaug_paper_cifar10()))
        elif aug_cfg == 'autoaug_extend':
            autoaug.transforms.insert(0, Augmentation(autoaug_policy()))
        elif aug_cfg == 'default':
            pass
        else:
            raise ValueError(f'unknown aug {aug_cfg}')
    transform_train.transforms.insert(0, autoaug)
    if C.get()['cutout'] > 0:
        transform_train.transforms.append(CutoutDefault(C.get()['cutout']))

    # --- load base datasets ---
    if dataset == 'cifar10':
        total_trainset = torchvision.datasets.CIFAR10(root=dataroot, train=True,  download=True, transform=transform_train)
        unsup_trainset = torchvision.datasets.CIFAR10(root=dataroot, train=True,  download=True, transform=None)
        testset        = torchvision.datasets.CIFAR10(root=dataroot, train=False, download=True, transform=transform_test)
    elif dataset == 'cifar100':
        total_trainset = torchvision.datasets.CIFAR100(root=dataroot, train=True,  download=True, transform=transform_train)
        unsup_trainset = torchvision.datasets.CIFAR100(root=dataroot, train=True,  download=True, transform=None)
        testset        = torchvision.datasets.CIFAR100(root=dataroot, train=False, download=True, transform=transform_test)
    else:
        raise ValueError(f'invalid dataset name={dataset}')

    # --- split supervised / unsupervised subsets ---
    if not with_noise:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=46000, random_state=random_state)
        train_idx, valid_idx = next(sss.split(range(len(total_trainset)), total_trainset.targets))

        trainset = Subset(total_trainset, train_idx)
        otherset = Subset(unsup_trainset, valid_idx)
        otherset = UnsupervisedDataset(otherset, transform_valid, autoaug, cutout=C.get()['cutout'])

    else:
        # load noisy labels & build noisy supervised subset
        train_labels_with_noise = np.load(C.get()['train_labels']).reshape(-1)
        noisy_trainset = copy.deepcopy(total_trainset)
        noisy_trainset.targets = train_labels_with_noise.tolist()

        all_idx = list(range(len(total_trainset)))
        sup_idx = [i for i in all_idx if i not in unsup_idx]

        trainset = Subset(noisy_trainset, sup_idx)
        otherset = Subset(unsup_trainset, unsup_idx)
        otherset = UnsupervisedDataset(otherset, transform_valid, autoaug, cutout=C.get()['cutout'])

        # assign .targets for sampler
        sup_labels_with_noise = [noisy_trainset.targets[i] for i in sup_idx]
        trainset.targets = sup_labels_with_noise

    # ────────────────────────────────────────────────────────────
    # **Fix for StratifiedSampler**: ensure Subset.trainset.targets exists
    if isinstance(trainset, Subset) and not hasattr(trainset, 'targets'):
        base = trainset.dataset
        trainset.targets = [base.targets[i] for i in trainset.indices]
    # ────────────────────────────────────────────────────────────
    nw = C.get().get('num_workers', 4)
    # --- DataLoaders ---
    trainloader = DataLoader(
        trainset, batch_size=batch, shuffle=False,
        num_workers=0,
        pin_memory=False, sampler=StratifiedSampler(trainset.targets), drop_last=True
    )
    unsuploader = DataLoader(
        otherset, batch_size=batch_unsup, shuffle=True,
        num_workers=0,
        pin_memory=False, drop_last=True
    )
    testloader = DataLoader(
        testset, batch_size=batch, shuffle=False,
        num_workers=0,
        pin_memory=False, drop_last=False
    )

    return trainloader, unsuploader, testloader


class CutoutDefault(object):
    def __init__(self, length): self.length = length
    def __call__(self, img):
        if self.length <= 0: return img
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y, x = np.random.randint(h), np.random.randint(w)
        y1, y2 = np.clip(y - self.length//2, 0, h), np.clip(y + self.length//2, 0, h)
        x1, x2 = np.clip(x - self.length//2, 0, w), np.clip(x + self.length//2, 0, w)
        mask[y1:y2, x1:x2] = 0.
        mask = torch.from_numpy(mask).expand_as(img)
        return img * mask


class Augmentation(object):
    def __init__(self, policies): self.policies = policies
    def __call__(self, img):
        import random
        for _ in range(1):
            policy = random.choice(self.policies)
            for name, pr, level in policy:
                if random.random() > pr: continue
                img = apply_augment(img, name, level)
        return img


class UnsupervisedDataset(Dataset):
    def __init__(self, dataset, transform_default, transform_aug, cutout=0):
        self.dataset = dataset
        self.transform_default = transform_default
        self.transform_aug = transform_aug
        self.transform_cutout = CutoutDefault(cutout)

    def __getitem__(self, index):
        img, _ = self.dataset[index]
        img1 = self.transform_default(img)
        img2 = self.transform_default(self.transform_aug(img))
        img2 = self.transform_cutout(img2)
        return img1, img2

    def __len__(self):
        return len(self.dataset)
