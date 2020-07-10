# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os

import numpy as np
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
from PIL import Image
from datasets.data_utils import ImageNetPolicy
from datasets.data_utils import SubsetDistributedSampler


def _mld_dataset(config):
    train_dir = '/headless/data/mel/pytest3/ftrain/'
    test_dir = '/headless/data/mel/pytest3/fvalid/'

    train_data = dset.ImageFolder(
        train_dir,
        transforms.Compose([
            transforms.Resize((224,224)),
            #transforms.RandomHorizontalFlip(),
            #transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
        ]))

    test_data = dset.ImageFolder(
        test_dir,
        transforms.Compose([
            transforms.Resize((224,224)),
            #transforms.RandomHorizontalFlip(),
            #transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
        ]))

    return train_data, test_data


def get_search_datasets(config):
    train_data, test_data = _mld_dataset(config)
    num_train = len(train_data)
    num_val = len(test_data)
    indices = list(range(num_train))
    vindices = list(range(num_val))
    split_mid = int(np.floor(0.5 * num_train))
    vsplit_mid = int(np.floor(0.5 * num_val))
    if config.distributed:
        train_sampler = SubsetDistributedSampler(train_data, indices[:split_mid])
        valid_sampler = SubsetDistributedSampler(test_data, vindices[vsplit_mid:num_val])
    else:
        train_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[:split_mid])
        valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(vindices[vsplit_mid:num_val])

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=config.batch_size,
        sampler=train_sampler,
        pin_memory=True, num_workers=config.workers)

    valid_loader = torch.utils.data.DataLoader(
        test_data, batch_size=config.batch_size,
        sampler=valid_sampler,
        pin_memory=True, num_workers=config.workers)

    return [train_loader, valid_loader], [train_sampler, valid_sampler]


def get_augment_datasets(config):
    train_data, test_data = _mld_dataset(config)
    if config.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_data)
    else:
        train_sampler = test_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=config.batch_size,
        sampler=train_sampler,
        pin_memory=True, num_workers=config.workers)

    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=config.batch_size,
        sampler=test_sampler,
        pin_memory=True, num_workers=config.workers)

    return [train_loader, test_loader], [train_sampler, test_sampler]
