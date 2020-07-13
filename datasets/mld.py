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

import torch
import torch.utils.data
import torchvision


class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
        callback_get_label func: a callback-like function which takes two arguments - dataset and index
    """

    def __init__(self, dataset, indices=None, num_samples=None, callback_get_label=None):
                
        # if indices is not provided, 
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) \
            if indices is None else indices

        # define custom callback
        self.callback_get_label = callback_get_label

        # if num_samples is not provided, 
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples
            
        # distribution of classes in the dataset 
        label_to_count = {}
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1
                
        # weight for each sample
        weights = [1.0 / label_to_count[self._get_label(dataset, idx)]
                   for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, dataset, idx):
        if isinstance(dataset, torchvision.datasets.MNIST):
            return dataset.train_labels[idx].item()
        elif isinstance(dataset, torchvision.datasets.ImageFolder):
            return dataset.imgs[idx][1]
        elif isinstance(dataset, torch.utils.data.Subset):
            return dataset.dataset.imgs[idx][1]
        elif self.callback_get_label:
            return self.callback_get_label(dataset, idx)
        else:
            raise NotImplementedError
                
    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(
            self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples
    def set_epoch(self, epoch: int):
        pass 

def _mld_dataset(config):
    train_dir = '/headless/data/mel/pytest3/ftrain/'
    test_dir = '/headless/data/mel/pytest3/fvalid/'

    train_data = dset.ImageFolder(
        train_dir,
        transforms.Compose([
            #transforms.Resize((224,224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
        ]))

    test_data = dset.ImageFolder(
        test_dir,
        transforms.Compose([
            #transforms.Resize((224,224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
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
    #if config.distributed:
    train_sampler = ImbalancedDatasetSampler(train_data)
    if config.distributed:
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_data)
    else:
        test_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=config.batch_size,
        sampler=train_sampler,
        pin_memory=False, num_workers=config.workers)

    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=config.batch_size,
        sampler=test_sampler,
        pin_memory=False, num_workers=config.workers)

    return [train_loader, test_loader], [train_sampler, test_sampler]
