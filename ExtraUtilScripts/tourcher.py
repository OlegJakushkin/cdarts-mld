import torch
from torchvision import transforms, datasets
import logging
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn

import utils
from config import SearchConfig
from nni.nas.pytorch.cdarts import CdartsTrainer
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import torchvision
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, utils, datasets
from torch.utils.data import Dataset, DataLoader, random_split, SubsetRandomSampler, WeightedRandomSampler

import numpy as np
from nni.nas.pytorch import mutables
from utils import parse_results
from aux_head import DistillHeadCIFAR, DistillHeadImagenet, AuxiliaryHeadCIFAR, AuxiliaryHeadImageNet


OPS = {
    'avg_pool_3x3': lambda C, stride, affine: PoolWithoutBN('avg', C, 3, stride, 1, affine=affine),
    'max_pool_3x3': lambda C, stride, affine: PoolWithoutBN('max', C, 3, stride, 1, affine=affine),
    'skip_connect': lambda C, stride, affine: nn.Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine),
    'sep_conv_3x3': lambda C, stride, affine: SepConv(C, C, 3, stride, 1, affine=affine),
    'sep_conv_5x5': lambda C, stride, affine: SepConv(C, C, 5, stride, 2, affine=affine),
    'sep_conv_7x7': lambda C, stride, affine: SepConv(C, C, 7, stride, 3, affine=affine),
    'dil_conv_3x3': lambda C, stride, affine: DilConv(C, C, 3, stride, 2, 2, affine=affine),  # 5x5
    'dil_conv_5x5': lambda C, stride, affine: DilConv(C, C, 5, stride, 4, 2, affine=affine),  # 9x9
    'conv_7x1_1x7': lambda C, stride, affine: FacConv(C, C, 7, stride, 3, affine=affine)
}

PRIMITIVES = [
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',  # identity
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5',
]


class DropPath(nn.Module):
    def __init__(self, p=0.):
        """
        Drop path with probability.
        Parameters
        ----------
        p : float
            Probability of an path to be zeroed.
        """
        super().__init__()
        self.p = p

    def forward(self, x):
        if self.training and self.p > 0.:
            keep_prob = 1. - self.p
            # per data point mask
            mask = torch.zeros((x.size(0), 1, 1, 1), device=x.device).bernoulli_(keep_prob)
            return x / keep_prob * mask

        return x


class PoolWithoutBN(nn.Module):
    """
    AvgPool or MaxPool with BN. `pool_type` must be `max` or `avg`.
    """

    def __init__(self, pool_type, C, kernel_size, stride, padding, affine=True):
        super().__init__()
        if pool_type.lower() == 'max':
            self.pool = nn.MaxPool2d(kernel_size, stride, padding)
        elif pool_type.lower() == 'avg':
            self.pool = nn.AvgPool2d(kernel_size, stride, padding, count_include_pad=False)
        else:
            raise NotImplementedError("Pool doesn't support pooling type other than max and avg.")

    def forward(self, x):
        out = self.pool(x)
        return out


class StdConv(nn.Module):
    """
    Standard conv: ReLU - Conv - BN
    """

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(C_in, C_out, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(C_out, affine=affine)
        )

    def forward(self, x):
        return self.net(x)


class FacConv(nn.Module):
    """
    Factorized conv: ReLU - Conv(Kx1) - Conv(1xK) - BN
    """

    def __init__(self, C_in, C_out, kernel_length, stride, padding, affine=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(C_in, C_in, (kernel_length, 1), stride, padding, bias=False),
            nn.Conv2d(C_in, C_out, (1, kernel_length), stride, padding, bias=False),
            nn.BatchNorm2d(C_out, affine=affine)
        )

    def forward(self, x):
        return self.net(x)


class DilConv(nn.Module):
    """
    (Dilated) depthwise separable conv.
    ReLU - (Dilated) depthwise separable - Pointwise - BN.
    If dilation == 2, 3x3 conv => 5x5 receptive field, 5x5 conv => 9x9 receptive field.
    """

    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(C_in, C_in, kernel_size, stride, padding, dilation=dilation, groups=C_in,
                      bias=False),
            nn.Conv2d(C_in, C_out, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine)
        )

    def forward(self, x):
        return self.net(x)


class SepConv(nn.Module):
    """
    Depthwise separable conv.
    DilConv(dilation=1) * 2.
    """

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super().__init__()
        self.net = nn.Sequential(
            DilConv(C_in, C_in, kernel_size, stride, padding, dilation=1, affine=affine),
            DilConv(C_in, C_out, kernel_size, 1, padding, dilation=1, affine=affine)
        )

    def forward(self, x):
        return self.net(x)


class FactorizedReduce(nn.Module):
    """
    Reduce feature map size by factorized pointwise (stride=2).
    """

    def __init__(self, C_in, C_out, affine=True):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)

    def forward(self, x):
        x = self.relu(x)
        out = torch.cat([self.conv1(x), self.conv2(x[:, :, 1:, 1:])], dim=1)
        out = self.bn(out)
        return out


class Node(nn.Module):
    def __init__(self, node_id, num_prev_nodes, channels, num_downsample_connect):
        super().__init__()
        self.ops = nn.ModuleList()
        choice_keys = []
        for i in range(num_prev_nodes):
            stride = 2 if i < num_downsample_connect else 1
            choice_keys.append("{}_p{}".format(node_id, i))
            self.ops.append(mutables.LayerChoice([ops.OPS[k](channels, stride, False) for k in ops.PRIMITIVES],
                                                 key=choice_keys[-1]))
        self.drop_path = ops.DropPath()
        self.input_switch = mutables.InputChoice(choose_from=choice_keys, n_chosen=2, key="{}_switch".format(node_id))

    def forward(self, prev_nodes):
        assert len(self.ops) == len(prev_nodes)
        out = [op(node) for op, node in zip(self.ops, prev_nodes)]
        out = [self.drop_path(o) if o is not None else None for o in out]
        return self.input_switch(out)


class Cell(nn.Module):

    def __init__(self, n_nodes, channels_pp, channels_p, channels, reduction_p, reduction):
        super().__init__()
        self.reduction = reduction
        self.n_nodes = n_nodes

        # If previous cell is reduction cell, current input size does not match with
        # output size of cell[k-2]. So the output[k-2] should be reduced by preprocessing.
        if reduction_p:
            self.preproc0 = ops.FactorizedReduce(channels_pp, channels, affine=False)
        else:
            self.preproc0 = ops.StdConv(channels_pp, channels, 1, 1, 0, affine=False)
        self.preproc1 = ops.StdConv(channels_p, channels, 1, 1, 0, affine=False)

        # generate dag
        self.mutable_ops = nn.ModuleList()
        for depth in range(2, self.n_nodes + 2):
            self.mutable_ops.append(Node("{}_n{}".format("reduce" if reduction else "normal", depth),
                                         depth, channels, 2 if reduction else 0))

    def forward(self, s0, s1):
        # s0, s1 are the outputs of previous previous cell and previous cell, respectively.
        tensors = [self.preproc0(s0), self.preproc1(s1)]
        for node in self.mutable_ops:
            cur_tensor = node(tensors)
            tensors.append(cur_tensor)

        output = torch.cat(tensors[2:], dim=1)
        return output

class Model(nn.Module):

    def __init__(self, dataset, n_layers, in_channels=3, channels=16, n_nodes=4, retrain=False, shared_modules=None):
        super().__init__()
        assert dataset in ["mld"]
        self.dataset = dataset
        self.input_size = 512
        self.in_channels = in_channels
        self.channels = channels
        self.n_nodes = n_nodes
        self.aux_size = {2 * n_layers // 3: self.input_size // 4}
        self.n_classes = 2
        self.aux_head_class = AuxiliaryHeadCIFAR if retrain else DistillHeadCIFAR
        if not retrain:
            self.aux_size = {n_layers // 3: 6, 2 * n_layers // 3: 6}

        self.n_layers = n_layers
        self.aux_head = nn.ModuleDict()
        self.ensemble_param = nn.Parameter(torch.rand(len(self.aux_size) + 1) / (len(self.aux_size) + 1)) \
            if not retrain else None

        stem_multiplier = 3
        c_cur = stem_multiplier * self.channels
        self.shared_modules = {}  # do not wrap with ModuleDict
        if shared_modules is not None:
            self.stem = shared_modules["stem"]
        else:
            self.stem = nn.Sequential(
                nn.Conv2d(in_channels, c_cur, 3, 1, 1, bias=False),
                nn.BatchNorm2d(c_cur)
            )
            self.shared_modules["stem"] = self.stem

        # for the first cell, stem is used for both s0 and s1
        # [!] channels_pp and channels_p is output channel size, but c_cur is input channel size.
        channels_pp, channels_p, c_cur = c_cur, c_cur, channels

        self.cells = nn.ModuleList()
        reduction_p, reduction = False, False
        aux_head_count = 0
        for i in range(n_layers):
            reduction_p, reduction = reduction, False
            if i in [n_layers // 3, 2 * n_layers // 3]:
                c_cur *= 2
                reduction = True

            cell = Cell(n_nodes, channels_pp, channels_p, c_cur, reduction_p, reduction)
            self.cells.append(cell)
            c_cur_out = c_cur * n_nodes
            if i in self.aux_size:
                if shared_modules is not None:
                    self.aux_head[str(i)] = shared_modules["aux" + str(aux_head_count)]
                else:
                    self.aux_head[str(i)] = self.aux_head_class(c_cur_out, self.aux_size[i], self.n_classes)
                    self.shared_modules["aux" + str(aux_head_count)] = self.aux_head[str(i)]
                aux_head_count += 1
            channels_pp, channels_p = channels_p, c_cur_out

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(channels_p, self.n_classes)

    def forward(self, x):
        s0 = s1 = self.stem(x)
        outputs = []

        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1)
            if str(i) in self.aux_head:
                outputs.append(self.aux_head[str(i)](s1))

        out = self.gap(s1)
        out = out.view(out.size(0), -1)  # flatten
        logits = self.linear(out)
        outputs.append(logits)

        if self.ensemble_param is None:
            assert len(outputs) == 2
            return outputs[1], outputs[0]
        else:
            em_output = torch.cat([(e * o) for e, o in zip(F.softmax(self.ensemble_param, dim=0), outputs)], 0)
            return logits, em_output

    def drop_path_prob(self, p):
        for module in self.modules():
            if isinstance(module, ops.DropPath):
                module.p = p

    def plot_genotype(self, results, logger):
        genotypes = parse_results(results, self.n_nodes)
        logger.info(genotypes)
        return genotypes

if __name__ == "__main__":
    config = SearchConfig()
    main_proc = not config.distributed or config.local_rank == 0
    if config.distributed:
        torch.cuda.set_device(config.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method=config.dist_url,
                                             rank=config.local_rank, world_size=config.world_size)
    if main_proc:
        os.makedirs(config.output_path, exist_ok=True)
    if config.distributed:
        torch.distributed.barrier()
    logger = utils.get_logger(os.path.join(config.output_path, 'search.log'))
    if main_proc:
        config.print_params(logger.info)

    utils.reset_seed(config.seed)

    data_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor()
    ])
    mel_dataset_train = datasets.ImageFolder(root='./ftrain',
                                             transform=data_transform)
    train_sampler = torch.utils.data.distributed.DistributedSampler(mel_dataset_train)
    dataset_loader_train = torch.utils.data.DataLoader(mel_dataset_train,
                                                       batch_size=64, shuffle=True, sampler=train_sampler,
                                                       num_workers=config.workers)


    mel_dataset_valid = datasets.ImageFolder(root='./fvalid',
                                             transform=data_transform)
    valid_sampler = torch.utils.data.distributed.DistributedSampler(mel_dataset_valid)
    dataset_loader_valid = torch.utils.data.DataLoader(mel_dataset_valid,
                                                       batch_size=64, shuffle=True, sampler=valid_sampler,
                                                       num_workers=config.workers)

    idx2class_train = {v: k for k, v in mel_dataset_train.class_to_idx.items()}
    idx2class_valid = {v: k for k, v in mel_dataset_valid.class_to_idx.items()}


    def get_class_distribution_loaders(dataloader_obj, dataset_obj, idx2class):
        count_dict = {k: 0 for k, v in dataset_obj.class_to_idx.items()}

        for _, j in dataloader_obj:
            y_idx = j.item()
            y_lbl = idx2class[y_idx]
            count_dict[str(y_lbl)] += 1

        return count_dict


    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18, 7))
    sns.barplot(data=pd.DataFrame.from_dict(
        [get_class_distribution_loaders(dataset_loader_train, mel_dataset_train, idx2class_train)]).melt(),
                x="variable", y="value", hue="variable", ax=axes[0]).set_title('Train Set').get_figure().savefig(
        'train.png')
    sns.barplot(data=pd.DataFrame.from_dict(
        [get_class_distribution_loaders(dataset_loader_valid, mel_dataset_valid, idx2class_valid)]).melt(),
                x="variable", y="value", hue="variable", ax=axes[1]).set_title('Val Set').get_figure().savefig(
        'valid.png')

    model_small = Model(config.dataset, 15).cuda()
    if config.share_module:
        model_large = Model(config.dataset, 35 , shared_modules=model_small.shared_modules).cuda()
    else:
        model_large = Model(config.dataset, 35).cuda()

    criterion = nn.CrossEntropyLoss()
    trainer = CdartsTrainer(model_small, model_large, criterion, [dataset_loader_train, dataset_loader_valid], [train_sampler, valid_sampler], logger,
                            config.regular_coeff, config.regular_ratio, config.warmup_epochs, config.fix_head,
                            config.epochs, config.steps_per_epoch, config.loss_alpha, config.loss_T, config.distributed,
                            config.log_frequency, config.grad_clip, config.interactive_type, config.output_path,
                            config.w_lr, config.w_momentum, config.w_weight_decay, config.alpha_lr, config.alpha_weight_decay,
                            config.nasnet_lr, config.local_rank, config.share_module)
    trainer.train()