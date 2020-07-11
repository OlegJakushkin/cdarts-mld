# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import logging
import os
import time
from argparse import ArgumentParser

import torch
import torch.nn as nn

import apex  # pylint: disable=import-error
import datasets
import utils
from apex.parallel import DistributedDataParallel  # pylint: disable=import-error
from config import RetrainConfig
from datasets.mld import get_augment_datasets
from model import Model
from nni.nas.pytorch.fixed import apply_fixed_architecture
from nni.nas.pytorch.utils import AverageMeterGroup


class CyclicIterator:
    def __init__(self, loader, sampler):
        self.loader = loader
        self.sampler = sampler
        self.epoch = 0
        self._next_epoch()

    def _next_epoch(self):
        self.iterator = iter(self.loader)
        self.epoch += 1

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return next(self.iterator)
        except StopIteration:
            self._next_epoch()
            return next(self.iterator)


def train(logger, config, train_loader, model, optimizer, criterion, epoch, main_proc, fake_batch=8, steps=150):
    meters = AverageMeterGroup()
    cur_lr = optimizer.param_groups[0]["lr"]
    if main_proc:
        logger.info("Epoch %d LR %.6f", epoch, cur_lr)

    model.train()
    for step in range(steps):
        totall_l =0
        for fb in range(fake_batch):
            x, y = next(train_loader)
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            optimizer.zero_grad()
            logits, aux_logits = model(x)
            loss = criterion(logits, y)
            if config.aux_weight > 0.:
                loss += config.aux_weight * criterion(aux_logits, y)
            loss = loss/fake_batch
            try:
                loss.backward()
            except:
                break;
            totall_l += loss
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()

        prec1, prec1 = utils.accuracy(logits, y, topk=(1, 1))
        metrics = {"prec1": prec1, "loss": totall_l}
        metrics = utils.reduce_metrics(metrics, config.distributed)
        meters.update(metrics)

        if main_proc and (step % config.log_frequency == 0 or step + 1 == len(train_loader)):
            logger.info("Epoch [%d/%d] Step [%d/%d]  %s", epoch + 1, config.epochs, step + 1, steps, meters)

    if main_proc:
        logger.info("Train: [%d/%d] Final Prec@1 %.4f", epoch + 1, config.epochs, meters.prec1.avg)


def validate(logger, config, valid_loader, model, criterion, epoch, main_proc):
    meters = AverageMeterGroup()
    model.eval()

    with torch.no_grad():
        for step, (x, y) in enumerate(valid_loader):
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            logits, _ = model(x)
            loss = criterion(logits, y)
            prec1,prec1  = utils.accuracy(logits, y, topk=(1, 1))
            metrics = {"prec1": prec1, "loss": loss}
            metrics = utils.reduce_metrics(metrics, config.distributed)
            meters.update(metrics)

            if main_proc and (step % config.log_frequency == 0 or step + 1 == len(valid_loader)):
                torch.save(model, 'model' + str(epoch) + '.pt')
                logger.info("Epoch [%d/%d] Step [%d/%d]  %s", epoch + 1, config.epochs, step + 1, len(valid_loader), meters)

    if main_proc:
        torch.save(model, 'model_final' + '.pt')
        logger.info("Train: [%d/%d] Final Prec@1 %.4f", epoch + 1, config.epochs, meters.prec1.avg)
    return meters.prec1.avg


def main():
    config = RetrainConfig()
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

    loaders, samplers = get_augment_datasets(config)
    train_loader, valid_loader = loaders
    train_sampler, valid_sampler = samplers
    train_loader = CyclicIterator(train_loader, train_sampler)
    #valid_loader = CyclicIterator(valid_loader, valid_sampler, False)

    model = Model(config.dataset, config.layers, in_channels=config.input_channels, channels=config.init_channels, retrain=True).cuda()
    if config.label_smooth > 0:
        criterion = utils.CrossEntropyLabelSmooth(config.n_classes, config.label_smooth)
    else:
        criterion = nn.CrossEntropyLoss()

    fixed_arc_path = os.path.join('', config.arc_checkpoint)
    with open(fixed_arc_path, "r") as f:
        fixed_arc = json.load(f)
    fixed_arc = utils.encode_tensor(fixed_arc, torch.device("cuda"))
    genotypes = utils.parse_results(fixed_arc, n_nodes=4)
    genotypes_dict = {i: genotypes for i in range(3)}
    apply_fixed_architecture(model, fixed_arc_path)
    param_size = utils.param_size(model, criterion,  [3, 224, 224])

    if main_proc:
        logger.info("Param size: %.6f", param_size)
        logger.info("Genotype: %s", genotypes)

    # change training hyper parameters according to cell type
    if 'cifar' in config.dataset:
        if param_size < 3.0:
            config.weight_decay = 3e-4
            config.drop_path_prob = 0.2
        elif 3.0 < param_size < 3.5:
            config.weight_decay = 3e-4
            config.drop_path_prob = 0.3
        else:
            config.weight_decay = 5e-4
            config.drop_path_prob = 0.3

    if config.distributed:
        apex.parallel.convert_syncbn_model(model)
        model = DistributedDataParallel(model, delay_allreduce=True)

    optimizer = torch.optim.SGD(model.parameters(), config.lr, momentum=config.momentum, weight_decay=config.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config.epochs, eta_min=1E-6)

    best_top1 = best_top5 = 0.
    for epoch in range(config.epochs):
        drop_prob = config.drop_path_prob * epoch / config.epochs
        if config.distributed:
            model.module.drop_path_prob(drop_prob)
        else:
            model.drop_path_prob(drop_prob)
        # training
        if config.distributed:
            train_sampler.set_epoch(epoch)
        train(logger, config, train_loader, model, optimizer, criterion, epoch, main_proc)

        # validation
        top1 = validate(logger, config, valid_loader, model, criterion, epoch, main_proc)
        best_top1 = max(best_top1, top1)
        #best_top5 = max(best_top5, top5)
        lr_scheduler.step()

    logger.info("Final best Prec@1 = %.4f", best_top1)


if __name__ == "__main__":
    main()
