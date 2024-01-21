#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import math
import numpy as np
from functools import partial
from torch.optim import Optimizer

def linear_fn(lr_factor: float, max_epochs: int):
    """Generate linear function."""
    return lambda x: (1 - x / max_epochs) * (1.0 - lr_factor) + lr_factor


def cosine_fn(lr_factor: float, max_epochs: int):
    """Generate cosine function."""
    return lambda x: (
        (1 - math.cos(x * math.pi / max_epochs)) / 2) * (lr_factor - 1) + 1


class LRScheduler:
    def __init__(self, optimizer: Optimizer,
                 scheduler_type: str = 'yolov8_lr',
                 iter_step_update: bool = True,
                 final_lr_ratio: float = 0.01,
                 total_epochs: int = 300,
                 iters_per_epoch: int = 1000,
                 warmup_epochs: int = 3,
                 warmup_lr_start: float = 0.1,
                 warmup_momentum: float = 0.8,
                 warmup_mim_iter: int = 1000,
                 **kwargs):
        """
        Supported lr schedulers: [cos, warmcos, multistep]

        Args:
            lr (float): learning rate.
            iters_per_peoch (int): number of iterations in one epoch.
            total_epochs (int): number of epochs in training.
            kwargs (dict):
                - cos: None
                - warmcos: [warmup_epochs, warmup_lr_start (default 1e-6)]
                - multistep: [milestones (epochs), gamma (default 0.1)]
        """
        self.optimizer = optimizer
        self._base_lr = [
            group['lr'] for group in optimizer.param_groups
        ]
        self._base_momentum = [
            group['momentum'] for group in optimizer.param_groups
        ]
        self._last_lr: float = self._base_lr[0]
        
        self.final_lr_ratio = final_lr_ratio
        self.iters_per_epoch = iters_per_epoch
        self.total_epochs = total_epochs
        self.total_iters = iters_per_epoch * total_epochs
        
        self.warmup_epochs = warmup_epochs
        self.warmup_lr_start = warmup_lr_start
        self.warmup_momentum = warmup_momentum
        self.warmup_total_iters = max(
            round(self.warmup_epochs * iters_per_epoch),
            warmup_mim_iter)

        self.__dict__.update(kwargs)

        self.scheduler_type = scheduler_type
        self.lr_func = self._get_lr_func(scheduler_type)
        self.iter_step_update = iter_step_update

    def get_last_lr(self):
        return self._last_lr
    
    def iter_step(self, iters):
        new_lr = self.lr_func(iters)
        if not self.iter_step_update:
            return
        if not isinstance(new_lr, list) and not isinstance(new_lr, tuple):
            new_lr = [new_lr] * len(self.optimizer.param_groups)
        for param_group, lr in zip(self.optimizer.param_groups, new_lr):
            param_group['lr'] = lr
        self._last_lr = new_lr[0]

    def _get_lr_func(self, name):
        if name == "cos":  # cosine lr schedule
            return self.cos_lr
        elif name == "warmcos":
            return self.warm_cos_lr
        elif name == "yoloxwarmcos":
            self.min_lr = self._base_lr[0] * self.final_lr_ratio
            self.no_aug_iters = self.iters_per_epoch * self.no_aug_epochs
            return self.yolox_warm_cos_lr
        elif name == "yoloxsemiwarmcos":
            self.min_lr = self._base_lr[0] * self.final_lr_ratio
            self.no_aug_iters = self.iters_per_epoch * self.no_aug_epochs
            self.normal_iters = self.iters_per_epoch * self.semi_epoch
            self.semi_iters = self.iters_per_epoch_semi * (
                self.total_epochs - self.semi_epoch - self.no_aug_epochs
            )
            return self.yolox_semi_warm_cos_lr
        elif name == "multistep":  # stepwise lr schedule
            self.milestones = [
                int(self.iters_per_epoch * milestone)
                for milestone in self.milestones
            ]
            self.gamma = getattr(self, "gamma", 0.1)
            return self.multistep_lr
        elif name == "yolov8_lr":  # stepwise lr schedule
            func_name = getattr(self, "scheduler_func", "linear")
            self.scheduler_fn = {
                "linear": linear_fn, "cosine": cosine_fn
            }[func_name](lr_factor=self.final_lr_ratio, max_epochs=self.total_epochs)
            return self.yolov8_lr
        else:
            raise ValueError("Scheduler version {} not supported.".format(name))
        

    def cos_lr(self, iters):
        """Cosine learning rate"""
        lr = self._base_lr[0] * 0.5 * (1.0 + math.cos(math.pi * iters / self.total_iters))
        return lr


    def warm_cos_lr(self, iters):
        """Cosine learning rate with warm up."""
        if iters <= self.warmup_total_iters:
            lr = (self._base_lr[0] - self.warmup_lr_start) * iters / float(
                self.warmup_total_iters
            ) + self.warmup_lr_start
        else:
            lr = self._base_lr[0] * 0.5 * (
                1.0
                + math.cos(
                    math.pi
                    * (iters - self.warmup_total_iters)
                    / (self.total_iters - self.warmup_total_iters)
                )
            )
        return lr


    def yolox_warm_cos_lr(self, iters):
        """Cosine learning rate with warm up."""
        if iters <= self.warmup_total_iters:
            # lr = (lr - warmup_lr_start) * iters / float(warmup_total_iters) + warmup_lr_start
            lr = (self._base_lr[0] - self.warmup_lr_start) * pow(
                iters / float(self.warmup_total_iters), 2
            ) + self.warmup_lr_start
        elif iters >= self.total_iters - self.no_aug_iters:
            lr = self.min_lr
        else:
            lr = self.min_lr + 0.5 * (self._base_lr[0] - self.min_lr) * (
                1.0
                + math.cos(
                    math.pi
                    * (iters - self.warmup_total_iters)
                    / (self.total_iters - self.warmup_total_iters - self.no_aug_iters)
                )
            )
        return lr


    def yolox_semi_warm_cos_lr(self, iters):
        """Cosine learning rate with warm up."""
        if iters <= self.warmup_total_iters:
            # lr = (lr - warmup_lr_start) * iters / float(warmup_total_iters) + warmup_lr_start
            lr = (self._base_lr[0] - self.warmup_lr_start) * pow(
                iters / float(self.warmup_total_iters), 2
            ) + self.warmup_lr_start
        elif iters >= self.normal_iters + self.semi_iters:
            lr = self.min_lr
        elif iters <= self.normal_iters:
            lr = self.min_lr + 0.5 * (self._base_lr[0] - self.min_lr) * (
                1.0
                + math.cos(
                    math.pi
                    * (iters - self.warmup_total_iters)
                    / (self.total_iters - self.warmup_total_iters - self.no_aug_iters)
                )
            )
        else:
            lr = self.min_lr + 0.5 * (self._base_lr[0] - self.min_lr) * (
                1.0
                + math.cos(
                    math.pi
                    * (
                        self.normal_iters
                        - self.warmup_total_iters
                        + (iters - self.normal_iters)
                        * self.iters_per_epoch
                        * 1.0
                        / self.iters_per_epoch_semi
                    )
                    / (self.total_iters - self.warmup_total_iters - self.no_aug_iters)
                )
            )
        return lr


    def multistep_lr(self, iters):
        """MultiStep learning rate"""
        lr = self._base_lr[0]
        for milestone in self.milestones:
            lr *= self.gamma if iters >= milestone else 1.0
        return lr


    def calculate_lr_factor(self, iters):
        cur_epoch = iters // self.iters_per_epoch
        lr_factor = self.scheduler_fn(cur_epoch)
        return lr_factor
    
    def yolov8_lr(self, iters):
        new_lrs = []
        if iters <= self.warmup_total_iters:
            self.iter_step_update = False
            lr_factor = self.calculate_lr_factor(iters)
            xp = [0, self.warmup_total_iters]
            for group_idx, param in enumerate(self.optimizer.param_groups):
                if group_idx == 2:
                    # bias learning rate will be handled specially
                    yp = [self.warmup_lr_start, lr * lr_factor]
                else:
                    yp = [0.0, lr * lr_factor]
                param['lr'] = np.interp(iters, xp, yp)
                if group_idx == 0:
                    self._last_lr = param['lr']
                if 'momentum' in param:
                    param['momentum'] = np.interp(
                        iters, xp,
                        [self.warmup_momentum, self._base_momentum[group_idx]])
        elif iters % self.iters_per_epoch == 0:
            self.iter_step_update = True
            lr_factor = self.calculate_lr_factor(iters)
            for lr in self._base_lr:
                new_lrs.append(lr * lr_factor)
        else:
            self.iter_step_update = False
        return new_lrs

