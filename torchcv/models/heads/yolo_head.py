#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import math
from loguru import logger

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Sequence, Tuple, Union

from ..utils import make_divisible

class YOLOv8DetectHead(nn.Module):
    
    def __init__(self,
                 num_classes: int,
                 in_channels: Sequence[int],
                 widen_factor: float = 1.0,
                 num_base_priors: int = 3,
                 featmap_strides: Sequence[int] = (8, 16, 32),
                 ):
        super().__init__()
        self.num_classes = num_classes
        self.widen_factor = widen_factor
        self.num_levels = len(in_channels)
        self.in_channels = [make_divisible(i, widen_factor) for i in in_channels]
        self.featmap_strides = featmap_strides
        self.num_out_attrib = 5 + self.num_classes
        self.reg_max = 16

        self._init_layers()

    def init_weights(self):
        """Initialize the bias of YOLOv5 head."""
        for mi, s in zip(self.convs_pred, self.featmap_strides):  # from
            b = mi.bias.data.view(3, -1)
            # obj (8 objects per 640 image)
            b.data[:, 4] += math.log(8 / (640 / s)**2)
            b.data[:, 5:] += math.log(0.6 / (self.num_classes - 0.999999))

            mi.bias.data = b.view(-1)
    
    def _init_layers(self):
        """initialize conv layers in YOLOv5 head."""
        self.convs_pred = nn.ModuleList()
        for i in range(self.num_levels):
            conv_pred = nn.Conv2d(self.in_channels[i],
                                  self.num_base_priors * self.num_out_attrib,
                                  1)

            self.convs_pred.append(conv_pred)
            
    def forward(self, x: Tuple[torch.Tensor],
                targets=None, raw_data=None):
        """Forward features from the upstream network.

        Args:
            x (Tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
        Returns:
            Tuple[List]: A tuple of multi-level classification scores, bbox
            predictions, and objectnesses.
        """
        assert len(x) == self.num_levels
        for i in range(self.num_levels):
            x[i] = self.convs_pred[i](x[i])
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.num_base_priors, self.num_out_attrib, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
        
            if not self.training:
                pass
        if self.training:
            return self.get_losses(x, targets)
        else:
            pass
    