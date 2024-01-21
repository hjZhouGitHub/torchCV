#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import warnings
from loguru import logger

from detectron2.config import LazyConfig

import os
import sys
ROOT = os.getcwd()
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from torchcv.core import launch, Trainer
from torchcv.config import setup_info_before_train
from torchcv.utils import configure_module, get_num_devices


def make_parser():
    parser = argparse.ArgumentParser("Train parser")
    parser.add_argument(
        "-f",
        "--config-file",
        default=None,
        type=str,
        help="plz input your experiment description file",
    )
    parser.add_argument("-o", "--output_dir", default=None, type=str)
    parser.add_argument("--weights", default=None, type=str, help="checkpoint file")
    parser.add_argument("--resume", default=False, action="store_true", help="resume training")
    parser.add_argument(
        "-o",
        "--occupy",
        dest="occupy",
        default=False,
        action="store_true",
        help="occupy GPU memory first for training.",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    
    # distributed
    parser.add_argument("--devices", default=None, type=int, help="device for training")
    parser.add_argument("--num_machines", default=1, type=int, help="num of node for training")
    parser.add_argument("--machine_rank", default=0, type=int, help="node rank for multi-node training")
    parser.add_argument("--dist-backend", default="nccl", type=str, help="distributed backend")
    parser.add_argument(
        "--dist-url",
        default="tcp://127.0.0.1:",
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument("--port", default="19527", type=str, help="distributed backend")
    return parser


@logger.catch
def main(args):
    cfg = LazyConfig.load(args.config_file)
    cfg = LazyConfig.apply_overrides(cfg, args.opts)
    setup_info_before_train(cfg, args)
    
    # if args.eval_only:
    #     model = Trainer.build_model(cfg)
    #     DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
    #         cfg.MODEL.WEIGHTS, resume=args.resume
    #     )
    #     logger.info("Run evaluation under eval-only mode")
    #     if cfg.train.model_ema.enabled and cfg.train.model_ema.use_ema_weights_for_eval_only:
    #         logger.info("Run evaluation with EMA.")
    #     else:
    #         logger.info("Run evaluation without EMA.")
    #     if "evaluator" in cfg.dataloader:
    #         ret = inference_on_dataset(
    #             model, instantiate(cfg.dataloader.test), instantiate(cfg.dataloader.evaluator)
    #         )
    #         print_csv_format(ret)
    #     return ret
    
    trainer = Trainer(cfg)
    return trainer.train()


if __name__ == "__main__":
    configure_module()
    args = make_parser().parse_args()
    
    num_gpu = get_num_devices() if args.devices is None else args.devices
    assert num_gpu <= get_num_devices()

    dist_url = args.dist_url + args.port
    launch(
        main,
        num_gpu,
        args.num_machines,
        args.machine_rank,
        backend=args.dist_backend,
        dist_url=dist_url,
        args=(args,),
    )
