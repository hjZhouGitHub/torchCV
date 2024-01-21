#!/usr/bin/env python3
# Copyright (c) Megvii, Inc. and its affiliates.

import datetime
import os
import time
from loguru import logger

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from detectron2.config import instantiate

from torchcv.data import DataPrefetcher
from torchcv.utils import (
    MeterBuffer,
    ModelEMA,
    adjust_status,
    all_reduce_norm,
    get_local_rank,
    get_model_info,
    get_rank,
    get_world_size,
    gpu_mem_usage,
    is_parallel,
    load_ckpt,
    mem_usage,
    occupy_mem,
    save_checkpoint,
    synchronize
)

class Trainer:
    def __init__(self, cfg):
        logger.info(f"Args: {cfg}")
        self.cfg = cfg
        
        # init some args
        ## train
        self.file_name = self.cfg.train.output_dir
        self.max_epoch = self.cfg.train.epoch
        self.use_model_ema = self.cfg.train.use_model_ema
        ## dist
        self.rank = get_rank()
        self.local_rank = get_local_rank()
        self.is_distributed = get_world_size() > 1
        self.device = "cuda:{}".format(self.local_rank)
        self.amp_training = self.cfg.train.fp16
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp_training)
        ## data/dataloader related attr
        self.data_type = torch.float16 if self.amp_training else torch.float32
        self.train_input_size = self.cfg.dataloader.train_info.input_size
        self.best_ap = 0
        
        ## build training loader
        self.train_input_size = self.cfg.dataloader.train_info.input_size
        self.train_loader = instantiate(self.cfg.dataloader.train)
        self.max_iter = len(self.train_loader)
        logger.info("Init prefetcher, this might take one minute or less...")
        self.prefetcher = DataPrefetcher(self.train_loader)
        
        ## create model
        torch.cuda.set_device(self.local_rank)
        model = instantiate(self.cfg.model)
        logger.info(f"Building Model: {get_model_info(model, self.train_input_size)}")
        # for name, param in model.named_parameters():
        #     logger.info(f'{name:<100s}:\t{param.requires_grad}')
        model.to(self.device)
        
        ## instantiate optimizer & scheduler
        self.last_opt_step = -1
        nbs = self.cfg.train.nominal_batch_size
        if self.cfg.lr_scheduler.iter_step_update:
            self.accumulate = 1
        else:
            self.accumulate = max(round(nbs / self.batch_size), 1)  # accumulate loss before optimizing
            self.cfg.optimizer.weight_decay *= self.batch_size * self.accumulate / nbs  # scale weight_decay
        self.cfg.optimizer.params.model = model
        self.optimizer = instantiate(self.cfg.optimizer)
        self.cfg.lr_scheduler.optimizer = self.optimizer
        self.lr_scheduler = instantiate(self.cfg.lr_scheduler)
        
        ## resume
        self.start_epoch = 0
        model = self.resume_train(model)
        if self.cfg.train.occupy:
            occupy_mem(self.local_rank)
        self.no_aug = self.start_epoch >= self.max_epoch - self.cfg.train.no_aug_epochs
        
        ## ddp
        if self.is_distributed:
            model = DDP(model, device_ids=[self.local_rank], broadcast_buffers=False, **cfg.train.ddp)
        ## build model ema
        if self.use_model_ema:
            self.ema_model = ModelEMA(model, 0.9998)
            self.ema_model.updates = self.max_iter * self.start_epoch
        self.model = model
        
        ## evaluator
        self.evaluator = instantiate(cfg.dataloader.evaluator)
        
        ## metric record
        self.meter = MeterBuffer(window_size=self.cfg.train.print_interval)
        
        ## Tensorboard and Wandb loggers
        if self.rank == 0:
            if self.cfg.train.logger == "tensorboard":
                self.tblogger = SummaryWriter(os.path.join(self.file_name, "tensorboard"))
            elif self.cfg.train.logger == "wandb":
                self.wandb_logger = None
                # self.wandb_logger = WandbLogger.initialize_wandb_logger(
                #     self.args,
                #     self.exp,
                #     self.evaluator.dataloader.dataset
                # )
            else:
                raise ValueError("logger must be either 'tensorboard' or 'wandb'")

    def train(self):
        self.before_train()
        try:
            self.train_in_epoch()
        except Exception:
            logger.exception("Exception during training:")
            raise
        finally:
            self.after_train()

    def train_in_epoch(self):
        for self.epoch in range(self.start_epoch, self.max_epoch):
            self.before_epoch()
            self.train_in_iter()
            self.after_epoch()

    def train_in_iter(self):
        for self.iter in range(self.max_iter):
            self.before_iter()
            self.train_one_iter()
            self.after_iter()

    def train_one_iter(self):
        iter_start_time = time.perf_counter()

        inps, targets = self.prefetcher.next()
        inps = inps.to(self.data_type)
        targets = targets.to(self.data_type)
        targets.requires_grad = False
        data_end_time = time.perf_counter()

        with torch.cuda.amp.autocast(enabled=self.amp_training):
            outputs = self.model(inps, targets)

        loss = outputs["total_loss"]

        self.scaler.scale(loss).backward()
        if self.progress_in_iter - self.last_opt_step >= self.accumulate:
            self.optimizer_step()
            self.last_opt_step = self.progress_in_iter

        iter_end_time = time.perf_counter()
        self.meter.update(
            iter_time=iter_end_time - iter_start_time,
            data_time=data_end_time - iter_start_time,
            lr=self.lr_scheduler.get_last_lr(),
            **outputs,
        )

    def before_train(self):
        logger.info(f"Training start ...")

    def after_train(self):
        logger.info(
            "Training of experiment is done. The best AP is {:.2f}".format(self.best_ap * 100)
        )
        logger.info(f"The best model save at {os.path.join(self.file_name, 'best.pt')}")
        if self.rank == 0:
            if self.cfg.train.logger == "wandb":
                self.wandb_logger.finish()

    def before_epoch(self):
        logger.info("---> Start train epoch{}".format(self.epoch + 1))

        if self.no_aug or self.epoch + 1 == self.max_epoch - self.cfg.train.no_aug_epochs:
            logger.info("--->No mosaic aug now!")
            self.train_loader.close_mosaic()
            logger.info("--->Add additional L1 loss now!")
            if self.is_distributed:
                self.model.module.head.use_l1 = True
            else:
                self.model.head.use_l1 = True
            self.cfg.train.eval_interval = 1
            if not self.no_aug:
                self.save_ckpt(ckpt_name="last_mosaic_epoch.pt")
            else:
                self.no_aug = False

    def after_epoch(self):
        if (self.epoch + 1) % self.cfg.train.eval_interval == 0:
            all_reduce_norm(self.model)
            self.evaluate_and_save_model()

    def before_iter(self):
        self.lr_scheduler.step(self.progress_in_iter)

    def after_iter(self):
        """
        `after_iter` contains two parts of logic:
            * log information
            * reset setting of resize
        """
        # log needed information
        if (self.iter + 1) % self.cfg.train.print_interval == 0:
            # TODO check ETA logic
            left_iters = self.max_iter * self.max_epoch - (self.progress_in_iter + 1)
            eta_seconds = self.meter["iter_time"].global_avg * left_iters
            eta_str = "ETA: {}".format(datetime.timedelta(seconds=int(eta_seconds)))

            progress_str = "epoch: {}/{}, iter: {}/{}".format(
                self.epoch + 1, self.max_epoch, self.iter + 1, self.max_iter
            )
            loss_meter = self.meter.get_filtered_meter("loss")
            loss_str = ", ".join(
                ["{}: {:.1f}".format(k, v.latest) for k, v in loss_meter.items()]
            )

            time_meter = self.meter.get_filtered_meter("time")
            time_str = ", ".join(
                ["{}: {:.3f}s".format(k, v.avg) for k, v in time_meter.items()]
            )

            mem_str = "gpu mem: {:.0f}Mb, mem: {:.1f}Gb".format(gpu_mem_usage(), mem_usage())

            logger.info(
                "{}, {}, {}, {}, lr: {:.3e}".format(
                    progress_str,
                    mem_str,
                    time_str,
                    loss_str,
                    self.meter["lr"].latest,
                )
                + (", size: {:d}, {}".format(self.train_input_size[0], eta_str))
            )

            if self.rank == 0:
                if self.cfg.train.logger == "tensorboard":
                    self.tblogger.add_scalar(
                        "train/lr", self.meter["lr"].latest, self.progress_in_iter)
                    for k, v in loss_meter.items():
                        self.tblogger.add_scalar(
                            f"train/{k}", v.latest, self.progress_in_iter)
                elif self.cfg.train.logger == "wandb":
                    metrics = {"train/" + k: v.latest for k, v in loss_meter.items()}
                    metrics.update({
                        "train/lr": self.meter["lr"].latest
                    })
                    self.wandb_logger.log_metrics(metrics, step=self.progress_in_iter)

            self.meter.clear_meters()

    @property
    def progress_in_iter(self):
        return self.epoch * self.max_iter + self.iter

    def optimizer_step(self):
        ## yolov8
        self.scaler.unscale_(self.optimizer)  # unscale gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)  # clip gradients
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        ##  yolox
        # self.scaler.step(self.optimizer)
        # self.scaler.update()
        
        if self.use_model_ema:
            self.ema_model.update(self.model)

    def resume_train(self, model):
        if self.cfg.train.resume:
            logger.info("Resume training")
            if self.cfg.train.weights is None:
                ckpt_file = os.path.join(self.file_name, "last.pt")
            else:
                ckpt_file = self.cfg.train.weights

            ckpt = torch.load(ckpt_file, map_location=self.device)
            # resume the model/optimizer state dict
            model.load_state_dict(ckpt["model"])
            self.optimizer.load_state_dict(ckpt["optimizer"])
            self.best_ap = ckpt.pop("best_ap", 0)
            # resume the training states variables
            self.start_epoch = ckpt.pop("start_epoch", 0)
            logger.info(
                f"Loaded checkpoint '{ckpt_file}' (epoch {self.start_epoch})")
        else:
            if self.cfg.train.weights is not None:
                ckpt_file = self.cfg.train.weights
                ckpt = torch.load(ckpt_file, map_location=self.device)
                model = load_ckpt(model, ckpt["model"])
                logger.info(f"Loading checkpoint: {ckpt_file} for fine tuning")
            self.start_epoch = 0

        return model

    def evaluate_and_save_model(self):
        if self.use_model_ema:
            evalmodel = self.ema_model.ema
        else:
            evalmodel = self.model
            if is_parallel(evalmodel):
                evalmodel = evalmodel.module

        with adjust_status(evalmodel, training=False):
            (ap50_95, ap50, summary), predictions = self.evaluator.evaluate(
                evalmodel, self.is_distributed, half=self.amp_training, return_outputs=True
            )

        update_best_ckpt = ap50_95 > self.best_ap
        self.best_ap = max(self.best_ap, ap50_95)

        if self.rank == 0:
            if self.args.logger == "tensorboard":
                self.tblogger.add_scalar("val/COCOAP50", ap50, self.epoch + 1)
                self.tblogger.add_scalar("val/COCOAP50_95", ap50_95, self.epoch + 1)
            logger.info("\n" + summary)
        synchronize()

        self.save_ckpt("last.pt", update_best_ckpt, ap=ap50_95)
        
    def save_ckpt(self, ckpt_name, update_best_ckpt=False, ap=None):
        if self.rank == 0:
            save_model = self.ema_model.ema if self.use_model_ema else self.model
            logger.info("Save weights to {}".format(self.file_name))
            ckpt_state = {
                "start_epoch": self.epoch + 1,
                "model": save_model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "best_ap": self.best_ap,
                "curr_ap": ap,
                "date": datetime.now().isoformat(),
            }
            save_checkpoint(
                ckpt_state,
                update_best_ckpt,
                self.file_name,
                ckpt_name,
            )
