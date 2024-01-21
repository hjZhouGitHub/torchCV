from detectron2.config import LazyCall as L

from torchcv.solver import LRScheduler

cos = L(LRScheduler)(
    scheduler_type = "cos",
    final_lr_ratio = 0.01,
    total_epochs = 300,
    iters_per_epoch = 1000,
)

warmcos = L(LRScheduler)(
    scheduler_type = "warmcos",
    final_lr_ratio = 0.01,
    total_epochs = 300,
    iters_per_epoch = 1000,
    warmup_epochs = 3,
    warmup_lr_start = 1e-6,
    warmup_mim_iter = 1000,
)

yoloxwarmcos = L(LRScheduler)(
    scheduler_type = "yoloxwarmcos",
    final_lr_ratio = 0.01,
    total_epochs = 300,
    iters_per_epoch = 1000,
    warmup_epochs = 3,
    warmup_lr_start = 0,
    warmup_mim_iter = 1000,
    no_aug_epochs = 10,
)

yoloxsemiwarmcos = L(LRScheduler)(
    scheduler_type = "yoloxsemiwarmcos",
    final_lr_ratio = 0.01,
    total_epochs = 300,
    iters_per_epoch = 1000,
    warmup_epochs = 3,
    warmup_lr_start = 0.0,
    warmup_mim_iter = 1000,
    no_aug_epochs = 10,
    semi_epoch = 10,
)

multistep = L(LRScheduler)(
    scheduler_type = "multistep",
    final_lr_ratio = 0.01,
    total_epochs = 300,
    iters_per_epoch = 1000,
    milestones = [180, 240],
    gamma = 0.1,
)

yolov8_lr = L(LRScheduler)(
    scheduler_type = "yolov8_lr",
    final_lr_ratio = 0.01,
    total_epochs = 300,
    iters_per_epoch = 1000,
    warmup_epochs = 3,
    warmup_lr_start = 0.1,
    warmup_momentum = 0.8,
    warmup_mim_iter = 1000,
    scheduler_func = "linear", 
)

