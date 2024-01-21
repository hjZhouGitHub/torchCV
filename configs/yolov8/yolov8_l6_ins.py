from torchcv.config import get_config
from ..models.dino_r50 import model

# get default config
dataloader = get_config("configs/commoncoco.py").dataloader
optimizer = get_config("configs/common/optim.py").AdamW
lr_multiplier = get_config("configs/common/schedule.py").lr_multiplier_12ep

train = dict(
    # Directory where output files are written to
    output_dir="./output",
    # The initialize checkpoint to be loaded
    init_checkpoint="",
    # The total training iterations
    max_iter=90000,
    # options for Automatic Mixed Precision
    amp=dict(enabled=False),
    # options for DistributedDataParallel
    ddp=dict(
        broadcast_buffers=False,
        find_unused_parameters=False,
        fp16_compression=False,
    ),
    # options for Gradient Clipping during training
    clip_grad=dict(
        enabled=False,
        params=dict(
            max_norm=0.1,
            norm_type=2,
        ),
    ),
    # training seed
    seed=-1,
    # options for Fast Debugging
    fast_dev_run=dict(enabled=False),
    # options for PeriodicCheckpointer, which saves a model checkpoint
    # after every `checkpointer.period` iterations,
    # and only `checkpointer.max_to_keep` number of checkpoint will be kept.
    checkpointer=dict(period=5000, max_to_keep=100),
    # run evaluation after every `eval_period` number of iterations
    eval_period=5000,
    # output log to console every `log_period` number of iterations.
    log_period=20,
    # logging training info to Wandb
    # note that you should add wandb writer in `train_net.py``
    wandb=dict(
        enabled=False,
        params=dict(
            dir="./wandb_output",
            project="detrex",
            name="detrex_experiment",
        )
    ),
    # model ema
    model_ema=dict(
        enabled=False,
        decay=0.999,
        device="",
        use_ema_weights_for_eval_only=False,
    ),
    # the training device, choose from {"cuda", "cpu"}
    device="cuda",
    # ...
)

# modify training config
train.init_checkpoint = "detectron2://ImageNetPretrained/torchvision/R-50.pkl"
train.output_dir = "./output/dino_r50_4scale_12ep"

# max training iterations
train.max_iter = 90000
train.eval_period = 5000
train.log_period = 20
train.checkpointer.period = 5000

# gradient clipping for training
train.clip_grad.enabled = True
train.clip_grad.params.max_norm = 0.1
train.clip_grad.params.norm_type = 2

# set training devices
train.device = "cuda"
model.device = train.device

# modify optimizer config
optimizer.lr = 1e-4
optimizer.betas = (0.9, 0.999)
optimizer.weight_decay = 1e-4
optimizer.params.lr_factor_func = lambda module_name: 0.1 if "backbone" in module_name else 1

# modify dataloader config
dataloader.train.num_workers = 16

# please notice that this is total batch size.
# surpose you're using 4 gpus for training and the batch size for
# each gpu is 16/4 = 4
dataloader.train.total_batch_size = 16

# dump the testing results into output_dir for visualization
dataloader.evaluator.output_dir = train.output_dir
