import os
import torch
import datetime
from loguru import logger
from omegaconf import OmegaConf

from detectron2.config import LazyConfig
from detectron2.utils.env import seed_all_rng
from detectron2.utils.collect_env import collect_env_info

from torchcv.utils import (setup_logger, 
                           get_rank, get_world_size, is_main_process,
                           configure_nccl, configure_omp)


def _try_get_key(cfg, *keys, default=None):
    """
    Try select keys from cfg until the first key that exists. Otherwise return default.
    """
    for k in keys:
        none = object()
        p = OmegaConf.select(cfg, k, default=none)
        if p is not none:
            return p
    return default

def get_config(config_path):
    """
    Returns a config object from a config_path.

    Args:
        config_path (str): config file name relative to detrex's "configs/"
            directory, e.g., "common/train.py"

    Returns:
        omegaconf.DictConfig: a config object
    """
    if not os.path.exists(config_path):
        raise RuntimeError("{} not available in configs!".format(config_path))
    cfg = LazyConfig.load(config_path)
    return cfg

def merge_args(cfg, args):
    """
    Perform some basic common setups at the beginning of a job, including:

    1. Set up the detectron2 logger
    2. Log basic information about environment, cmdline arguments, and config
    3. Backup the config to the output directory

    Args:
        cfg (CfgNode or omegaconf.DictConfig): the full config to be used
        args (argparse.NameSpace): the command line arguments to be logged
    """
    if args.output_dir:
        cfg.train.output_dir = args.output_dir
    if args.resume:
        cfg.train.resume = args.resume
    if args.weights:
        cfg.train.weights = args.weights
    if args.occupy:
        cfg.train.occupy = args.occupy
        
def setup_info_before_train(cfg, args):
    merge_args(cfg, args)
    
    if is_main_process():
        os.makedirs(cfg.train.output_dir, exist_ok=True)
    
    rank = get_rank()
    today = datetime.date.today().strftime('%y%m%d')
    setup_logger(
        cfg.train.output_dir,
        distributed_rank=rank,
        filename=f"train_log_{today}.txt",
        mode="a",
    )
    logger.info("Rank of current process: {}. World size: {}".format(rank, get_world_size()))
    logger.info("Environment info:\n" + collect_env_info())
    
    logger.info("Command line arguments: " + str(args))
    if is_main_process():
        path = os.path.join(cfg.train.output_dir, "config.yaml")
        LazyConfig.save(cfg, path)
        logger.info("Full config saved to {}".format(path))
    
    logger.info(f"Fix seed: {cfg.train.seed}")
    seed_all_rng(cfg.train.seed)
    configure_nccl()
    configure_omp()
    torch.backends.cudnn.benchmark = True
    