from typing import Any, Dict, List
from loguru import logger
from torch import nn, optim
from torchcv.utils.utils import colorstr

def build_optimizer(model, name="auto", lr=0.001, momentum=0.9, decay=1e-5, iterations=1e5):
    """
    Constructs an optimizer for the given model, based on the specified optimizer name, learning rate, momentum,
    weight decay, and number of iterations.

    Args:
        model (torch.nn.Module): The model for which to build an optimizer.
        name (str, optional): The name of the optimizer to use. If 'auto', the optimizer is selected
            based on the number of iterations. Default: 'auto'.
        lr (float, optional): The learning rate for the optimizer. Default: 0.001.
        momentum (float, optional): The momentum factor for the optimizer. Default: 0.9.
        decay (float, optional): The weight decay for the optimizer. Default: 1e-5.
        iterations (float, optional): The number of iterations, which determines the optimizer if
            name is 'auto'. Default: 1e5.

    Returns:
        (torch.optim.Optimizer): The constructed optimizer.
    """

    g = [], [], []  # optimizer parameter groups
    bn = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)  # normalization layers, i.e. BatchNorm2d()
    if name == "auto":
        logger.info(
            f"{colorstr('optimizer:')} 'optimizer=auto' found, "
            f"determining best 'optimizer', 'lr0' and 'momentum' automatically... "
        )
        name, lr, momentum = ("SGD", 0.01, 0.9) if iterations > 10000 else ("AdamW", 0.001, 0.9)
        # self.args.warmup_bias_lr = 0.0  # no higher than 0.01 for Adam

    for module_name, module in model.named_modules():
        for param_name, param in module.named_parameters(recurse=False):
            fullname = f"{module_name}.{param_name}" if module_name else param_name
            if "bias" in fullname:  # bias (no decay)
                g[2].append(param)
            elif isinstance(module, bn):  # weight (no decay)
                g[1].append(param)
            else:  # weight (with decay)
                g[0].append(param)

    if name in ("Adam", "Adamax", "AdamW", "NAdam", "RAdam"):
        optimizer = getattr(optim, name, optim.Adam)(g[2], lr=lr, betas=(momentum, 0.999), weight_decay=0.0)
    elif name == "RMSProp":
        optimizer = optim.RMSprop(g[2], lr=lr, momentum=momentum)
    elif name == "SGD":
        optimizer = optim.SGD(g[2], lr=lr, momentum=momentum, nesterov=True)
    else:
        raise NotImplementedError(
            f"Optimizer '{name}' not found in list of available optimizers "
            f"[Adam, AdamW, NAdam, RAdam, RMSProp, SGD, auto]."
            "To request support for addition optimizers please visit https://github.com/ultralytics/ultralytics."
        )

    optimizer.add_param_group({"params": g[0], "weight_decay": decay})  # add g0 with weight_decay
    optimizer.add_param_group({"params": g[1], "weight_decay": 0.0})  # add g1 (BatchNorm2d weights)
    logger.info(
        f"{colorstr('optimizer:')} {type(optimizer).__name__}(lr={lr}, momentum={momentum}) with parameter groups "
        f'{len(g[1])} weight(decay=0.0), {len(g[0])} weight(decay={decay}), {len(g[2])} bias(decay=0.0)'
    )
    return optimizer

def get_yolo_optimizer_params(model: nn.Module) -> List[Dict[str, Any]]:
    params_groups = [], [], []
    norm_module_types = tuple(v for k, v in nn.__dict__.items() if 'Norm' in k)
    for v in model.modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
            params_groups[2].append(v.bias)
        # Includes SyncBatchNorm
        if isinstance(v, norm_module_types):
            params_groups[1].append(v.weight)
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
            params_groups[0].append(v.weight)
    
    # Note: Make sure bias is in the last parameter group
    optimizer_cfg = []
    optimizer_cfg.append({
        'params': params_groups[0],
    })
    optimizer_cfg.append({
        'params': params_groups[1],
        'weight_decay': 0.0
    })
    optimizer_cfg.append({
        'params': params_groups[2],
        'weight_decay': 0.0
    })
    logger.info(f"Optimizer groups: {len(params_groups[2])} .bias, "
                f"{len(params_groups[0])} conv.weight, {len(params_groups[1])} other")
    return optimizer_cfg

