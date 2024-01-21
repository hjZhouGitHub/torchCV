from typing import Any, Dict, List
from loguru import logger

import torch.nn as nn

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

