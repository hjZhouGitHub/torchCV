import torch

from detectron2.config import LazyCall as L

from torchcv.solver import get_yolo_optimizer_params, build_optimizer

SGD = L(torch.optim.SGD)(
    params=L(get_yolo_optimizer_params)(
        # params.model is meant to be set to the model object, before instantiating
        # the optimizer.
    ),
    lr=0.02,
    momentum=0.9,
    weight_decay=1e-4,
    nesterov=True,
)


AdamW = L(torch.optim.AdamW)(
    params=L(get_yolo_optimizer_params)(
        # params.model is meant to be set to the model object, before instantiating
        # the optimizer.
    ),
    lr=1e-4,
    betas=(0.9, 0.999),
    weight_decay=0.1,
)


optimizer = L(build_optimizer)(
    name="SGD",
    lr=0.01,
    momentum=0.9,
    weight_decay=1e-4,
)