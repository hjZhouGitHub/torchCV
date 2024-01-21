import uuid
import random
import numpy as np

import torch
import torch.utils.data as torchdata

def build_coco_train_loader(dataset, sampler=None, collate_fn=None, is_distributed=True, **dataloader_cfg):
    if sampler is None:
        if is_distributed:
            sampler = torchdata.distributed.DistributedSampler(
                dataset, shuffle=True
            )
        else:
            sampler = torchdata.RandomSampler(dataset)

    if collate_fn is None:
        collate_fn = torchdata.default_collate
        
    return torchdata.DataLoader(
        dataset=dataset,
        sampler=sampler,
        collate_fn=collate_fn,
        worker_init_fn=worker_init_reset_seed,
        **dataloader_cfg)

def build_coco_test_loader(dataset, sampler=None, collate_fn=None, is_distributed=True, **dataloader_cfg):
    if sampler is None:
        if is_distributed:
            sampler = torchdata.distributed.DistributedSampler(
                dataset, shuffle=False
            )
        else:
            sampler = torchdata.SequentialSampler(dataset)

    if collate_fn is None:
        collate_fn = torchdata.default_collate
        
    return torchdata.DataLoader(
        dataset=dataset,
        sampler=sampler,
        collate_fn=collate_fn,
        **dataloader_cfg)



def default_collate_fn(batch):
    pass

def list_collate(batch):
    """
    Function that collates lists or tuples together into one list (of lists/tuples).
    Use this as the collate function in a Dataloader, if you want to have a list of
    items as an output, as opposed to tensors (eg. Brambox.boxes).
    """
    items = list(zip(*batch))

    for i in range(len(items)):
        if isinstance(items[i][0], (list, tuple)):
            items[i] = list(items[i])
        else:
            items[i] = torchdata.default_collate(items[i])

    return items

def worker_init_reset_seed(worker_id):
    seed = uuid.uuid4().int % 2**32
    random.seed(seed)
    torch.set_rng_state(torch.manual_seed(seed).get_state())
    np.random.seed(seed)