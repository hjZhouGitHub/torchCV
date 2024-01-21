#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
import os
import gc
import copy
import time
import json
import pickle
from pathlib import Path

import numpy as np
from loguru import logger
from collections import defaultdict
# from pycocotools.coco import COCO
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

from torch.utils.data import Dataset
from .transforms.base import Compose

class COCODataset(Dataset):
    """
    COCO dataset class.
    """

    def __init__(
        self,
        data_dir: str = "data/coco/",
        ann_file: str = "annotations/instances_train2017.json",
        data_prefix: str = "train2017",
        pre_transform: List[Union[dict, Callable]] = [],
        transform: List[Union[dict, Callable]] = [],
        train_transform2: List[Union[dict, Callable]] = [],
        transform_need_data_num: int = 1,
        filter_empty_gt: bool = False,
        serialize_data: bool = True,
        test_mode: bool = False,
    ):
        """
        COCO dataset initialization. Annotation data are read into memory by COCO API.
        Args:
            data_dir (str): dataset root directory
            ann_file (str): COCO json file name
            data_prefix (str): COCO data name (e.g. 'train2017' or 'val2017')
            transform: data augmentation strategy
        """
        self.data_dir = data_dir
        self.data_prefix = os.path.join(self.data_dir, data_prefix)
        self.ann_file = os.path.join(self.data_dir, ann_file)
        
        if self.ann_file and os.path.isfile(self.ann_file):
            # self.parse_coco_file(self.ann_file)
            self.data_list = self.load_data_list()
        else:
            raise FileNotFoundError(f"{self.ann_file} is not found!!!")
        
        self.show_dataset_info()
        if filter_empty_gt:
            self.data_list = self.filter_data()
        
        self.serialize_data = serialize_data
        if self.serialize_data:
            self.data_bytes, self.data_address = self._serialize_data()
        
        # Build pipeline.
        self.is_augment = not test_mode
        self.pre_transform = Compose(pre_transform)
        self.pipeline = Compose(transform)
        self.pipeline2 = Compose(train_transform2)
        self.transform_need_data_num = transform_need_data_num
        self.test_mode = test_mode
        
    def __len__(self) -> int:
        if self.serialize_data:
            return len(self.data_address)
        else:
            return len(self.data_list)

    def parse_coco_file(self, coco_json) -> None:
        logger.info(f'Loading {coco_json} into memory...')
        tic = time.perf_counter()
        with open(coco_json, 'r') as f:
            dataset = json.load(f)
        assert type(dataset)==dict, 'annotation file format {} not supported'.format(type(dataset))
        logger.info(f'Done (t={time.perf_counter()- tic:0.2f}s)')
        
        if 'annotations' not in dataset or 'images' not in dataset or 'categories' not in dataset:
            raise KeyError(f"{coco_json} is wrong, annotations/images/categories not in keys!!!")
        
        anns, cats, imgs = {}, {}, {}
        imgid_to_anns, cat_to_imgids = defaultdict(list),defaultdict(list)
        for ann in dataset['annotations']:
            imgid_to_anns[ann['image_id']].append(ann)
            anns[ann['id']] = ann

        for img in dataset['images']:
            imgs[img['id']] = img

        for cat in dataset['categories']:
            cats[cat['id']] = cat

        for ann in dataset['annotations']:
            cat_to_imgids[ann['category_id']].append(ann['image_id'])

        return cats, imgs, imgid_to_anns
        
    def load_data_list(self, ) -> List[dict]:
        cats, imgs, imgid_to_anns = self.parse_coco_file(self.ann_file)
        self.catid2label = {cat_id: i for i, cat_id in enumerate(sorted(cats.keys()))}
        self.label2cate = {i: cats[cat_id] for i, cat_id in enumerate(sorted(cats.keys()))}
        
        data_list = []
        for img_id, img_info in imgs.items():
            data_info = {}
            assert(img_info["id"] == img_id)
            data_info['img_id'] = img_info["id"]
            data_info['width']  = int(img_info["width"])
            data_info['height'] = int(img_info["height"])
            data_info['img_path'] = self.data_prefix
            data_info['file_name'] = img_info["file_name"]
            
            raw_ann_info = imgid_to_anns[img_id]
            instances = []
            for ann in raw_ann_info:
                assert(ann["image_id"] == img_id)
                instance = {}

                if ann.get('ignore', False) or ann.get('iscrowd', False):
                    continue
                x1, y1, w, h = ann['bbox']
                inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
                inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
                if inter_w * inter_h == 0:
                    continue
                if ann['area'] <= 0 or w < 1 or h < 1:
                    continue
                if ann['category_id'] not in cats.keys():
                    logger.error(f"Image id-{img_id}: {ann['category_id']} not in {list(cats.keys())}")
                    raise
                bbox = [x1, y1, x1 + w, y1 + h]
                instance['bbox'] = bbox
                instance['label'] = self.catid2label[ann['category_id']]

                if ann.get('segmentation', None):
                    instance['segmentation'] = ann['segmentation']

                instances.append(instance)
            data_info['instances'] = instances
            
            data_list.append(data_info)
        logger.info('Data list created!')
        return data_list

    def filter_data(self) -> List[dict]:
        valid_data_infos = []
        for data_info in self.data_list:
            if data_info["instances"]:
                valid_data_infos.append(data_info)
        logger.info(f'Remove emtpy gt image: {len(self.data_list) - len(valid_data_infos)}')
        return valid_data_infos
    
    def show_dataset_info(self) -> None:
        total_num = len(self.data_list)
        empty_num = 0
        total_ins_num = 0
        categoty_info = {}
        for data in self.data_list:
            instances = data['instances']
            if not instances:
                empty_num += 1
                continue
            for ins in instances:
                total_ins_num += 1
                label = ins['label']
                name = self.label2cate[label]['name']
                num = categoty_info.get(name, 0)
                categoty_info[name] = num + 1
        logger.info(f"Loaded {total_num} images from {Path(self.ann_file).name}, empty num is {empty_num}.")
        logger.info(f"Distribution of {total_ins_num} instances among all {len(self.label2cate)} categories:")
        
        message = f"{'*'*106}\n"
        length = 15
        for label, cate in self.label2cate.items():
            name = cate['name'][:length] + "..." if len(cate['name']) > length else cate['name']
            message += f"{name:<20s}  {np.random.randint(1, 100000):>8d}  {' '*5}"
            if (label+1) % 3 == 0:
                message += "\n"
        message += f"\n{'*'*106}"
        logger.info(message)
        
    def _serialize_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Serialize ``self.data_list`` to save memory when launching multiple
        workers in data loading. This function will be called in ``full_init``.

        Hold memory using serialized objects, and data loader workers can use
        shared RAM from master process instead of making a copy.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Serialized result and corresponding
            address.
        """

        def _serialize(data):
            buffer = pickle.dumps(data, protocol=4)
            return np.frombuffer(buffer, dtype=np.uint8)

        # Serialize data information list avoid making multiple copies of
        # `self.data_list` when iterate `import torch.utils.data.dataloader`
        # with multiple workers.
        logger.info(f"Serializing {len(self.data_list)} elements to byte tensors and concatenating them all ...")
        data_list = [_serialize(x) for x in self.data_list]
        address_list = np.asarray([len(x) for x in data_list], dtype=np.int64)
        data_address: np.ndarray = np.cumsum(address_list)
        # TODO Check if np.concatenate is necessary
        data_bytes = np.concatenate(data_list)
        # Empty cache for preventing making multiple copies of
        # `self.data_info` when loading data multi-processes.
        self.data_list.clear()
        gc.collect()
        logger.info(f"Serialized dataset takes {len(data_bytes) / 1024**2:.2f} MiB")
        return data_bytes, data_address

    def __getitem__(self, idx: int) -> dict:
        """Get the idx-th image and data information of dataset after
        ``self.pipeline``, and ``full_init`` will be called if the dataset has
        not been fully initialized.

        During training phase, if ``self.pipeline`` get ``None``,
        ``self._rand_another`` will be called until a valid image is fetched or
         the maximum limit of refetech is reached.

        Args:
            idx (int): The index of self.data_list.

        Returns:
            dict: The idx-th image and data information of dataset after
            ``self.pipeline``.
        """
        # Performing full initialization by calling `__getitem__` will consume
        # extra memory. If a dataset is not fully initialized by setting
        # `lazy_init=True` and then fed into the dataloader. Different workers
        # will simultaneously read and parse the annotation. It will cost more
        # time and memory, although this may work. Therefore, it is recommended
        # to manually call `full_init` before dataset fed into dataloader to
        # ensure all workers use shared RAM from master process.
        if self.test_mode:
            data = self.prepare_data(idx)
            data = self.pipeline(data)
            if data is None:
                raise Exception('Test time pipline should not get `None` '
                                'data_sample')
            return data

        results = []
        exist_idx = []
        for _ in range(500):
            if idx in exist_idx:
                idx = self._rand_another()
                continue
            data = self.prepare_data(idx)
            # Broken images or random augmentations may cause the returned data
            # to be None
            if data is None:
                idx = self._rand_another()
                continue
            results.append(data)
            exist_idx.append(idx)
            if len(results) >= self.transform_need_data_num:
                results = self.pipeline(results)
                return results

        raise Exception(f'Cannot find valid image after {self.max_refetch}! '
                        'Please check your image path and pipeline')

    def _rand_another(self) -> int:
        """Get random index.

        Returns:
            int: Random index from 0 to ``len(self)-1``
        """
        return np.random.randint(0, len(self))
    
    def prepare_data(self, idx) -> Any:
        """Get data processed by ``self.pipeline``.

        Args:
            idx (int): The index of ``data_info``.

        Returns:
            Any: Depends on ``self.pipeline``.
        """
        if self.serialize_data:
            start_addr = 0 if idx == 0 else self.data_address[idx - 1].item()
            end_addr = self.data_address[idx].item()
            bytes = memoryview(
                self.data_bytes[start_addr:end_addr])  # type: ignore
            data_info = pickle.loads(bytes)  # type: ignore
        else:
            data_info = copy.deepcopy(self.data_list[idx])
        # Some codebase needs `sample_idx` of data information. Here we convert
        # the idx to a positive number and save it in data information.
        if idx >= 0:
            data_info['sample_idx'] = idx
        else:
            data_info['sample_idx'] = len(self) + idx
        return self.pre_transform(data_info)
    
    def close_augment(self):
        if self.is_augment:
            self.is_augment = False
            self.pipeline = self.pipeline2
            self.transform_need_data_num = 1
        
    