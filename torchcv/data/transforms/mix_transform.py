
import cv2
import random
import numpy as np

from typing import Dict, List, Optional, Tuple, Union, Sequence

from .base import BaseTransform
from torchcv.data.transforms.base import Compose
from torchcv.structures.instance import Instances

class Mosaic(BaseTransform):
    """Mosaic augmentation.
    Args:
        imgsz (Sequence[int]): Image size after mosaic pipeline of single
            image. The shape order should be (height, width).
            Default to (640, 640).
    """

    def __init__(self,
                 img_scale: Tuple[int, int] = (640, 640),
                 center_ratio_range: Tuple[float, float] = (0.5, 1.5),
                 pad_val: float = 114.0,
                 prob: float = 1.0,):
        assert 0 <= prob <= 1.0, "The probability should be in range [0, 1]. " f"got {prob}."
        self.prob = prob
        self.img_scale = img_scale
        self.center_ratio_range = center_ratio_range
        self.pad_val = pad_val

    def transform(self, results: List):
        if random.uniform(0, 1) > self.prob:
            return results[0]
        
        assert len(results) >= 4
        mosaic_labels = []
        img_scale_w, img_scale_h = self.img_scale
        if len(results['img'].shape) == 3:
            mosaic_img = np.full(
                (int(img_scale_h * 2), int(img_scale_w * 2), 3),
                self.pad_val,
                dtype=results['img'].dtype)
        else:
            mosaic_img = np.full((int(img_scale_h * 2), int(img_scale_w * 2)),
                                 self.pad_val,
                                 dtype=results['img'].dtype)
        
        # mosaic center x, y
        center_x = int(random.uniform(*self.center_ratio_range) * img_scale_w)
        center_y = int(random.uniform(*self.center_ratio_range) * img_scale_h)
        for i in range(4):
            results_patch = results[i].copy()
            # Load image
            img_i = results_patch['img']
            h_i, w_i = img_i.shape[:2]
            # keep_ratio resize
            if not results_patch['instances'].normalized:
                results_patch['instances'].normalize(w_i, h_i)
            scale_ratio_i = min(img_scale_h / h_i, img_scale_w / w_i)
            resize_img_i = cv2.resize(img_i, (int(w_i * scale_ratio_i), int(h_i * scale_ratio_i)))
            resize_h, resize_w = resize_img_i.shape[:2]
            results_patch["img"] = resize_img_i

            # place img in img4
            if i == 0:  # top left
                x1a, y1a, x2a, y2a = max(center_x - resize_w, 0), max(center_y - resize_h, 0), center_x, center_y  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = resize_w - (x2a - x1a), resize_h - (y2a - y1a), resize_w, resize_h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = center_x, max(center_y - resize_h, 0), min(center_x + resize_w, self.img_scale[0] * 2), center_y
                x1b, y1b, x2b, y2b = 0, resize_h - (y2a - y1a), min(resize_w, x2a - x1a), resize_h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(center_x - resize_w, 0), center_y, center_x, min(self.img_scale[1] * 2, center_y + resize_h)
                x1b, y1b, x2b, y2b = resize_w - (x2a - x1a), 0, resize_w, min(y2a - y1a, resize_h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = center_x, center_y, min(center_x + resize_w, self.img_scale[0] * 2), min(self.img_scale[1] * 2, center_y + resize_h)
                x1b, y1b, x2b, y2b = 0, 0, min(resize_w, x2a - x1a), min(y2a - y1a, resize_h)

            mosaic_img[y1a:y2a, x1a:x2a] = resize_img_i[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
            padw = x1a - x1b
            padh = y1a - y1b

            results_patch = self._update_labels(results_patch, padw, padh)
            mosaic_labels.append(results_patch)
        final_labels = self._cat_labels(mosaic_labels)
        final_labels["img"] = mosaic_img
        return final_labels

    def _update_labels(self, results, padw, padh):
        """Update labels"""
        nh, nw = results["img"].shape[:2]
        results["instances"].convert_bbox(format="xyxy")
        results["instances"].denormalize(nw, nh)
        results["instances"].add_padding(padw, padh)
        return results

    def _cat_labels(self, mosaic_labels):
        if len(mosaic_labels) == 0:
            return {}
        cls = []
        instances = []
        for labels in mosaic_labels:
            cls.append(labels["cls"])
            instances.append(labels["instances"])
        final_labels = {
            "ori_shape": mosaic_labels[0]["ori_shape"],
            "file_name": mosaic_labels[0]["file_name"],
            "cls": np.concatenate(cls, 0),
            "instances": Instances.concatenate(instances, axis=0)}
        final_labels["instances"].clip(self.img_scale * 2)
        return final_labels
    
class MixUp(BaseTransform):

    def __init__(self, 
                 pre_transform = [], 
                 pre_transform_num = 4, 
                 prob = 0.0) -> None:
        assert 0 <= prob <= 1.0, "The probability should be in range [0, 1]. " f"got {prob}."
        self.prob = prob
        self.pre_transform_num = pre_transform_num
        self.pre_transform = Compose(pre_transform)
    
    def transform(self, results: List):
        # Applies MixUp augmentation https://arxiv.org/pdf/1710.09412.pdf
        if random.uniform(0, 1) > self.prob:
            results = self.pre_transform(results)
            return results if isinstance(results, Dict) else results[0]
        
        assert len(results) >= 8
        r = np.random.beta(32.0, 32.0)  # mixup ratio, alpha=beta=32.0
        results1 = self.pre_transform(results[:self.pre_transform_num])
        results2 = self.pre_transform(results[self.pre_transform_num:])
        
        results1["img"] = (results1["img"] * r + results2["img"] * (1 - r)).astype(np.uint8)
        results1["instances"] = Instances.concatenate([results1["instances"], results2["instances"]], axis=0)
        results1["cls"] = np.concatenate([results1["cls"], results2["cls"]], 0)
        return results1
    