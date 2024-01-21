from omegaconf import OmegaConf
from detectron2.config import LazyCall as L

from torchcv.data import COCODataset, build_coco_train_loader, build_coco_test_loader
from torchcv.data.transforms import *

from 

# -----data related-----
data_root = 'data/coco/'  # Root path of data
is_distributed = True
batch_size_per_gpu = 2
num_workers = 4
img_size = 1280

transform_hyp = {
    
}

pre_transform = [
    L(LoadImageFromFile)(),
    L(LoadYOLOv8Annotations)(with_mask=True,
                             mask2bbox=True),
]

mosaic_transform = [
    L(MixUp)(pre_transform = [
                L(Mosaic)(
                    img_scale=(img_size, img_size), 
                    prob=transform_hyp.mosaic),
                L(CopyPaste)(
                    p=transform_hyp.copy_paste),
                L(RandomPerspective)(
                    degrees=transform_hyp.degrees,
                    translate=transform_hyp.translate,
                    scale=transform_hyp.scale,
                    shear=transform_hyp.shear,
                    perspective=transform_hyp.perspective,
                    border=[-img_size // 2, -img_size // 2],),
             ], 
             pre_transform_num=4,
             prob=transform_hyp.mixup),
]

last_transorm = [
    L(Albumentations)(p=1.0),
    L(RandomHSV)(hgain=transform_hyp.hsv_h, sgain=transform_hyp.hsv_s, vgain=transform_hyp.hsv_v),
    L(RandomFlip)(direction="vertical", p=transform_hyp.flipud),
    L(RandomFlip)(direction="horizontal", p=transform_hyp.fliplr),
    L(Format)(bbox_format="xywh",
              normalize=True,
              return_mask=True,
              return_keypoint=False,
              batch_idx=True,
              mask_ratio=transform_hyp.mask_ratio,
              mask_overlap=transform_hyp.overlap_mask,),
]

train_transform = [
    *mosaic_transform,
    *last_transorm
]

train_transform2 = [
    L(LetterBox)(new_shape=(img_size, img_size)),
    L(RandomPerspective)(
        degrees=transform_hyp.degrees,
        translate=transform_hyp.translate,
        scale=transform_hyp.scale,
        shear=transform_hyp.shear,
        perspective=transform_hyp.perspective,
        border=[0, 0],),
    *last_transorm
]

valid_transform = [
    L(LetterBox)(new_shape=(img_size, img_size), scaleup=False),
    L(Format)(bbox_format="xywh",
              normalize=True,
              return_mask=True,
              return_keypoint=False,
              batch_idx=True,
              mask_ratio=transform_hyp.mask_ratio,
              mask_overlap=transform_hyp.overlap_mask,),
]

dataloader = OmegaConf.create()
dataloader.train = L(build_coco_train_loader)(
    dataset=L(COCODataset)(
        data_root=data_root,
        ann_file="annotations/instances_train2017.json",
        data_prefix="train2017/",
        pre_transform=pre_transform,
        transform=train_transform,
        train_transform2=train_transform2,
        transform_need_data_num=8,
        filter_empty_gt=False,
    ),
    is_distributed=is_distributed,
    batch_size=batch_size_per_gpu,
    num_workers=num_workers,
    persistent_workers=True,
    pin_memory=True,
    drop_last=True,
)

dataloader.test = L(build_coco_test_loader)(
    dataset=L(COCODataset)(
        data_root=data_root,
        ann_file="annotations/instances_val2017.json",
        data_prefix="val2017/",
        pre_transform=pre_transform,
        transform=valid_transform,
        filter_empty_gt=False,
        test_mode=True,
    ),
    is_distributed=is_distributed,
    batch_size=batch_size_per_gpu*2,
    num_workers=num_workers,
    persistent_workers=True,
    pin_memory=True,
    drop_last=False,
)

from torchcv.evaluators import COCOEvaluator
dataloader.evaluator = L(COCOEvaluator)(
    dataset_name="${..test.dataset.names}",
)
