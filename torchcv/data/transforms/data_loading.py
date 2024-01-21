import cv2
import numpy as np
from typing import Optional, List, Tuple
from pathlib import Path
from minio import Minio

from detectron2.structures import PolygonMasks

from .base import BaseTransform
from torchcv.structures.instance import Instances
from torchcv.structures.ops import segments2boxes

class LoadImageFromFile(BaseTransform):
    """Load an image from file.

    Required Keys:

    - img_path

    Modified Keys:

    - img
    - img_shape
    - ori_shape

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes`.
            Defaults to 'color'.
        imdecode_backend (str): The image decoding backend type. The backend
            argument for :func:`mmcv.imfrombytes`.
            See :func:`mmcv.imfrombytes` for details.
            Defaults to 'cv2'.
        file_client_args (dict, optional): Arguments to instantiate a
            FileClient. See :class:`mmengine.fileio.FileClient` for details.
            Defaults to None. It will be deprecated in future. Please use
            ``backend_args`` instead.
            Deprecated in version 2.0.0rc4.
        ignore_empty (bool): Whether to allow loading empty image or file path
            not existent. Defaults to False.
        backend_args (dict, optional): Instantiates the corresponding file
            backend. It may contain `backend` key to specify the file
            backend. If it contains, the file backend corresponding to this
            value will be used and initialized with the remaining values,
            otherwise the corresponding file backend will be selected
            based on the prefix of the file path. Defaults to None.
            New in version 2.0.0rc4.
    """

    def __init__(self,
                 to_float32: bool = False) -> None:
        self.to_float32 = to_float32
        
    def transform(self, results: dict) -> Optional[dict]:
        """Functions to load image.

        Args:
            results (dict): Result dict from
                :class:`mmengine.dataset.BaseDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        image_file = Path(results['img_path']) / results['file_name']
        if not image_file.exists():
            image_file = Path(results['img_path']) / Path(results['file_name']).name
            if not image_file.exists():
                raise FileExistsError(f"Image id {results['img_id']}: "
                                      f"{results['file_name']} is not found!!!")
            
        img = cv2.imread(str(image_file))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.to_float32:
            img = img.astype(np.float32)

        assert results['width'] == img.shape[1] and results['height'] == img.shape[0], \
               f"Mismatched image{(results['img_id'])} shape, got {img.shape[:2]}, "\
               f"expect {(results['height'], results['width'])}."
        
        results['img'] = img
        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'to_float32={self.to_float32},')

        return repr_str
    
class LoadImageFromMinIO(BaseTransform):
    def __init__(self,
                 bucket_name: str = None,
                 to_float32: bool = False) -> None:
        self.bucket_name = bucket_name
        self.to_float32 = to_float32
        self.MinIOClient = Minio(
            "ossapi.cowarobot.cn:9000",
            access_key="perception-user",
            secret_key="h9qdK5F3PT",
            region="shjd-oss",
            secure=False
        )
    
    def transform(self, results: dict) -> Optional[dict]:
        """Functions to load image.

        Args:
            results (dict): Result dict from
                :class:`mmengine.dataset.BaseDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        img_name = Path(results['file_name']).name
        
            # try:
        if self.bucket_name == "ai-roadabnormal-zhj":
            bucket_name = self.bucket_name
        else:
            if "Time" in img_name and "Cam" in img_name:
                bucket_name = 'ai-roaddetect-panorama-zhj'
            elif "panorama" in img_name:
                bucket_name = 'ai-roaddetect-panorama-zhj'
            elif "surround" in img_name:
                bucket_name = 'ai-roaddetect-surround-zhj'
            else:
                bucket_name = self.bucket_name
        for i in range(20):
            try:
                img_bytes = self.MinIOClient.get_object(bucket_name, img_name).data
                buffer = np.frombuffer(img_bytes, dtype=np.uint8)
                img = cv2.imdecode(buffer, flags=cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                if img:
                    break
            except:
                pass
        
        assert img, f"Image id {results['img_id']}: {results['file_name']} is None!!!"
        if self.to_float32:
            img = img.astype(np.float32)

        assert results['width'] == img.shape[1] and results['height'] == img.shape[0], \
               f"Mismatched image{(results['img_id'])} shape, got {img.shape[:2]}, "\
               f"expect {(results['height'], results['width'])}."
        
        results['img'] = img
        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]
        return results

class LoadYOLOv8Annotations(BaseTransform):
    def __init__(self, 
                 with_bbox: bool = True,
                 with_label: bool = True,
                 with_mask: bool = False,
                 mask2bbox: bool = False,
                 merge_polygons: bool = True) -> None:
        super().__init__()
        self.with_bbox = with_bbox
        self.with_label = with_label
        self.with_mask = with_mask
        self.mask2bbox = mask2bbox
        self.merge_polygons = merge_polygons
        
    def transform(self, results: dict) -> dict:
        instances = results.get('instances', [])
        gt_labels, gt_bboxes, gt_masks, gt_keypoints = [], [], [], None
        if instances:
            for instance in instances:
                gt_labels.append(instance['label'])
                gt_bboxes.append(instance['bbox'])
                if 'segmentation' in instance:
                    gt_mask = instance['segmentation']
                    if len(gt_mask) > 1 and self.merge_polygons:
                        gt_mask = self.merge_multi_segment(gt_mask)
                    gt_masks.append(np.array(gt_mask[0]).reshape(-1, 2))
            gt_labels = np.array(gt_labels, dtype=np.float32).reshape((-1, 1))
            if gt_masks and self.mask2bbox:
                gt_bboxes = segments2boxes(gt_masks)
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32).reshape((-1, 4))
        else:
            gt_labels = np.zeros((0, 1), dtype=np.float32)
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_masks = []
        results["cls"] = gt_labels
        results["instances"] = Instances(gt_bboxes,
                                         gt_masks, 
                                         gt_keypoints,
                                         bbox_format="xyxy",
                                         normalized=False)
        h, w = results['img'].shape[:2]
        results["instances"].normalize(w, h)
        return results
    
    def merge_multi_segment(self,
                            gt_masks: List[List]) -> List[np.ndarray]:
        """Merge multi segments to one list.

        Find the coordinates with min distance between each segment,
        then connect these coordinates with one thin line to merge all
        segments into one.
        Args:
            gt_masks(List(np.array)):
                original segmentations in coco's json file.
                like [segmentation1, segmentation2,...],
                each segmentation is a list of coordinates.
        Return:
            gt_masks(List(np.array)): merged gt_masks
        """
        s = []
        segments = [np.array(i).reshape(-1, 2) for i in gt_masks]
        idx_list = [[] for _ in range(len(gt_masks))]

        # record the indexes with min distance between each segment
        for i in range(1, len(segments)):
            idx1, idx2 = self.min_index(segments[i - 1], segments[i])
            idx_list[i - 1].append(idx1)
            idx_list[i].append(idx2)

        # use two round to connect all the segments
        # first round: first to end, i.e. A->B(partial)->C
        # second round: end to first, i.e. C->B(remaining)-A
        for k in range(2):
            # forward first round
            if k == 0:
                for i, idx in enumerate(idx_list):
                    # middle segments have two indexes
                    # reverse the index of middle segments
                    if len(idx) == 2 and idx[0] > idx[1]:
                        idx = idx[::-1]
                        segments[i] = segments[i][::-1, :]
                    # add the idx[0] point for connect next segment
                    segments[i] = np.roll(segments[i], -idx[0], axis=0)
                    segments[i] = np.concatenate(
                        [segments[i], segments[i][:1]])
                    # deal with the first segment and the last one
                    if i in [0, len(idx_list) - 1]:
                        s.append(segments[i])
                    # deal with the middle segment
                    # Note that in the first round, only partial segment
                    # are appended.
                    else:
                        idx = [0, idx[1] - idx[0]]
                        s.append(segments[i][idx[0]:idx[1] + 1])
            # forward second round
            else:
                for i in range(len(idx_list) - 1, -1, -1):
                    # deal with the middle segment
                    # append the remaining points
                    if i not in [0, len(idx_list) - 1]:
                        idx = idx_list[i]
                        nidx = abs(idx[1] - idx[0])
                        s.append(segments[i][nidx:])
        return [np.concatenate(s).reshape(-1, ).tolist()]

    def min_index(self, arr1: np.ndarray, arr2: np.ndarray) -> Tuple[int, int]:
        """Find a pair of indexes with the shortest distance.

        Args:
            arr1: (N, 2).
            arr2: (M, 2).
        Return:
            tuple: a pair of indexes.
        """
        dis = ((arr1[:, None, :] - arr2[None, :, :])**2).sum(-1)
        return np.unravel_index(np.argmin(dis, axis=None), dis.shape)

    
class LoadDetAnnotations(BaseTransform):
    """Because the yolo series does not need to consider ignore bboxes for the
    time being, in order to speed up the pipeline, it can be excluded in
    advance.

    Args:
        mask2bbox (bool): Whether to use mask annotation to get bbox.
            Defaults to False.
        poly2mask (bool): Whether to transform the polygons to bitmaps.
            Defaults to False.
        merge_polygons (bool): Whether to merge polygons into one polygon.
            If merged, the storage structure is simpler and training is more
            effcient, especially if the mask inside a bbox is divided into
            multiple polygons. Defaults to True.
    """

    def __init__(self,
                 with_bbox: bool = True,
                 with_label: bool = True,
                 with_mask: bool = False,
                 mask2bbox: bool = False,
                 merge_polygons: bool = True,
                 **kwargs):
        super().__init__()
        self.with_bbox = with_bbox
        self.with_label = with_label
        self.with_mask = with_mask
        self.mask2bbox = mask2bbox
        self.merge_polygons = merge_polygons

    def transform(self, results: dict) -> dict:
        """Function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:``mmengine.BaseDataset``.

        Returns:
            dict: The dict contains loaded bounding box, label and
            semantic segmentation.
        """
        if self.with_bbox:
            self._load_bboxes(results)
        if self.with_label:
            self._load_labels(results)
        if self.with_mask:
            self._load_masks(results)
            if self.mask2bbox:
                gt_bboxes = results['gt_masks'].get_bboxes(dst_type='hbox')
                results['gt_bboxes'] = gt_bboxes
        return results

    def _load_bboxes(self, results: dict):
        """Private function to load bounding box annotations.
        Note: BBoxes with ignore_flag of 1 is not considered.
        Args:
            results (dict): Result dict from :obj:``mmengine.BaseDataset``.

        Returns:
            dict: The dict contains loaded bounding box annotations.
        """
        gt_bboxes = []
        for instance in results.get('instances', []):
            gt_bboxes.append(instance['bbox'])

        results['gt_bboxes_type'] = 'xyxy'
        if gt_bboxes:
            results['gt_bboxes'] = np.array(
                    gt_bboxes, dtype=np.float32).reshape((-1, 4))
        else:
            results['gt_bboxes'] = np.zeros((0, 4))

    def _load_labels(self, results: dict):
        """Private function to load label annotations.

        Note: BBoxes with ignore_flag of 1 is not considered.
        Args:
            results (dict): Result dict from :obj:``mmengine.BaseDataset``.
        Returns:
            dict: The dict contains loaded label annotations.
        """
        gt_bboxes_labels = []
        for instance in results.get('instances', []):
            gt_bboxes_labels.append(instance['bbox_label'])
        results['gt_bboxes_labels'] = np.array(
            gt_bboxes_labels, dtype=np.int64)

    def _load_masks(self, results: dict) -> None:
        """Private function to load segmentation annotations.

        Args:
            results (dict): Result dict from :obj:``mmengine.BaseDataset``.
        """
        gt_masks = []
        for instance in results.get('instances', []):
            if 'segmentation' in instance:
                gt_mask = instance['segmentation']
                if isinstance(gt_mask, list):
                    gt_mask = [
                        np.array(polygon) for polygon in gt_mask
                        if len(polygon) % 2 == 0 and len(polygon) >= 6
                    ]
                    if len(gt_mask) > 1 and self.merge_polygons:
                        gt_mask = self.merge_multi_segment(gt_mask)
                    gt_masks.append(gt_mask)
                else:
                    raise NotImplementedError(
                        'Only supports mask annotations in polygon '
                        'format currently')

        h, w = results['ori_shape']
        gt_masks = PolygonMasks([segmentation for segmentation in gt_masks])
        results['gt_masks'] = gt_masks

    def merge_multi_segment(self,
                            gt_masks: List[np.ndarray]) -> List[np.ndarray]:
        """Merge multi segments to one list.

        Find the coordinates with min distance between each segment,
        then connect these coordinates with one thin line to merge all
        segments into one.
        Args:
            gt_masks(List(np.array)):
                original segmentations in coco's json file.
                like [segmentation1, segmentation2,...],
                each segmentation is a list of coordinates.
        Return:
            gt_masks(List(np.array)): merged gt_masks
        """
        s = []
        segments = [np.array(i).reshape(-1, 2) for i in gt_masks]
        idx_list = [[] for _ in range(len(gt_masks))]

        # record the indexes with min distance between each segment
        for i in range(1, len(segments)):
            idx1, idx2 = self.min_index(segments[i - 1], segments[i])
            idx_list[i - 1].append(idx1)
            idx_list[i].append(idx2)

        # use two round to connect all the segments
        # first round: first to end, i.e. A->B(partial)->C
        # second round: end to first, i.e. C->B(remaining)-A
        for k in range(2):
            # forward first round
            if k == 0:
                for i, idx in enumerate(idx_list):
                    # middle segments have two indexes
                    # reverse the index of middle segments
                    if len(idx) == 2 and idx[0] > idx[1]:
                        idx = idx[::-1]
                        segments[i] = segments[i][::-1, :]
                    # add the idx[0] point for connect next segment
                    segments[i] = np.roll(segments[i], -idx[0], axis=0)
                    segments[i] = np.concatenate(
                        [segments[i], segments[i][:1]])
                    # deal with the first segment and the last one
                    if i in [0, len(idx_list) - 1]:
                        s.append(segments[i])
                    # deal with the middle segment
                    # Note that in the first round, only partial segment
                    # are appended.
                    else:
                        idx = [0, idx[1] - idx[0]]
                        s.append(segments[i][idx[0]:idx[1] + 1])
            # forward second round
            else:
                for i in range(len(idx_list) - 1, -1, -1):
                    # deal with the middle segment
                    # append the remaining points
                    if i not in [0, len(idx_list) - 1]:
                        idx = idx_list[i]
                        nidx = abs(idx[1] - idx[0])
                        s.append(segments[i][nidx:])
        return [np.concatenate(s).reshape(-1, )]

    def min_index(self, arr1: np.ndarray, arr2: np.ndarray) -> Tuple[int, int]:
        """Find a pair of indexes with the shortest distance.

        Args:
            arr1: (N, 2).
            arr2: (M, 2).
        Return:
            tuple: a pair of indexes.
        """
        dis = ((arr1[:, None, :] - arr2[None, :, :])**2).sum(-1)
        return np.unravel_index(np.argmin(dis, axis=None), dis.shape)

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(with_bbox={self.with_bbox}, '
        repr_str += f'with_label={self.with_label}, '
        repr_str += f'with_mask={self.with_mask}, '
        repr_str += f'with_seg={self.with_seg}, '
        repr_str += f'mask2bbox={self.mask2bbox}, '
        repr_str += f'poly2mask={self.poly2mask}, '
        repr_str += f"imdecode_backend='{self.imdecode_backend}', "
        repr_str += f'backend_args={self.backend_args})'
        return repr_str