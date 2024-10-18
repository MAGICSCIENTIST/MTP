import warnings
import numpy as np
from osgeo import gdal, ogr, osr, gdalconst
from typing import Dict, Optional, Union

import mmengine.fileio as fileio
import mmcv
from mmcv.transforms import BaseTransform
from mmcv.transforms import LoadAnnotations as MMCV_LoadAnnotations

from mmseg.registry import TRANSFORMS
from mmseg.registry import VISUALIZERS
from mmseg.utils import datafrombytes
from mmengine.visualization import Visualizer




@VISUALIZERS.register_module()
class SegLocalVisualizer(Visualizer):
    """Visualizer to visualize segmentation results.

    Args:
        vis_backends (list[dict]): List of visualization backends.
        name (str, optional): Name of the visualizer. Defaults to 'visualizer'.
    """

    def __init__(self, vis_backends, name='visualizer'):
        super().__init__(name)
        self.vis_backends = [build_visualizer(backend) for backend in vis_backends]

    def visualize(self, result, meta=None, **kwargs):
        """Visualize the results.

        Args:
            result (dict): The result dict contains the data to visualize.
            meta (dict): Meta information of the data.
        """
        for backend in self.vis_backends:
            backend.visualize(result, meta, **kwargs)

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'vis_backends={self.vis_backends}, name={self.name})')
    

@TRANSFORMS.register_module() 
class LoadSingleRSImageFromFile(BaseTransform): 
    """Load a Remote Sensing mage from file. 

    Required Keys: 

    - img_path 

    Modified Keys: 

    - img 
    - img_shape 
    - ori_shape 

    Args: 
        to_float32 (bool): Whether to convert the loaded image to a float32 
            numpy array. If set to False, the loaded image is a float64 array. 
            Defaults to True. 
    """ 

    def __init__(self, to_float32: bool = True): 
        self.to_float32 = to_float32 

        if gdal is None: 
            raise RuntimeError('gdal is not installed') 

    def transform(self, results: Dict) -> Dict: 
        """Functions to load image. 

        Args: 
            results (dict): Result dict from :obj:``mmcv.BaseDataset``. 

        Returns: 
            dict: The dict contains loaded image and meta information. 
        """ 

        filename = results['img_path'] 
        ds = gdal.Open(filename) 
        if ds is None: 
            raise Exception(f'Unable to open file: {filename}') 
        img = np.einsum('ijk->jki', ds.ReadAsArray()) 

        if self.to_float32: 
            img = img.astype(np.float32) 

        results['img'] = img 
        results['img_shape'] = img.shape[:2] 
        results['ori_shape'] = img.shape[:2] 
        return results 

    def __repr__(self): 
        repr_str = (f'{self.__class__.__name__}(' 
                    f'to_float32={self.to_float32})') 
        return repr_str 
    


@TRANSFORMS.register_module() 
class LoadMergeRSImageFromFile(BaseTransform): 
    """Load a Remote Sensing mage from file. 

    Required Keys: 

    - img_path 

    Modified Keys: 

    - img 
    - img_shape 
    - ori_shape 

    Args: 
        to_float32 (bool): Whether to convert the loaded image to a float32 
            numpy array. If set to False, the loaded image is a float64 array. 
            Defaults to True. 
    """ 

    def __init__(self, to_float32: bool = True): 
        self.to_float32 = to_float32 

        if gdal is None: 
            raise RuntimeError('gdal is not installed') 

    def transform(self, results: Dict) -> Dict: 
        """Functions to load image. 

        Args: 
            results (dict): Result dict from :obj:``mmcv.BaseDataset``. 

        Returns: 
            dict: The dict contains loaded image and meta information. 
        """ 

        filename = results['img_path'] 
        ds = gdal.Open(filename) 
        if ds is None: 
            raise Exception(f'Unable to open file: {filename}') 
        
        
        img = np.einsum('ijk->jki', ds.ReadAsArray()[[1,3,4],:,:])

        if self.to_float32: 
            img = img.astype(np.float32) 

        results['img'] = img 
        results['img_shape'] = img.shape[:2] 
        results['ori_shape'] = img.shape[:2] 
        return results 

    def __repr__(self): 
        repr_str = (f'{self.__class__.__name__}(' 
                    f'to_float32={self.to_float32})') 
        return repr_str 
    

@TRANSFORMS.register_module()
class LoadAnnotationsTIF(MMCV_LoadAnnotations):
    """Load annotations for semantic segmentation provided by dataset.

    The annotation format is as the following:

    .. code-block:: python

        {
            # Filename of semantic segmentation ground truth file.
            'seg_map_path': 'a/b/c'
        }

    After this module, the annotation has been changed to the format below:

    .. code-block:: python

        {
            # in str
            'seg_fields': List
             # In uint8 type.
            'gt_seg_map': np.ndarray (H, W)
        }

    Required Keys:

    - seg_map_path (str): Path of semantic segmentation ground truth file.

    Added Keys:

    - seg_fields (List)
    - gt_seg_map (np.uint8)

    Args:
        reduce_zero_label (bool, optional): Whether reduce all label value
            by 1. Usually used for datasets where 0 is background label.
            Defaults to None.
        imdecode_backend (str): The image decoding backend type. The backend
            argument for :func:``mmcv.imfrombytes``.
            See :fun:``mmcv.imfrombytes`` for details.
            Defaults to 'pillow'.
        backend_args (dict): Arguments to instantiate a file backend.
            See https://mmengine.readthedocs.io/en/latest/api/fileio.htm
            for details. Defaults to None.
            Notes: mmcv>=2.0.0rc4, mmengine>=0.2.0 required.
    """

    def __init__(
        self,
        reduce_zero_label=None,
        backend_args=None,
        imdecode_backend='pillow',
    ) -> None:
        super().__init__(
            with_bbox=False,
            with_label=False,
            with_seg=True,
            with_keypoints=False,
            imdecode_backend=imdecode_backend,
            backend_args=backend_args)
        self.reduce_zero_label = reduce_zero_label
        if self.reduce_zero_label is not None:
            warnings.warn('`reduce_zero_label` will be deprecated, '
                          'if you would like to ignore the zero label, please '
                          'set `reduce_zero_label=True` when dataset '
                          'initialized')
        self.imdecode_backend = imdecode_backend

    def _load_seg_map(self, results: dict) -> None:
        """Private function to load semantic segmentation annotations.

        Args:
            results (dict): Result dict from :obj:``mmcv.BaseDataset``.

        Returns:
            dict: The dict contains loaded semantic segmentation annotations.
        """

        img_bytes = fileio.get(
            results['seg_map_path'], backend_args=self.backend_args)
        try:
            
            gt_semantic_seg = mmcv.imfrombytes(
                img_bytes, flag='unchanged',
                backend=self.imdecode_backend).squeeze().astype(np.uint8)
            
        except Exception as e:
            pass

        # reduce zero_label
        if self.reduce_zero_label is None:
            self.reduce_zero_label = results['reduce_zero_label']
        assert self.reduce_zero_label == results['reduce_zero_label'], \
            'Initialize dataset with `reduce_zero_label` as ' \
            f'{results["reduce_zero_label"]} but when load annotation ' \
            f'the `reduce_zero_label` is {self.reduce_zero_label}'
        if self.reduce_zero_label:
            # avoid using underflow conversion
            gt_semantic_seg[gt_semantic_seg == 0] = 255
            gt_semantic_seg = gt_semantic_seg - 1
            gt_semantic_seg[gt_semantic_seg == 254] = 255


        # FUCK MMCV. IM USING MY LABELMAP
        # label_map={
        #     255: 0, 
        #     1: 1, 
        #     0: 255
        #     },
        # gt_semantic_seg_copy = gt_semantic_seg.copy()
        # gt_semantic_seg[gt_semantic_seg_copy == 1] = 2
        # gt_semantic_seg[gt_semantic_seg_copy == 255] = 1
        # gt_semantic_seg[gt_semantic_seg_copy == 0] = 255
        # gt_semantic_seg = gt_semantic_seg-1
        # gt_semantic_seg[gt_semantic_seg_copy == 254] = 255


        # modify if custom classes
        if results.get('label_map', None) is not None:
            # Add deep copy to solve bug of repeatedly
            # replace `gt_semantic_seg`, which is reported in
            # https://github.com/open-mmlab/mmsegmentation/pull/1445/
            gt_semantic_seg_copy = gt_semantic_seg.copy()
            for old_id, new_id in results['label_map'].items():
                gt_semantic_seg[gt_semantic_seg_copy == old_id] = new_id


        results['gt_seg_map'] = gt_semantic_seg
        results['seg_fields'].append('gt_seg_map')

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(reduce_zero_label={self.reduce_zero_label}, '
        repr_str += f"imdecode_backend='{self.imdecode_backend}', "
        repr_str += f'backend_args={self.backend_args})'
        return repr_str
