# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.registry import DATASETS
from mmseg.datasets.basesegdataset import BaseSegDataset


@DATASETS.register_module()
class YZTDL2Dataset(BaseSegDataset):
    """YZTLine dataset.

    In segmentation map annotation for LoveDA, 0 is the ignore index.
    ``reduce_zero_label`` should be set to True. The ``img_suffix`` and
    ``seg_map_suffix`` are both fixed to '.png'.
    """
    
    METAINFO = dict(
        classes=(
            # 'background', 
            '耕地',
            '种植园用地',
            '林地',
            '湿地草地',  #04
            '商业服务用地',
            '工矿用地',
            '住宅用地',
            '公共管理与公共服务用地',
            '特殊用地',
            '交通运输用地',
            '水域水利设施',
            '其他用地',
            
            
            
            


            # 'background', 
            
        ),
        palette=[
            # [0, 0, 0],
            [255, 0, 0],
            [0, 0, 255],
            [0, 255, 0],
            [0, 255, 255],
            [255, 255, 0],
            [255, 0, 255],
            [0, 255, 128],
            [128, 255, 0],
            [0, 128, 255],
            [255, 128, 0],
            [128, 0, 255],
            [255, 255, 255],
        ],         

        # label_map={
        #     255: 0, 
        #     1: 1, 
        #     0: 255
        #     },

        )

    def __init__(self,
                 img_suffix='.tif',
                 seg_map_suffix='.tif',
                 reduce_zero_label=False,
                 **kwargs) -> None:      
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,            
            **kwargs)
