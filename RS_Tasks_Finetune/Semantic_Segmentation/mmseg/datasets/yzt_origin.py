# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.registry import DATASETS
from mmseg.datasets.basesegdataset import BaseSegDataset


@DATASETS.register_module()
class YZTOriginDataset(BaseSegDataset):
    """YZTLine dataset.

    In segmentation map annotation for LoveDA, 0 is the ignore index.
    ``reduce_zero_label`` should be set to True. The ``img_suffix`` and
    ``seg_map_suffix`` are both fixed to '.png'.
    """
    # background = 0
    # Farmland = 1
    # Forest = 2
    # Grass = 3
    # Wetland = 4
    # Water = 5
    # Residential = 6
    # WasteLand = 7  
    METAINFO = dict(
        classes=(
            # 'background', 
            'Farmland', 
            'Forest',
            'Grass',
            'Wetland',
            'Water',
            'Residential',
            'WasteLand',

            # 'background', 
            
        ),
        palette=[
            # [0, 0, 0],
            [255, 255, 255], 
            [0, 255, 0],
            [255, 0, 0],
            [159, 129, 183],
            [0, 0, 255],
            [255, 255, 0],
            [0, 255, 255],
            # [255, 195, 128], 
            
            # [255, 255, 255], 
            # [255, 255, 0]
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
                 reduce_zero_label=True,
                 **kwargs) -> None:      
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,            
            **kwargs)
