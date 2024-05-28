# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import logging
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


import os.path as osp

import RS_Tasks_Finetune.Semantic_Segmentation.mmseg.datasets.transforms.loading
import RS_Tasks_Finetune.Semantic_Segmentation.mmseg.models.backbones.vit_rvsa_mtp
import RS_Tasks_Finetune.Semantic_Segmentation.mmcv_custom.layer_decay_optimizer_constructor_vit

from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmengine.runner import Runner
from mmseg.models.losses import CrossEntropyLoss

from mmseg.registry import RUNNERS

from mmengine.hooks.checkpoint_hook import CheckpointHook
from mmseg.evaluation.metrics.iou_metric import IoUMetric

# from mmseg.datasets.transforms import LoadAnnotations
# from mmseg.datasets.transforms.formatting import PackSegInputs
# from RS_Tasks_Finetune.Semantic_Segmentation.mmseg.datasets.transforms.loading import LoadSingleRSImageFromFile


def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume',
        action='store_true',
        default=False,
        help='resume from the latest checkpoint in the work_dir automatically')
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()
    # img_path = "D:\\learn\\MyWork\\mylib\\outputSD\\temp_1024\\splitFile\\splitFile_raster\\train\\161_0_123648.tif"
    # gt_path = "D:\\learn\\MyWork\\mylib\\outputSD\\temp_1024\\splitFile\\splitFile_raster_line\\train\\161_0_123648.tif"
    # transforms = LoadSingleRSImageFromFile()
    # results = dict(
    #     img_path=img_path,
    #     seg_map_path=gt_path,
    #     reduce_zero_label=False,
    #     seg_fields=[]
    # )
    # data_dict = transforms(results)
    # print(data_dict.keys())

    # transforms = LoadAnnotations()
    # results = dict(
    #     img_path=img_path,
    #     seg_map_path=gt_path,
    #     reduce_zero_label=False,
    #     seg_fields=[]
    # )
    # data_dict = transforms(results)
    # print(data_dict.keys())

    # PackSegInputs


    # load config
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    # enable automatic-mixed-precision training
    if args.amp is True:
        optim_wrapper = cfg.optim_wrapper.type
        if optim_wrapper == 'AmpOptimWrapper':
            print_log(
                'AMP training is already enabled in your config.',
                logger='current',
                level=logging.WARNING)
        else:
            assert optim_wrapper == 'OptimWrapper', (
                '`--amp` is only supported when the optimizer wrapper type is '
                f'`OptimWrapper` but got {optim_wrapper}.')
            cfg.optim_wrapper.type = 'AmpOptimWrapper'
            cfg.optim_wrapper.loss_scale = 'dynamic'

    # resume training
    cfg.resume = args.resume

    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    # start training
    runner.train()


if __name__ == '__main__':
    main()
