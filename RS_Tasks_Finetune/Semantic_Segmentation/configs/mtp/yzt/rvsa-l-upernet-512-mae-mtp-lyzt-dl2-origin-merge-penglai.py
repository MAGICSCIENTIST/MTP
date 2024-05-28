custom_imports = dict(imports=[
        'RS_Tasks_Finetune.Semantic_Segmentation.mmseg.models.backbones.vit_rvsa_mtp',
        'RS_Tasks_Finetune.Semantic_Segmentation.mmseg.datasets.transforms.loading',
        'RS_Tasks_Finetune.Semantic_Segmentation.mmcv_custom.layer_decay_optimizer_constructor_vit',
        'RS_Tasks_Finetune.Semantic_Segmentation.mmcv_custom.custom_layer_decay_optimizer_constructor'
    ], allow_failed_imports=False)
############################### default runtime #################################

default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)
vis_backends = [dict(type='LocalVisBackend'),
                dict(type='TensorboardVisBackend')]
visualizer = dict(
    type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')
log_processor = dict(by_epoch=False)
log_level = 'INFO'
load_from = None
resume = False

############################### dataset #################################

dataset_type = 'YZTOriginDataset'
data_root = 'D:\\learn\\MyWork\\mylib\\output\\temp\\splitFile_raster_merge_dl2_origin_256'
crop_size = (256, 256)
train_pipeline = [
    dict(type='LoadSingleRSImageFromFile'),
    dict(type='LoadAnnotationsTIF', reduce_zero_label=True),
    dict(
        type='RandomResize',
        scale=(512, 512),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]
val_pipeline = [
    dict(type='LoadSingleRSImageFromFile'),    
    dict(type='Resize', scale=crop_size, keep_ratio=False),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotationsTIF', reduce_zero_label=True),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadSingleRSImageFromFile'),
    dict(type='LoadAnnotationsTIF', reduce_zero_label=True ),
    dict(type='PackSegInputs')

]
train_dataloader = dict(
    batch_size=2,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='splitFile_raster\\train', seg_map_path='splitFile_current\\train'),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='splitFile_raster\\val', seg_map_path='splitFile_current\\val'),
        pipeline=val_pipeline))
test_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='splitFile_raster\\val', seg_map_path='splitFile_current\\val'),
        pipeline=test_pipeline))

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'], ignore_index=255)
# test_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'], format_only=True)
test_evaluator = val_evaluator

############################### running schedule #################################

# optimizer

optim_wrapper = dict(
    optimizer=dict(
    type='AdamW', lr=6e-5, betas=(0.9, 0.999), weight_decay=0.05),
    constructor='LayerDecayOptimizerConstructor_ViT', 
    paramwise_cfg=dict(
        num_layers=24, 
        layer_decay_rate=0.9,
        )
        )

param_scheduler = [
    dict(
        #1e-6
        type='LinearLR', start_factor=5e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='CosineAnnealingLR',
        eta_min=0.0,
        T_max=78500,
        begin=1500,
        end=80000,
        by_epoch=False,
    )
]

# training schedule for 80k
train_cfg = dict(type='IterBasedTrainLoop', max_iters=80000, val_interval=5000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=5000),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook', draw=True, interval=1))

############################### model #################################

norm_cfg = dict(type='SyncBN', requires_grad=True)

data_preprocessor = dict(
    type='SegDataPreProcessor',
    size = crop_size,
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255)
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='RVSA_MTP',
        img_size=256,
        patch_size=16,
        drop_path_rate=0.3,
        out_indices=[7, 11, 15, 23],
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        use_checkpoint=False,
        use_abs_pos_emb=True,
        interval=6,
        pretrained = 'D:\\learn\\MTP\\pretrain\\last_vit_l_rvsa_ss_is_rd_pretrn_model_encoder.pth'
    ),
    decode_head=dict(
        type='UPerHead',
        in_channels=[1024, 1024, 1024, 1024],
        num_classes=7,
        ignore_index=255,
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0, class_weight= [0.4713643  ,0.17672684 ,1.66769399,0 ,1.52009316, 0.42749887, 1.60416765 ])
    ),
    train_cfg=dict(),
    test_cfg=dict(mode='slide', stride=(128,128), crop_size=(256, 256)))