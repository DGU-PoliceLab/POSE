_base_ = '../../_base_/default_runtime.py'

model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='ResNet3dSlowOnly',
        depth=50,
        pretrained=None,
        in_channels=17,
        base_channels=32,
        num_stages=3,
        out_indices=(2, ),
        stage_blocks=(4, 6, 3),
        conv1_stride_s=1,
        pool1_stride_s=1,
        inflate=(0, 1, 1),
        spatial_strides=(2, 2, 2),
        temporal_strides=(1, 1, 2),
        dilations=(1, 1, 1)),
    cls_head=dict(
        type='I3DHead',
        in_channels=512,
        num_classes=2,
        spatial_type='avg',
        dropout_ratio=0.5,
        average_clips='prob')
)

ann_file_train = 'data/anno/custom_dataset_train.pkl'  # Your annotation for training
ann_file_val = 'data/anno/custom_dataset_val.pkl'      # Your annotation for validation

train_dataloader = dict(
    batch_size=16,
    dataset=dict(
        type='PoseDataset',  # 적절한 데이터셋 클래스 지정
        ann_file=ann_file_train,
        pipeline=[
            dict(
                clip_len=48,
                num_clips=1,
                test_mode=False,
                type='UniformSampleFrames'),
            dict(type='PoseDecode'),
            dict(allow_imgpad=True, hw_ratio=1.0, type='PoseCompact'),
            dict(scale=(-1, 64), type='Resize'),
            dict(crop_size=64, type='CenterCrop'),
            dict(sigma=0.6, type='GeneratePoseTarget', use_score=True, with_kp=True, with_limb=False),
            dict(input_format='NCTHW_Heatmap', type='FormatShape'),
            dict(type='PackActionInputs'),
        ]
    )
)

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=50)
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)

optim_wrapper = dict(
    optimizer=optimizer
)

param_scheduler = dict(type='StepLR', step_size=10, gamma=0.1)

val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=16,
    dataset=dict(
        type='PoseDataset',  # 적절한 데이터셋 클래스 지정
        ann_file=ann_file_val,
        pipeline=[
            dict(
                clip_len=48,
                num_clips=1,
                test_mode=True,
                type='UniformSampleFrames'),
            dict(type='PoseDecode'),
            dict(allow_imgpad=True, hw_ratio=1.0, type='PoseCompact'),
            dict(scale=(-1, 64), type='Resize'),
            dict(crop_size=64, type='CenterCrop'),
            dict(sigma=0.6, type='GeneratePoseTarget', use_score=True, with_kp=True, with_limb=False),
            dict(input_format='NCTHW_Heatmap', type='FormatShape'),
            dict(type='PackActionInputs'),
        ]
    )
)

val_evaluator = dict(type='AccMetric')