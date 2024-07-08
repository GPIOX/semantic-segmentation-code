_base_ = [
    './datasets/watertank_seg.py',
    'base_trainer.py'
]

train_dataloader = dict(batch_size=4, num_workers=4)
test_dataloader = dict(batch_size=1, num_workers=4)

optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer, clip_grad=None)
param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=160000,
        by_epoch=False,
    )
]

decode_loss = [
    dict(
        type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)
]

model_cfg = dict(
    type='Model', a=1, b=2)

logger = [
    dict(
        type='CSVLogger',
        name='CSV',
    ),
]
