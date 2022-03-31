dataset = 'RefCOCOgGoogle'
data_root = './data/'
img_norm_cfg = dict(
    mean=[0., 0., 0.], std=[1., 1., 1.])

train_pipeline = [
    dict(type='LoadImageAnnotationsFromFile',
         max_token=20, with_mask=True, dataset="RefCOCOgGoogle"),
    dict(type='LargeScaleJitter', out_max_size=640,
         jitter_min=0.3, jitter_max=1.4),
    # dict(type='Resize', img_scale=(640, 640)),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='GenerateMaskVertices', num_ray=12, polar=False),
    dict(type='DefaultFormatBundle'),
    dict(type='CollectData', keys=[
         'img', 'ref_expr_inds',
         'gt_mask_rle', 'is_crowd', 'gt_mask_vertices', 'mass_center'])
]
val_pipeline = [
    dict(type='LoadImageAnnotationsFromFile',
         max_token=20, with_mask=True, dataset="RefCOCOgGoogle"),
    dict(type='Resize', img_scale=(640, 640)),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='CollectData', keys=[
         'img', 'ref_expr_inds', 'is_crowd', 'gt_mask_rle'])
]
test_pipeline = val_pipeline.copy()

word_emb_cfg = dict(type='GloVe')
data = dict(
    samples_per_gpu=128,
    workers_per_gpu=4,
    train=dict(
        type=dataset,
        which_set='train',
        img_source=['coco'],
        annsfile=data_root + 'annotations/refcocog-google/instances.json',
        imgsfile=data_root + 'images/mscoco/train2014',
        pipeline=train_pipeline,
        word_emb_cfg=word_emb_cfg),
    val=dict(
        type=dataset,
        which_set='val',
        img_source=['coco'],
        annsfile=data_root + 'annotations/refcocog-google/instances.json',
        imgsfile=data_root + 'images/mscoco/train2014',
        pipeline=val_pipeline,
        word_emb_cfg=word_emb_cfg)
)
