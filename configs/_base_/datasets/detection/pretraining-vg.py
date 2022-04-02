dataset = 'PretrainingVG'
data_root = './data/'
img_norm_cfg = dict(
    mean=[0., 0., 0.], std=[1., 1., 1.])

train_pipeline = [
    dict(type='LoadImageAnnotationsFromFile',
         max_token=40, with_bbox=True, dataset=dataset),
    dict(type='Resize', img_scale=(640, 640)),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='CollectData', keys=[
         'img', 'ref_expr_inds', 'gt_bbox'])
]
val_pipeline = [
    dict(type='LoadImageAnnotationsFromFile',
         max_token=40, with_bbox=True, dataset=dataset),
    dict(type='Resize', img_scale=(640, 640)),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='CollectData', keys=[
         'img', 'ref_expr_inds', 'gt_bbox'])
]
test_pipeline = val_pipeline.copy()

word_emb_cfg = dict(type='GloVe')
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=1,
    train=dict(
        type=dataset,
        which_set='train',
        img_source=['coco', 'flickr', 'visual-genome', 'saiaprtc12'],
        annsfile=data_root + 'annotations/pretraining-vg/instances.json',
        imgsfile=dict(
            coco=data_root + 'images/mscoco/train2014',
            vg=data_root + 'images/visual-genome',
            saiaprtc12=data_root + 'images/saiaprtc12',
            flickr=data_root + 'images/flickr30k'
        ),
        pipeline=train_pipeline,
        word_emb_cfg=word_emb_cfg),
    val_refcoco_unc=dict(
        type=dataset,
        which_set='val_refcoco_unc',
        img_source=['coco', 'flickr', 'visual-genome', 'saiaprtc12'],
        annsfile=data_root + 'annotations/pretraining-vg/instances.json',
        imgsfile=dict(
            coco=data_root + 'images/mscoco/train2014',
            vg=data_root + 'images/visual-genome',
            saiaprtc12=data_root + 'images/saiaprtc12',
            flickr=data_root + 'images/flickr30k'
        ),
        pipeline=val_pipeline,
        word_emb_cfg=word_emb_cfg),
    val_refcocoplus_unc=dict(
        type=dataset,
        which_set='val_refcocoplus_unc',
        img_source=['coco', 'flickr', 'visual-genome', 'saiaprtc12'],
        annsfile=data_root + 'annotations/pretraining-vg/instances.json',
        imgsfile=dict(
            coco=data_root + 'images/mscoco/train2014',
            vg=data_root + 'images/visual-genome',
            saiaprtc12=data_root + 'images/saiaprtc12',
            flickr=data_root + 'images/flickr30k'
        ),
        pipeline=test_pipeline,
        word_emb_cfg=word_emb_cfg),
    val_refcocog_umd=dict(
        type=dataset,
        which_set='val_refcocog_umd',
        img_source=['coco', 'flickr', 'visual-genome', 'saiaprtc12'],
        annsfile=data_root + 'annotations/pretraining-vg/instances.json',
        imgsfile=dict(
            coco=data_root + 'images/mscoco/train2014',
            vg=data_root + 'images/visual-genome',
            saiaprtc12=data_root + 'images/saiaprtc12',
            flickr=data_root + 'images/flickr30k'
        ),
        pipeline=test_pipeline,
        word_emb_cfg=word_emb_cfg)
)
