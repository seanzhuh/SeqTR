# SeqTR

![overview](./teaser/overview.png)  

This is the official implementation of [SeqTR: A Simple yet Universal Network for Visual Grounding](https://arxiv.org/abs/2203.16265), which simplifies and unifies the modelling for visual grounding tasks under a novel point prediction paradigm. 

<!-- To this end, different grounding tasks can be tackled in one network with the simple cross-entropy loss. We surpass or maintain on par with state-of-the-arts, and also outperform a set of larget-scale pre-trained models with much less expenditure, suggesting a simple and universal approach is indeed feasible. -->

## Installation

### Prerequisites

```
pip install -r requirements.txt
wget https://github.com/explosion/spacy-models/releases/download/en_vectors_web_lg-2.1.0/en_vectors_web_lg-2.1.0.tar.gz -O en_vectors_web_lg-2.1.0.tar.gz
pip install en_vectors_web_lg-2.1.0.tar.gz
```
Then install SeqTR package in editable mode:
```
pip install -e .
```

### Data Preparation

1. Download our [preprocessed json files](https://drive.google.com/drive/folders/1IXnSieVr5CHF2pVJpj0DlwC6R3SbfolU?usp=sharing) including the merged dataset for pre-training, and [DarkNet-53 model weights](https://drive.google.com/drive/folders/1W8y_WS-8cnuU0LnF8e1v8ZowZvpEaolk?usp=sharing) trained on MS-COCO object detection task.
2. Download the train2014 images from [mscoco](https://cocodataset.org/) or from [Joseph Redmon's mscoco mirror](https://pjreddie.com/projects/coco-mirror/), of which the download speed is faster than the official website.
3. Download [original Flickr30K images](http://shannon.cs.illinois.edu/DenotationGraph/) and [ReferItGame images](https://drive.google.com/file/d/1R6Tm7tQTHCil6A_eOhjudK3rgaBxkD2t/view?usp=sharing).

The project structure should look like the following:

```
| -- SeqTR
     | -- data
        | -- annotations
            | -- flickr30k
                | -- instances.json
                | -- ix_to_token.pkl
                | -- token_to_ix.pkl
                | -- word_emb.npz
            | -- referitgame-berkeley
            | -- refcoco-unc
            | -- refcocoplus-unc
            | -- refcocog-umd
            | -- refcocog-google
            | -- pretraining-vg 
        | -- weights
            | -- darknet.weights
            | -- yolov3.weights
        | -- images
            | -- mscoco
                | -- train2014
                    | -- COCO_train2014_000000000072.jpg
                    | -- ...
            | -- saiaprtc12
                | -- 25.jpg
                | -- ...
            | -- flickr30k
                | -- 36979.jpg
                | -- ...
     | -- configs
     | -- seqtr
     | -- tools
     | -- teaser
```
Note that the darknet.weights excludes val/test images of RefCOCO/+/g datasets while yolov3.weights does not.

## Training

### Phrase Localization and Referring Expression Comprehension

We train SeqTR to perform grouning at bounding box level on a single V100 GPU. The following script performs the training:
```
python tools/train.py configs/seqtr/detection/seqtr_det_[DATASET_NAME].py --cfg-options ema=True
```
[DATASET_NAME] is one of "flickr30k", "referitgame-berkeley", "refcoco-unc", "refcocoplus-unc", "refcocog-umd", and "refcocog-google".

### Referring Expression Segmentation

To train SeqTR to generate the target sequence of ground-truth mask, which is then assembled into the predicted mask by connecting the points, run the following script:

```
python tools/train.py configs/seqtr/segmentation/seqtr_mask_[DATASET_NAME].py --cfg-options ema=True
```

Note that instead of sampling 18 points and does not shuffle the sequence for RefCOCO dataset, for RefCOCO+ and RefCOCOg, we uniformly sample 12 points on the mask contour and randomly shffle the sequence with 20% percentage. Therefore, to execute the training on RefCOCO+/g datasets, modify **num_ray at line 1 to 18** and **model.head.shuffle_fraction to 0.2 at line 35**, in configs/seqtr/segmentation/seqtr_mask_darknet.py.

## Evaluation

```
python tools/test.py [PATH_TO_CONFIG_FILE] --load-from [PATH_TO_CHECKPOINT_FILE]
```

## Pre-training + fine-tuning

We train SeqTR on 8 V100 GPUs while disabling Large Scale Jittering (LSJ) and Exponential Moving Average (EMA):
 
```
bash tools/dist_train.sh configs/seqtr/detection/seqtr_det_pretraining-vg.py 8
```

## Models

<table>
<th></th><th colspan="4">RefCOCO</th><th colspan="4">RefCOCO+</th><th colspan="4">RefCOCOg</th>
<tr>
<td></td><td>val</td><td>testA</td><td>testB</td><td>model</td><td>val</td><td>testA</td><td>testB</td><td>model</td><td>val-g</td><td>val-u</td><td>val-u</td><td>model</td>
</tr>
<tr>
<td>SeqTR on REC</td><td>81.23</td><td>85.00</td><td>76.08</td><td></td><td>68.82</td><td>75.37</td><td>58.78</td><td></td><td>-</td><td>71.35</td><td>71.58</td><td></td>
</tr>
<tr>
<td>SeqTR* on REC</td><td>83.72</td><td>86.51</td><td>81.24</td><td></td><td>71.45</td><td>76.26</td><td>64.88</td><td></td><td>71.50</td><td>74.86</td><td>74.21</td><td></td>
</tr>
<tr>
<td>SeqTR pre-trained+finetuned on REC</td><td>87.00</td><td>90.15</td><td>83.59</td><td></td><td>78.69</td><td>84.51</td><td>71.87</td><td></td><td>-</td><td>82.69</td><td>83.37</td><td></td>
</tr>
<tr>
<td>SeqTR on RES</td><td>67.26</td><td>69.79</td><td>64.12</td><td></td><td>54.14</td><td>58.93</td><td>48.19</td><td></td><td>-</td><td>55.67</td><td>55.64</td><td></td>
</tr>
</table>
SeqTR* denotes that its visual encoder is initialized with yolov3.weights, while the visual encoder of the rest are initialized with darknet.weights.

## Citation

```
@article{zhu2022seqtr,
  title={SeqTR: A Simple yet Universal Network for Visual Grounding},
  author={Zhu, ChaoYang and Zhou, YiYi and Shen, YunHang and Luo, Gen and Pan, XingJia and Lin, MingBao and Chen, Chao and Cao, LiuJuan and Sun, XiaoShuai and Ji, RongRong},
  journal={arXiv preprint arXiv:2203.16265},
  year={2022}
}
```

## Acknowledgement

Our code is built upon the open-sourced [mmcv](https://github.com/open-mmlab/mmcv) and [mmdetection](https://github.com/open-mmlab/mmdetection) libraries. 