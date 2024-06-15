import re
import os
import pickle
import torch
import apex
import numpy as np
from typing import Literal
from mmcv import Config
from PIL import Image, ImageDraw, ImageOps
import pycocotools.mask as maskUtils

from seqtr.models import build_model, ExponentialMovingAverage
from seqtr.datasets.pipelines import Resize, Normalize, Pad, CollectData
from seqtr.utils import load_checkpoint

STATIC_PATH = "data/annotations/"

def clean_string(expression):
    return re.sub(r"([.,'!?\"()*#:;])", '', expression.lower()).replace('-', ' ').replace('/', ' ')

def scan_dir(dir_path, end_with=".py"):
    file_list = os.listdir(dir_path)
    filtered_files = [file for file in file_list if file.endswith(end_with)]
    return filtered_files

from mmdet.datasets.pipelines import to_tensor
class ForwardOnceFormatBundle:
    def _add_default_meta_keys(self, results):
        """Add default meta keys.
        """
        img = results['img']
        results.setdefault('pad_shape', img.shape)
        results.setdefault('scale_factor', 1.0)
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results.setdefault('img_norm_cfg',
                            dict(mean=np.zeros(num_channels, dtype=np.float32),
                                 std=np.ones(num_channels, dtype=np.float32),
                                 to_rgb=False))
        return results

    def __call__(self, results):
        """Call function to transform and format common fields in results.
        """
        if 'img' in results:
            img = results['img']
            # add default meta keys
            results = self._add_default_meta_keys(results)
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            img = np.ascontiguousarray(img.transpose(2, 0, 1))
            results['img'] = to_tensor(img).unsqueeze(0)

        if 'ref_expr_inds' in results:
            results['ref_expr_inds'] = to_tensor(results['ref_expr_inds'])
        return results

class ReferenceProxy(object):
    def __init__(self, device: str="cuda", args: dict=None):
        # fixed
        self.__device = torch.device(device)
        self.__max_token = args.get('max_token', 20)
        self.__use_fp16 = args.get('use_fp16', False)
        self.__ema = args.get('ema', False)
        self.__build_image_transforms()
        
        self.__clean_state()
    
    def __clean_state(self):
        self.cfg = None
        self.model = None
        self.__token2ix = None
        self.__word_embed = None
        self.mode = ["det"] #, "seg"]
        self.__build_state = False
        self.__load_state = False
         
    def __reload_config(self, cfg_dir):
        self.cfg = Config.fromfile(cfg_dir)
        self.cfg.use_fp16 = self.__use_fp16
        self.cfg.ema = self.__ema
    
    def __build_model(self) -> None:
        assert self.cfg is not None
        self.model = build_model(self.cfg.model,
                                 word_emb=self.__word_embed,
                                 num_token=len(self.__token2ix))
        self.model.to(self.__device)
        # FP16
        if self.cfg.use_fp16:
            self.model = apex.amp.initialize(self.model, opt_level="O1")
            for m in self.model.modules():
                if hasattr(m, "fp16_enabled"):
                    m.fp16_enabled = True
        # Expoential Moving Average
        if self.cfg.ema:
            self.model_ema = ExponentialMovingAverage(self.model, self.cfg.ema_factor)
        else:
            self.model_ema = None
        self.__build_state = True
    
    def __load_checkpoint(self, ckpt_path: str) -> None:
        assert self.cfg is not None
        load_checkpoint(self.model, self.model_ema, None, ckpt_path)
        if self.cfg.ema:
            self.model_ema.apply_shadow()
        self.__check_is_eval()
        self.__load_state = True
    
    def __load_word_embed(self, datasets: str) -> None:
        token2ix_path = os.path.join(STATIC_PATH, datasets, 'token_to_ix.pkl')
        word_emb_path = os.path.join(STATIC_PATH, datasets, 'word_emb.npz')
        if os.path.exists(token2ix_path) and os.path.exists(word_emb_path):
            with open(token2ix_path, 'rb') as handle:
                token2ix = pickle.load(handle)
            npz = np.load(word_emb_path, allow_pickle=True)
            self.__token2ix = token2ix
            self.__word_embed = npz['word_emb']
        else:
            raise FileNotFoundError(f"Please check the word embedding files in {STATIC_PATH}")
        
    def __check_is_eval(self):
        if self.model.training:
            self.model.eval()
            
    def __change_mode(self, new_mode):
        self.__clean_state()
        if new_mode == 'mixed':
            self.mode = ['det', 'seg']
        elif new_mode in ['det', 'seg']:
            self.mode = [new_mode]
        else:
            raise NotImplementedError("Invalid mode!")
        
    def __build_image_transforms(self):
        self.pipeline = [
            Resize(img_scale=(640, 640)),
            Normalize(mean=[0., 0., 0.], std=[1., 1., 1.]),
            Pad(size_divisor=32),
            ForwardOnceFormatBundle(),
            CollectData(keys=['img', 'ref_expr_inds'])
        ]
    
    def __process_experssion(self, expression: str) -> dict:
        expression = clean_string(expression)
        ref_expr_inds = torch.zeros(self.__max_token, dtype=torch.long)
        for idx, word in enumerate(expression.split()):
            if word in self.__token2ix:
                ref_expr_inds[idx] = self.__token2ix[word]
            else:
                ref_expr_inds[idx] = self.__token2ix['UNK']
            if idx + 1 == self.__max_token:
                break
        return ref_expr_inds.unsqueeze(0).to(self.__device)
    
    def __fake_results_dict_pre(self, image, text):
        return dict(
            filename = None,
            img = image,
            img_shape = image.shape,
            ori_shape = image.shape,
            ref_expr_inds = self.__process_experssion(text),
            expression = text,
            max_token = self.__max_token,
            with_bbox = False,
            with_mask = False
        )
        
    def __fake_results_dict_post(self, results):
        results['img_metas'] = [results['img_metas'].data]
        results['img'] = results['img'].to(self.__device)
        results['ref_expr_inds'] = results['ref_expr_inds'].to(self.__device)
        results['with_bbox'] = 'det' in self.mode
        results['with_mask'] = 'seg' in self.mode
        return results

    def __convert_input(self, image: Image, text: str):
        image = np.array(image)
        fake_results = self.__fake_results_dict_pre(image, text)
        for each_step in self.pipeline:
            fake_results = each_step(fake_results)
        fake_results = self.__fake_results_dict_post(fake_results)
        return fake_results
        
    def __bbox_draw(self, image: Image, pred_bbox: torch.Tensor, color: str='red') -> Image:
        img_draw = ImageDraw.Draw(image)
        img_draw.rectangle(pred_bbox.tolist(), outline=color, width=2)
        return image
    
    def __mask_draw(self, image: Image, pred_mask: torch.Tensor, color: str='green'):
        decoded_mask = maskUtils.decode(pred_mask)
        mask_image = Image.fromarray((decoded_mask * 255).astype('uint8'))
        colored_mask = ImageOps.colorize(mask_image, black='black', white=color)
        colored_mask.putalpha(128)
        image_rgba = image.convert('RGBA')
        masked_image = Image.alpha_composite(image_rgba, colored_mask)
        return masked_image
    
    def get_mode(self) -> str:
        if len(self.mode) == 1:
            return self.mode[0]
        else:
            return "mixed"
    
    def load_config(self, cfg_dir: str, ckpt_dir: str, word_embed_datasets: str, mode: Literal['det', 'seg', 'mixed']='det') -> None:
        self.__change_mode(mode)
        self.__reload_config(cfg_dir)
        self.__load_word_embed(datasets=word_embed_datasets)
        self.__build_model()
        self.__load_checkpoint(ckpt_dir)
        
    def check_health(self) -> tuple[bool, str]:
        if not self.__build_state:
            return (False, "Model has not been built yet!")
        if not self.__load_state:
            return (False, "Checkpoint has not been loaded yet!")
        return (True, None)
    
    def inference(self, image: Image, text: str, rep_img: Image=None) -> Image:
        print(image, text)
        input = self.__convert_input(image, text)
        predictions = self.model(**input,
                                return_loss=False,
                                rescale=True)
        # single image process
        img_bbox, img_mask = None, None
        if 'det' in self.mode:
            pred_bboxes = predictions.pop('pred_bboxes')[0]
            img_bbox = self.__bbox_draw(image if rep_img is None else rep_img, pred_bboxes, color=(3,250,4))
            return img_bbox
        if 'seg' in self.mode:
            pred_masks = predictions.pop('pred_masks')[0]
            img_mask = self.__mask_draw(image if rep_img is None else rep_img, pred_masks, color='green')
            return img_mask
        