from generativeimage2text.common import Config
import json
import os.path as op
from generativeimage2text.common import qd_tqdm as tqdm
from generativeimage2text.common import json_dump
from generativeimage2text.common import pilimg_from_base64
from generativeimage2text.torch_common import recursive_to_device
from generativeimage2text.tsv_io import TSVFile, tsv_writer, tsv_reader
from generativeimage2text.common import write_to_file
import torch
import PIL
from pprint import pformat
import logging
from transformers import BertTokenizer
import torchvision.transforms as transforms
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image
from azfuse import File
from matplotlib import pyplot as plt

from generativeimage2text.common import init_logging
from generativeimage2text.common import parse_general_args
from generativeimage2text.tsv_io import load_from_yaml_file
from generativeimage2text.torch_common import torch_load
from generativeimage2text.torch_common import load_state_dict
from generativeimage2text.torch_common import resize_2d_pos_embed
from generativeimage2text.layers.CLIP import clip
from generativeimage2text.layers.decoder import (TransformerDecoderTextualHead,
                             AutoRegressiveBeamSearch, GeneratorWithBeamSearch)
from generativeimage2text.layers.decoder import CaptioningModel
from generativeimage2text.process_image import load_image_by_pil
from generativeimage2text.data_layer.transform import RenameKey, SelectTransform
from generativeimage2text.data_layer.transform import ImageTransform2Dict
from generativeimage2text.data_layer.transform import get_inception_train_transform
from generativeimage2text.data_layer.builder import collate_fn
from generativeimage2text.model import get_git_model


def get_data(image_file, prefix, target, tokenizer, image_transform):
    max_text_len = 40
    # prefix encoding --- none for IC
    prefix_encoding = tokenizer(
        prefix, padding='do_not_pad',
        add_special_tokens=False,
        truncation=True, max_length=max_text_len)
    # caption - target encoding  -- input ids - token type ids - attention mask
    # 1012 i teleia sto decode
    target_encoding = tokenizer(
        target, padding='do_not_pad',
        add_special_tokens=False,
        truncation=True, max_length=max_text_len)
    # len of target + 1 , i teleia
    # need predict [0,0,0,1,1,1,1,1] i [1,1,1,1,]
    need_predict = [0] * len(prefix_encoding['input_ids']) + [1] * len(target_encoding['input_ids'])
    # payload sum of input ids
    payload = prefix_encoding['input_ids'] + target_encoding['input_ids']

    if len(payload) > max_text_len:
        payload = payload[-(max_text_len - 2):]
        need_predict = need_predict[-(max_text_len - 2):]

    # CLS - 101 .... SEP - 102
    input_ids = [tokenizer.cls_token_id] + payload + [tokenizer.sep_token_id]
    need_predict = [0] + need_predict + [1]
    print('heeeeeeeeeeee')
    im = load_image_by_pil(image_file)
    # print('*'*8)
    # print(im)
    # print('*'*8)
    data = {
        'caption_tokens': torch.tensor(input_ids),
        'need_predict': torch.tensor(need_predict),
        'image': im,
        'caption': {},
        # this iteration can be used for crop-size selection so that all GPUs
        # can process the image with the same input size
        'iteration': 0,
    }


    data = image_transform(data)

    return data

def get_image_transform(cfg):
    return get_multi_scale_image_transform(cfg, is_train=True)

def get_default_mean():
    return [0.485, 0.456, 0.406]

def get_default_std():
    return [0.229, 0.224, 0.225]

def get_transform_image_norm(cfg, default=None):
    if cfg.data_normalize == 'default':
        normalize = transforms.Normalize(
            mean=get_default_mean(), std=get_default_std())
    elif cfg.data_normalize == 'clip':
        # clip model
        normalize = transforms.Normalize(
            (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    else:
        raise NotImplementedError(cfg.data_normalize)
    return normalize

def get_transform_vit_default(cfg, is_train):
    default_normalize = transforms.Normalize(
            mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    normalize = get_transform_image_norm(cfg, default_normalize)
    transform = get_inception_train_transform(
        bgr2rgb=True,
        crop_size=cfg.train_crop_size,
        normalize=normalize,
        small_scale=cfg.input_small_scale,
        no_color_jitter=cfg.no_color_jitter,
        no_flip=cfg.no_flip,
        no_aspect_dist=cfg.no_aspect_dist,
        resize_crop=cfg.resize_crop,
        max_size=cfg.train_max_size,
        interpolation=cfg.interpolation or Image.BILINEAR,
    )
    return transform

def get_transform_image(cfg, is_train):
    train_transform = cfg.train_transform
    if train_transform == 'vitp':
        transform = get_transform_vit_default(
            cfg, is_train=is_train)
    else:
        raise NotImplementedError(train_transform)
    return transform

class ImageTransform2Images(object):
    def __init__(self, sep_transform, first_joint=None):
        self.image_transform = sep_transform
        self.first_joint = first_joint

    def __call__(self, imgs):
        if self.first_joint is not None:
            imgs = self.first_joint(imgs)
        return [self.image_transform(im) for im in imgs]

    def __repr__(self):
        return 'ImageTransform2Images(image_transform={})'.format(
            self.image_transform,
        )

def get_transform_images(cfg, is_train):
    trans = get_transform_image(cfg, is_train)
    trans = ImageTransform2Images(trans)
    return trans

def trans_select_for_crop_size(
    data, train_crop_sizes,
    iteration_multi=0,
):
    if iteration_multi <= 0:
        if len(train_crop_sizes) == 1:
            idx = 0
        else:
            idx = data['iteration'] % len(train_crop_sizes)
    elif data['iteration'] <= iteration_multi:
        idx = data['iteration'] % len(train_crop_sizes)
    else:
        idx = -1
    return idx

def get_multi_scale_image_transform(cfg, is_train, get_one=get_transform_image):
    def get_multi_res_transform(s):
        old = cfg.train_crop_size if is_train else cfg.test_crop_size
        all_t = []
        multi_res_factors = cfg.multi_res_factors or []
        for i, f in enumerate(multi_res_factors):
            if is_train:
                cfg.train_crop_size = s // f
            else:
                cfg.test_crop_size = s // f
            key = 'image_{}'.format(i)
            all_t.append(RenameKey({'image': key}, not_delete_origin=True))
            t = get_one(cfg, is_train)
            t = ImageTransform2Dict(t, key=key)
            all_t.append(t)
        # get_one depends on train_crop_size
        if is_train:
            cfg.train_crop_size = s
        else:
            cfg.test_crop_size = s
        t = get_one(cfg, is_train)
        t = ImageTransform2Dict(t)
        all_t.append(t)
        if is_train:
            cfg.train_crop_size = old
        else:
            cfg.test_crop_size = old
        return transforms.Compose(all_t)

   # min size range [160-224]
   # train crop sizes [160-176-192-208 -224]
    if is_train:
        if cfg.min_size_range32 is None:
            train_crop_sizes = [cfg.train_crop_size]
        else:
            train_crop_sizes = list(range(
                cfg.min_size_range32[0],
                cfg.min_size_range32[1] + cfg.patch_size - 1, cfg.patch_size,
            ))
    else:
        train_crop_sizes = [cfg.test_crop_size]

    crop_trans = []
    for s in train_crop_sizes:
        t = get_multi_res_transform(s)
        crop_trans.append(t)
    iteration_multi = 0
    image_transform = SelectTransform(
        crop_trans,
        lambda d: trans_select_for_crop_size(
            d, train_crop_sizes, iteration_multi))
    return image_transform

def forward_backward_example(image_files, captions, prefixs=None):
    if prefixs is None:
        prefixs = [''] * len(captions)
    cfg = {
        'crop_region_extend_in_datatransform': 4,
        'data_normalize': 'clip',
        'train_crop_size': 224,
        'input_small_scale': 0.8,
        'no_color_jitter': True,
        'no_flip': True,
        'no_aspect_dist': True,
        'interpolation': 'bicubic',
        'min_size_range32': [160, 224], # in pretraining, it is multi-scale from 160 to 224; while for fine-tuning, it is single scale
        'patch_size': 16,
        'train_transform': 'vitp',
    }
    cfg = Config(cfg, {})
    all_data = []
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    image_transform = get_image_transform(cfg)


    #### ?? image_transform ??
    ## transformations normalization etc..


    for image_file, prefix, target in zip(image_files, prefixs, captions):
        data = get_data(image_file, prefix, target,
                        tokenizer, image_transform)
        all_data.append(data)

    data = collate_fn(all_data)
    # logging.info(image_transform)

    data = recursive_to_device(data, 'cpu') # cuda

    param = {}
    model = get_git_model(tokenizer, param)
    model.train()
    # model.cuda()
    loss_dict = model(data)
    loss = sum(loss_dict.values())
    loss.backward()
    logging.info(loss)




if __name__ == '__main__':
    print('hello..')
    init_logging()
    kwargs = parse_general_args()
    logging.info('param:\n{}'.format(pformat(kwargs)))
    # function_name = kwargs['type']
    # del kwargs['type']
    # locals()[function_name](**kwargs)


    # VQA ################
    # forward_backward_example(image_files=['aux_data/images/1.jpg', 'aux_data/images/2.jpg'],
    #                          prefixs=['what is this?', 'how many trees?'],
    #                          captions=['several boats in a large body of water', '1'])


    # IC ################
    forward_backward_example(image_files=['aux_data/images/1.jpg', 'aux_data/images/2.jpg'],
                             captions=['a couple of boats in a large body of water.',
                                         'a view of a mountain with a tree'])




