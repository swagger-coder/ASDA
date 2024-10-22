# -*- coding: utf-8 -*-

"""
refcoco, refcoco+ and refcocog referring image detection and segmentation PyTorch dataset.
"""
import sys
import cv2
import torch
import random
import numpy as np
import os.path as osp
import torch.utils.data as data
sys.path.append('.')
import utils
import re

from pytorch_pretrained_bert.tokenization import BertTokenizer
from utils.transforms import letterbox, random_affine, random_copy, random_crop, random_erase
import copy 

import clip

sys.modules['utils'] = utils
cv2.setNumThreads(0)

def read_examples(input_line, unique_id):
    """Read a list of `InputExample`s from an input file."""
    examples = []
    # unique_id = 0
    line = input_line #reader.readline()
    # if not line:
    #     break
    line = line.strip()
    text_a = None
    text_b = None
    m = re.match(r"^(.*) \|\|\| (.*)$", line)
    if m is None:
        text_a = line
    else:
        text_a = m.group(1) #'man in black'
        text_b = m.group(2)
    
    examples.append(
        InputExample(unique_id=unique_id, text_a=text_a, text_b=text_b))
    # unique_id += 1
    return examples

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_length:
      break
    if len(tokens_a) > len(tokens_b):
      tokens_a.pop()
    else:
      tokens_b.pop()

## Bert text encoding
class InputExample(object):
    def __init__(self, unique_id, text_a, text_b):
        self.unique_id = unique_id
        self.text_a = text_a
        self.text_b = text_b

class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self, unique_id, tokens, input_ids, input_mask, input_type_ids):
        self.unique_id = unique_id
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_type_ids = input_type_ids

def convert_examples_to_features(examples, seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""
    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a) # ['far', 'left', 'vase']

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, seq_length - 3) 
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > seq_length - 2:
                tokens_a = tokens_a[0:(seq_length - 2)]
        tokens = []
        input_type_ids = [] 
        tokens.append("[CLS]")
        input_type_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            input_type_ids.append(0)
        tokens.append("[SEP]")
        input_type_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                input_type_ids.append(1)
            tokens.append("[SEP]")
            input_type_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < seq_length:
            input_ids.append(0)
            input_mask.append(0)
            input_type_ids.append(0)

        assert len(input_ids) == seq_length
        assert len(input_mask) == seq_length
        assert len(input_type_ids) == seq_length
        features.append(
            InputFeatures(
                unique_id=example.unique_id,
                tokens=tokens,
                input_ids=input_ids,
                input_mask=input_mask,
                input_type_ids=input_type_ids))
    return features

class DatasetNotFoundError(Exception):
    pass

class ReferDataset(data.Dataset):
    SUPPORTED_DATASETS = {
        'refcoco': {
            'splits': ('train', 'val', 'testA', 'testB'),
            'params': {'dataset': 'refcoco', 'split_by': 'unc'}
        },
        'refcoco+': {
            'splits': ('train', 'val', 'testA', 'testB'),
            'params': {'dataset': 'refcoco+', 'split_by': 'unc'}
        },
        'refcocog': {
            'splits': ('train', 'val', 'test'),
            'params': {'dataset': 'refcocog', 'split_by': 'unc'}
        },
        'refcocog_g': {
            'splits': ('train', 'val'),
            'params': {'dataset': 'refcocog', 'split_by': 'google'}
        },
        'refcocog_u': {
            'splits': ('train', 'val', 'test'),
            'params': {'dataset': 'refcocog', 'split_by': 'unc'}
        },
        'grefcoco': {
            'splits': ('train', 'val', 'testA', 'testB'),
            'params': {'dataset': 'grefcoco', 'split_by': 'unc'}
        }
    }

    def __init__(self, data_root, split_root='data', dataset='refcoco', imsize=256,
                 transform=None, augment=False, split='train', max_query_len=128, 
                 bert_model='bert-base-uncased'):
        self.images = []
        self.data_root = data_root
        self.split_root = split_root
        self.dataset = dataset
        self.imsize = imsize
        self.query_len = max_query_len
        self.transform = transform
        self.split = split
        self.tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=True) # should be true for English
        self.augment=augment

        valid_splits = self.SUPPORTED_DATASETS[self.dataset]['splits']

        if split not in valid_splits:
            raise ValueError(
                'Dataset {0} does not have split {1}'.format(
                    self.dataset, split))
        
        self.anns_root = osp.join(self.data_root, 'anns', self.dataset, self.split+'.txt')
        self.mask_root = osp.join(self.data_root, 'masks', self.dataset)
        self.im_dir = osp.join(self.data_root, 'images', 'train2014')
        
        dataset_path = osp.join(self.split_root, self.dataset)
        splits = [split]
        for split in splits:
            imgset_file = '{0}_{1}.pth'.format(self.dataset, split)
            imgset_path = osp.join(dataset_path, imgset_file)
            self.images += torch.load(imgset_path)
        
    def exists_dataset(self):
        return osp.exists(osp.join(self.split_root, self.dataset))

    def pull_item(self, idx):
        img_file, seg_id, bbox, phrase = self.images[idx]
        bbox = np.array(bbox, dtype=int) # x1y1x2y2

        img_path = osp.join(self.im_dir, img_file)
        img = cv2.imread(img_path) # BGR [512, 640, 3]
        ## duplicate channel if gray image
        if img.shape[-1] > 1:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #RGB
        else:
            img = np.stack([img] * 3)
        
        ## seg map
        seg_map = np.load(osp.join(self.mask_root, str(seg_id)+'.npy')) # [512, 640]
        seg_map = np.array(seg_map).astype(np.float32)
        return img, phrase, bbox, seg_map

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img, phrase, bbox, seg_map = self.pull_item(idx)
        phrase = phrase.lower()
        if self.augment:
            augment_flip, augment_hsv, augment_affine, augment_crop, augment_copy, augment_erase = \
                    True,        True,           True,        False,        False,          False

        ## seems a bug in torch transformation resize, so separate in advance
        h,w = img.shape[0], img.shape[1]
        # print("img.shape", img.shape)
        if self.augment:
            ## random horizontal flip
            if augment_flip and random.random() > 0.5:
                img = cv2.flip(img, 1) 
                seg_map = cv2.flip(seg_map, 1) 
                bbox[0], bbox[2] = w-bbox[2]-1, w-bbox[0]-1
                phrase = phrase.replace('right','*&^special^&*').replace('left','right').replace('*&^special^&*','left')

            ## random copy and add left or right
            if augment_copy:
                img, seg_map, phrase, bbox = random_copy(img, seg_map, phrase, bbox)

            ## random erase for occluded
            if augment_erase:
                img, seg_map = random_erase(img, seg_map)

            ## random padding and crop
            if augment_crop:
                img, seg_map = random_crop(img, seg_map, 40, h, w)

            ## random intensity, saturation change
            if augment_hsv:
                fraction = 0.50
                img_hsv = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2HSV)
                S = img_hsv[:, :, 1].astype(np.float32)
                V = img_hsv[:, :, 2].astype(np.float32)
                a = (random.random() * 2 - 1) * fraction + 1
                if a > 1:
                    np.clip(S, a_min=0, a_max=255, out=S)
                a = (random.random() * 2 - 1) * fraction + 1
                V *= a
                if a > 1:
                    np.clip(V, a_min=0, a_max=255, out=V)

                img_hsv[:, :, 1] = S.astype(np.uint8)
                img_hsv[:, :, 2] = V.astype(np.uint8)
                img = cv2.cvtColor(cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR), cv2.COLOR_BGR2RGB)

            img, seg_map, ratio, dw, dh = letterbox(img, seg_map, self.imsize)
            bbox[0], bbox[2] = bbox[0]*ratio+dw, bbox[2]*ratio+dw
            bbox[1], bbox[3] = bbox[1]*ratio+dh, bbox[3]*ratio+dh

            ## random affine transformation
            if augment_affine:
                img, seg_map, bbox, M = random_affine(img, seg_map, bbox, \
                    degrees=(-5, 5), translate=(0.10, 0.10), scale=(0.90, 1.10)) # 255 white fill

        else:   ## should be inference, or specified training
            img, _, ratio, dw, dh = letterbox(img, None, self.imsize)
            bbox[0], bbox[2] = bbox[0]*ratio+dw, bbox[2]*ratio+dw
            bbox[1], bbox[3] = bbox[1]*ratio+dh, bbox[3]*ratio+dh

        draw_img = copy.deepcopy(img)
        # Norm, to tensor
        if self.transform is not None:
            img = self.transform(img)

        ## encode phrase to clip input
        word_id = clip.tokenize(phrase, 17, truncate=True)
        word_mask = ~ (word_id == 0)

        if self.augment: # train
            seg_map = cv2.resize(seg_map, (self.imsize // 2, self.imsize // 2),interpolation=cv2.INTER_NEAREST) # (208, 208)
            seg_map = np.reshape(seg_map, [1, np.shape(seg_map)[0], np.shape(seg_map)[1]])
            return img, np.array(word_id, dtype=int), np.array(word_mask, dtype=int), \
            np.array(bbox, dtype=np.float32), np.array(seg_map, dtype=np.float32)
        else:
            seg_map = np.reshape(seg_map, [1, np.shape(seg_map)[0], np.shape(seg_map)[1]])
            return img, np.array(word_id, dtype=int), np.array(word_mask, dtype=int), \
            np.array(bbox, dtype=np.float32), np.array(seg_map, dtype=np.float32), np.array(ratio, dtype=np.float32), \
            np.array(dw, dtype=np.float32), np.array(dh, dtype=np.float32), self.images[idx][0], self.images[idx][3], np.array(draw_img, dtype=np.uint8)
