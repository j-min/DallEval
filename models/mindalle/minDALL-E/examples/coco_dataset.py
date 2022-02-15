from pathlib import Path
from random import randint, choice
from re import I

import PIL
from PIL.Image import ImageTransformHandler

from torch.utils.data import Dataset
from torchvision import transforms as T

import json
from copy import deepcopy
import numpy as np
import torch

import torch
from torch.utils import data
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms

class COCOTextImageDataset(Dataset):
    def __init__(self,
                 split='karpathy_test',
                 image_dir='/playpen3/home/jmincho/workspace/datasets/COCO/images/',
                 text_data_file='/playpen3/home/jmincho/workspace/datasets/COCO/dataset_coco.json',
                 text_len=64,
                 image_resolution=256,
                 tokenizer=None,
                 load_image=False,
                 proc_id=-1,
                 n_proc=-1,
                 ):
        """
        @param folder: Folder containing images and text files matched by their paths' respective "stem"
        @param truncate_captions: Rather than throw an exception, captions which are too long will be truncated.
        """
        super().__init__()

        transform = transforms.Compose(
            [transforms.Resize((image_resolution, image_resolution)),
             #  transforms.RandomCrop(image_resolution),
             transforms.ToTensor(),
             transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
        )

        self.image_transform = transform
        self.image_dir = Path(image_dir).resolve()

        self.text_len = text_len

        self.tokenizer = tokenizer
        self.load_image = load_image

        self.split = split

        with open(text_data_file, 'r') as f:
            karpathy_data = json.load(f)
            print('Loaded text data from {}'.format(text_data_file))

        self.load_image = load_image

        if self.load_image:
            self.image_dir = Path(image_dir).resolve()

        split_rename = {
            'train': 'train',
            'restval': 'train',
            'val': 'val',
            'test': 'test'
        }

        karpathy_split_name = self.split.split('_')[-1]

        data = []
        for datum in karpathy_data['images']:
            re_split = split_rename[datum['split']]
            if re_split != karpathy_split_name:
                continue

            if re_split == 'train':
                for d in datum['sentences']:
                    img_id = datum['filename'].split('.')[0]
                    new_datum = {
                        'filename': datum['filename'],
                        'img_id': img_id,
                        'sent': d['raw'].strip(),
                        'targets': [d['raw'].strip() for d in datum['sentences']],
                        'is_train': True,
                    }
                    data.append(new_datum)
            else:
                img_id = datum['filename'].split('.')[0]
                new_datum = {
                    'filename': datum['filename'],
                    'img_id': img_id,
                    # 'sent': d['raw'],
                    'targets': [d['raw'].strip() for d in datum['sentences']],
                    'is_train': False,
                }
                data.append(new_datum)
        if n_proc > 1:
            data = data[proc_id::n_proc]

        self.text_data = data

    def __len__(self):
        return len(self.text_data)

    def __getitem__(self, ind):
        datum = self.text_data[ind]

        caption = datum['targets'][0]

        tokens = self.tokenizer.encode(caption)
        input_ids = torch.LongTensor(tokens.ids)

        if self.load_image:
            if 'val2014' in datum['img_id']:
                img_path = self.image_dir / 'val2014' / datum['filename']
            elif 'train2014' in datum['img_id']:
                img_path = self.image_dir / 'train2014' / datum['filename']

            image_tensor = self.image_transform(PIL.Image.open(img_path))

            return image_tensor, input_ids
        else:
            out = {
                'img_id': datum['img_id'],
                'caption': caption,
                'filename': datum['filename'],
                'input_ids': input_ids
            }
            return out

    def text_collate_fn(self, batch):
        B = len(batch)
        L = max([len(b['input_ids']) for b in batch])

        batch_datum = {
            'img_id': [],
            'filenames': [],
            'caption': [],
            'input_ids': torch.LongTensor(B, L),
        }
        if self.load_image:
            batch_datum['img_path'] = []

        for i, datum in enumerate(batch):
            batch_datum['img_id'].append(datum['img_id'])
            batch_datum['filenames'].append(datum['filename'])
            batch_datum['caption'].append(datum['caption'])
            batch_datum['input_ids'][i] = datum['input_ids']

            if self.load_image:

                batch_datum['img_path'].append(datum['img_path'])

        return batch_datum


class COCO30KDataset(Dataset):
    def __init__(self,
                 data_path='../IS_FID/uid_caption.csv',
                 proc_id=-1,
                 n_proc=-1,
                 text_len=64,
                 image_resolution=256,
                 tokenizer=None,
                 load_image=False,
                 ):
        """
        @param folder: Folder containing images and text files matched by their paths' respective "stem"
        @param truncate_captions: Rather than throw an exception, captions which are too long will be truncated.
        """
        super().__init__()

        uid_caption_path = Path(data_path).resolve()
        import pandas as pd

        assert uid_caption_path.is_file(), uid_caption_path
        print('Load from:', uid_caption_path)
        df = pd.read_csv(uid_caption_path)
        assert len(df) == 30000, len(df)
        captions = df['caption'].tolist()
        uids = df['uid'].tolist()

        if n_proc > 1:
            print('total ', len(captions), 'data')
            captions = captions[proc_id::n_proc]
            uids = uids[proc_id::n_proc]
            print('sampled ', len(captions), 'data')

        transform = transforms.Compose(
            [transforms.Resize((image_resolution, image_resolution)),
             #  transforms.RandomCrop(image_resolution),
             transforms.ToTensor(),
             transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
        )

        self.image_transform = transform

        self.text_len = text_len

        self.tokenizer = tokenizer
        self.load_image = load_image

        self.text_data = captions
        self.uids = uids

    def __len__(self):
        return len(self.text_data)

    def __getitem__(self, ind):
        caption = self.text_data[ind]
        uid = self.uids[ind]

        tokens = self.tokenizer.encode(caption)
        input_ids = torch.LongTensor(tokens.ids)

        filename = uid + '.jpg'

        out = {
            'img_id': uid,
            'caption': caption,
            'filename': filename,
            'input_ids': input_ids
        }
        return out

    def text_collate_fn(self, batch):
        B = len(batch)
        L = max([len(b['input_ids']) for b in batch])

        batch_datum = {
            'img_id': [],
            'filenames': [],
            'caption': [],
            'input_ids': torch.LongTensor(B, L),
        }

        for i, datum in enumerate(batch):
            batch_datum['img_id'].append(datum['img_id'])
            batch_datum['filenames'].append(datum['filename'])
            batch_datum['caption'].append(datum['caption'])
            batch_datum['input_ids'][i] = datum['input_ids']

        return batch_datum
