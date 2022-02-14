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

class COCOTextImageDataset(Dataset):
    def __init__(self,
                split='karpathy_test',
                image_dir='/playpen3/home/jmincho/workspace/datasets/COCO/images/',
                text_data_file='/playpen3/home/jmincho/workspace/datasets/COCO/dataset_coco.json',
                text_len=256,
                image_size=128,
                truncate_captions=False,
                resize_ratio=0.75,
                tokenizer=None,
                shuffle=False,
                load_image=False,
                ):
        """
        @param folder: Folder containing images and text files matched by their paths' respective "stem"
        @param truncate_captions: Rather than throw an exception, captions which are too long will be truncated.
        """
        super().__init__()
        self.shuffle = shuffle

        self.split = split

        with open(text_data_file, 'r') as f:
            karpathy_data = json.load(f)
            print('Loaded text data from {}'.format(text_data_file))

        self.load_image = load_image

        if self.load_image:
            self.image_dir = Path(image_dir).resolve()

        self.text_len = text_len
        self.truncate_captions = truncate_captions
        self.resize_ratio = resize_ratio
        self.tokenizer = tokenizer
        if self.load_image:
            self.image_transform = T.Compose([
                T.Lambda(lambda img: img.convert('RGB')
                if img.mode != 'RGB' else img),
                T.RandomResizedCrop(image_size,
                                    scale=(self.resize_ratio, 1.),
                                    ratio=(1., 1.)),
                T.ToTensor()
            ])

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

        self.text_data = data

        self.keys = list(range(len(self.text_data)))

    def __len__(self):
        return len(self.keys)

    def random_sample(self):
        return self.__getitem__(randint(0, self.__len__() - 1))

    def sequential_sample(self, ind):
        if ind >= self.__len__() - 1:
            return self.__getitem__(0)
        return self.__getitem__(ind + 1)

    def skip_sample(self, ind):
        if self.shuffle:
            return self.random_sample()
        return self.sequential_sample(ind=ind)

    def __getitem__(self, ind):
        key = self.keys[ind]

        datum = self.text_data[key]

        caption = datum['targets'][0]

        tokenized_text = self.tokenizer.tokenize(
            caption,
            self.text_len,
            truncate_text=self.truncate_captions
        ).squeeze(0)

        if self.load_image:
            if 'val2014' in datum['img_id']:
                img_path = self.image_dir / 'val2014' / datum['filename']
            elif 'train2014' in datum['img_id']:
                img_path = self.image_dir / 'train2014' / datum['filename']

            image_tensor = self.image_transform(PIL.Image.open(img_path))

            return tokenized_text, image_tensor
        else:
            out = {
                'img_id': datum['img_id'],
                'caption': caption,
                'filename': datum['filename'],
                'tokenized_text': tokenized_text
            }
            return out

    def text_collate_fn(self, batch):
        B = len(batch)
        L = max([len(b['tokenized_text']) for b in batch])

        batch_datum = {
            'img_id': [],
            'filenames': [],
            'caption': [],
            'tokenized_text': torch.LongTensor(B, L),
        }
        if self.load_image:
            batch_datum['img_path'] = []

        for i, datum in enumerate(batch):
            batch_datum['img_id'].append(datum['img_id'])
            batch_datum['filenames'].append(datum['filename'])
            batch_datum['caption'].append(datum['caption'])
            batch_datum['tokenized_text'][i] = datum['tokenized_text']

            if self.load_image:


                batch_datum['img_path'].append(datum['img_path'])

        return batch_datum


class COCO30KDataset(Dataset):
    def __init__(self,
                 data_path='../IS_FID/uid_caption.csv',
                 proc_id=-1,
                 n_proc=-1,
                text_len=256,
                image_size=128,
                truncate_captions=False,
                resize_ratio=0.75,
                tokenizer=None,
                shuffle=False,
                load_image=False,
                ):
        """
        @param folder: Folder containing images and text files matched by their paths' respective "stem"
        @param truncate_captions: Rather than throw an exception, captions which are too long will be truncated.
        """
        super().__init__()
        self.shuffle = shuffle

        self.load_image = load_image

        uid_caption_path = Path(data_path).resolve()
        import pandas as pd

        assert uid_caption_path.is_file(), uid_caption_path
        print('Load from:', uid_caption_path)
        df = pd.read_csv(uid_caption_path)
        assert len(df) == 30000, len(df)
        captions = df['caption']
        uids = df['uid']

        self.text_len = text_len
        self.truncate_captions = truncate_captions
        self.resize_ratio = resize_ratio
        self.tokenizer = tokenizer
        if self.load_image:
            self.image_transform = T.Compose([
                T.Lambda(lambda img: img.convert('RGB')
                if img.mode != 'RGB' else img),
                T.RandomResizedCrop(image_size,
                                    scale=(self.resize_ratio, 1.),
                                    ratio=(1., 1.)),
                T.ToTensor()
            ])

        captions = captions.tolist()
        uids = uids.tolist()

        if n_proc > 0:
            print('total ', len(captions), 'data')
            captions = captions[proc_id::n_proc]
            uids = uids[proc_id::n_proc]
            print('sampled ', len(captions), 'data')

        self.text_data = captions
        self.uids = uids

        self.keys = list(range(len(self.text_data)))

        print('length:', len(self.keys))

    def __len__(self):
        return len(self.keys)

    def random_sample(self):
        return self.__getitem__(randint(0, self.__len__() - 1))

    def sequential_sample(self, ind):
        if ind >= self.__len__() - 1:
            return self.__getitem__(0)
        return self.__getitem__(ind + 1)

    def skip_sample(self, ind):
        if self.shuffle:
            return self.random_sample()
        return self.sequential_sample(ind=ind)

    def __getitem__(self, ind):
        key = self.keys[ind]

        # datum = self.text_data[key]
        caption = self.text_data[key]
        uid = self.uids[key]

        # caption = datum['targets'][0]

        tokenized_text = self.tokenizer.tokenize(
            caption,
            self.text_len,
            truncate_text=self.truncate_captions
        ).squeeze(0)

        if self.load_image:

            pass
        else:
            filename = uid + '.jpg'

            out = {
                'img_id': uid,
                'caption': caption,
                'filename': filename,
                'tokenized_text': tokenized_text
            }
            return out

    def text_collate_fn(self, batch):
        B = len(batch)
        L = max([len(b['tokenized_text']) for b in batch])

        batch_datum = {
            'img_id': [],
            'filenames': [],
            'caption': [],
            'tokenized_text': torch.LongTensor(B, L),
        }
        if self.load_image:
            batch_datum['img_path'] = []

        for i, datum in enumerate(batch):
            batch_datum['img_id'].append(datum['img_id'])
            batch_datum['filenames'].append(datum['filename'])
            batch_datum['caption'].append(datum['caption'])
            batch_datum['tokenized_text'][i] = datum['tokenized_text']

            if self.load_image:
                batch_datum['img_path'].append(datum['img_path'])

        return batch_datum
