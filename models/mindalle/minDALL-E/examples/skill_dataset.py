import os
import sys
import argparse
from typing import Optional
from datetime import datetime

from pathlib import Path
import json
import PIL

import torch
from torch.utils import data
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms

class SkillTextImageDataset(Dataset):
    def __init__(self,
                 image_dir='/playpen3/home/jmincho/workspace/datasets/PaintSkills/object/images',
                 text_data_file='/playpen3/home/jmincho/workspace/datasets/PaintSkills/object/scenes/object_train.json',
                 text_len=64,
                 image_resolution=256,
                 tokenizer=None,
                 load_image=True,
                 ):
        super().__init__()
        transform = transforms.Compose(
            [transforms.Resize((image_resolution, image_resolution)),
             #  transforms.RandomCrop(image_resolution),
             transforms.ToTensor(),
             transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
        )

        self.image_transform = transform
        self.image_dir = Path(image_dir).resolve()

        with open(text_data_file, 'r') as f:
            _text_data = json.load(f)['data']
            print('Loaded text data from {}'.format(text_data_file))

        self.text_data = _text_data
        self.text_len = text_len

        self.tokenizer = tokenizer
        self.load_image = load_image

    def __len__(self):
        return len(self.text_data)

    def __getitem__(self, ind):
        datum = self.text_data[ind]

        text = datum['text']
        tokens = self.tokenizer.encode(text)
        input_ids = torch.LongTensor(tokens.ids)

        if self.load_image:
            img_fname = f"image_{datum['id']}"
            img_path = self.image_dir.joinpath(img_fname).with_suffix('.png')

            image_tensor = self.image_transform(PIL.Image.open(img_path))

            return image_tensor, input_ids
        else:
            datum['input_ids'] = input_ids
            return datum

    def text_collate_fn(self, batch):

        B = len(batch)
        L = max([len(b['tokens']) for b in batch])

        batch_datum = {
            'id': [],
            'text': [],
            'input_ids': torch.LongTensor(B, L),
        }
        for i, datum in enumerate(batch):

            batch_datum['id'].append(datum['id'])
            batch_datum['text'].append(datum['text'])
            batch_datum['input_ids'][i] = datum['input_ids']

        return batch_datum
