#@markdown Data preparation
from random import randint, choice
import io
import os
import PIL
import random
import numpy as np
import torch
import torchvision
import transformers
import more_itertools
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm
from dataclasses import dataclass, field
import torchvision.transforms as T
import torchvision.transforms.functional as TF


class RuDalleDataset(Dataset):
    clip_filter_thr = 0.24
    def __init__(
            self,
            file_path,
            csv_path,
            tokenizer,
            resize_ratio=0.75,
            shuffle=True,
            load_first=None,
            caption_score_thr=0.6,
            text_seq_length=None,
            device=None,
    ):
        """ tokenizer - объект с методами tokenizer_wrapper.BaseTokenizerWrapper """

#         self.text_seq_length = model.get_param('text_seq_length')
        self.text_seq_length = text_seq_length
        self.tokenizer = tokenizer
        self.target_image_size = 256
        self.image_size = 256
        self.samples = []

        self.device = device


        self.image_transform = T.Compose([
                T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
                T.RandomResizedCrop(self.image_size,
                                    scale=(1., 1.), # в train было scale=(0.75., 1.),
                                    ratio=(1., 1.)),
                T.ToTensor()
            ])

        df = pd.read_csv(csv_path)
        for caption, f_path  in zip(df['caption'], df['name']):
            if len(caption)>10 and len(caption)<100 and os.path.isfile(f'{file_path}/{f_path}'):

              self.samples.append([file_path, f_path, caption])
        if shuffle:
            np.random.shuffle(self.samples)
            print('Shuffled')

    def __len__(self):
        return len(self.samples)

    def load_image(self, file_path, img_name):
        image = PIL.Image.open(f'{file_path}/{img_name}')
        return image

    def __getitem__(self, item):
        item = item % len(self.samples)  # infinite loop, modulo dataset size
        file_path, img_name, text = self.samples[item]
        try:
          image = self.load_image(file_path, img_name)
          image = self.image_transform(image) #.to(device)
        except Exception as err:  # noqa
            print(err)
            random_item = random.randint(0, len(self.samples) - 1)
            return self.__getitem__(random_item)
        text =  self.tokenizer.encode_text(text, text_seq_length=self.text_seq_length).squeeze(0) #.to(device)

        if self.device is not None:
            image = image.to(self.device)
            text = text.to(self.device)

        return text, image


import json
from pathlib import Path

class SkillTextImageDataset(Dataset):
    def __init__(self,
                args=None,
                 skill_name='object',
                 split='train',
                 image_dir='/playpen3/home/jmincho/workspace/datasets/PaintSkills/object/images',
                 text_data_file='/playpen3/home/jmincho/workspace/datasets/PaintSkills/object/scenes/object_train.json',
                 text_len=256,
                 image_size=128,
                 truncate_captions=False,
                 resize_ratio=0.75,
                 tokenizer=None,
                 shuffle=False,
                 load_image=True,
                 ):
        """
        @param folder: Folder containing images and text files matched by their paths' respective "stem"
        @param truncate_captions: Rather than throw an exception, captions which are too long will be truncated.
        """
        super().__init__()
        self.args = args


        self.shuffle = shuffle

        self.split = split

        self.load_image = load_image

        if self.load_image:
            self.image_dir = Path(image_dir).resolve()

        with open(text_data_file, 'r') as f:
            _text_data = json.load(f)['data']
            print('Loaded text data from {}'.format(text_data_file))

        self.text_data = _text_data
        self.keys = list(range(len(self.text_data)))

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

        if self.load_image:
            if self.paintskill_real:
                # img_fname = datum['questionId']
                img_fname = datum['id']
                img_path = self.image_dir.joinpath(
                    self.split).joinpath(img_fname).with_suffix('.jpg')
            else:
                # img_fname = f"image_{datum['questionId']}"
                img_fname = f"image_{datum['id']}"
                img_path = self.image_dir.joinpath(
                    img_fname).with_suffix('.png')

        text = datum['ru_text']
        description = text

        tokenized_text = self.tokenizer.encode_text(
            description,
            text_seq_length=self.text_len,
            # truncate_text=self.truncate_captions
        ).squeeze(0)

        if self.load_image:
            try:
                image_tensor = self.image_transform(PIL.Image.open(img_path))
            except (PIL.UnidentifiedImageError, OSError) as corrupt_image_exceptions:
                print(
                    f"An exception occurred trying to load file {img_path}.")
                print(f"Skipping index {ind}")
                return self.skip_sample(ind)

            return tokenized_text, image_tensor
        else:
            datum['input_ids'] = tokenized_text
            return datum

    def text_collate_fn(self, batch):
        B = len(batch)
        L = max([len(b['input_ids']) for b in batch])

        batch_datum = {
            'id': [],
            'text': [],
            'input_ids': torch.LongTensor(B, L).fill_(0),
        }
        for i, datum in enumerate(batch):
            batch_datum['id'].append(datum['id'])
            batch_datum['text'].append(datum['ru_text'])
            batch_datum['input_ids'][i][:len(datum['input_ids'])] = datum['input_ids']

        return batch_datum
