from pathlib import Path
from random import randint, choice
from re import I

import PIL

from torch.utils.data import Dataset
from torchvision import transforms as T

import json
from copy import deepcopy
import numpy as np
import torch

class SkillTextImageDataset(Dataset):
    def __init__(self,
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

        # text_file = self.text_files[key]
        # image_file = self.image_files[key]

        # {
        #     "id": "Object_train_00000",
        #     "scene": "empty",
        #     "text": "a photo of human",
        #     "answer": 2,
        #     "objects": [
        #         {
        #             "id": 0,
        #             "shape": "humanSophie",
        #             "color": "plain",
        #             "relation": null,
        #             "scale": 4.267770484580929,
        #             "texture": "plain",
        #             "rotation": null,
        #             "state": "sitting"
        #         }
        #     ]
        # },

        datum = self.text_data[key]

        if self.load_image:
            img_fname = f"image_{datum['id']}"
            img_path = self.image_dir.joinpath(img_fname).with_suffix('.png')

        text = datum['text']
        description = text

        tokenized_text = self.tokenizer.tokenize(
            description,
            self.text_len,
            truncate_text=self.truncate_captions
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
            return datum, tokenized_text

    def text_collate_fn(self, batch):
        B = len(batch)
        L = max([len(b[1]) for b in batch])

        batch_datum = {
            'id': [],
            'text': [],
            'tokenized_text': torch.LongTensor(B, L),
        }
        for i, (datum, tokenized_text) in enumerate(batch):
            batch_datum['id'].append(datum['id'])
            batch_datum['text'].append(datum['text'])
            batch_datum['tokenized_text'][i] = tokenized_text

        return batch_datum
