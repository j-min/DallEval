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



class SocialDataset(Dataset):
    def __init__(self,
                 tokenizer=None,
                 ):
        """
        @param folder: Folder containing images and text files matched by their paths' respective "stem"
        @param truncate_captions: Rather than throw an exception, captions which are too long will be truncated.
        """
        super().__init__()

        social_data_dir = Path(__file__).resolve().parent.joinpath('social_bias')

        prompts = []

        for prompt_name in ['objects', 'politics', 'profession', 'other']:
            with open(social_data_dir.joinpath(f"prompts_{prompt_name}.json"), 'r') as f:
                prompt_data = json.load(f)

                prompts += prompt_data['classifier_prompts']
                prompts += prompt_data['bias_prompts']


        self.data = prompts
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, ind):

        caption = self.data[ind]

        tokens = self.tokenizer.encode(caption)
        input_ids = torch.LongTensor(tokens.ids)

        out = {
            'caption': caption + ' person',
            'input_ids': input_ids
        }
        return out

    def text_collate_fn(self, batch):
        B = len(batch)
        L = max([len(b['input_ids']) for b in batch])

        batch_datum = {
            'caption': [],
            'input_ids': torch.LongTensor(B, L),
        }

        for i, datum in enumerate(batch):
            batch_datum['caption'].append(datum['caption'])
            batch_datum['input_ids'][i] = datum['input_ids']

        return batch_datum
