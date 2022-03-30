
from tqdm import tqdm
from pathlib import Path
import pandas as pd
from PIL import Image

import random
random.seed(10)


import argparse
import numpy as np

from torch import nn
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets.coco import CocoCaptions

import clip

class RetrievalDataset(Dataset):
    def __init__(self,
                 uid_caption_path='uid_caption.csv',
                 img_dir='../IS_FID/COCO30K/DALLE_CC_zero/',
                 transform=None,
                 is_gt=False,
                 coco_img_dir='../../../datasets/COCO/images/val2014/',
                ):
        df = pd.read_csv(uid_caption_path)
        self.df = df

        self.img_dir = Path(img_dir)

        if transform is None:
            transform = preprocess

        self.transform = transform

        self.is_gt = is_gt
        self.coco_img_dir = Path(coco_img_dir)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, ix):
        row = self.df.iloc[ix]
        uid = row.uid
        caption =  row.caption

        if self.is_gt:
            coco_img_id = uid.split('_')[0]
            img_fname = f"COCO_val2014_{coco_img_id.zfill(12)}.jpg"
            img_path = self.coco_img_dir.joinpath(img_fname)
        else:
            img_path = self.img_dir.joinpath(f'{uid}.jpg')

        img = Image.open(img_path).convert('RGB')

        img_tensor = self.transform(img)

        return img_tensor, caption

    def collate_fn(self, batch):
        B = len(batch)

        images = torch.zeros(B, 3, 224,224)

        captions = []

        for i, (img, caption) in enumerate(batch):
            images[i] = img
            captions.append(caption)

        tokens = clip.tokenize(captions)

        return images, tokens


def forward_calc_retrieval_score(
    model_image_dir,
    coco_image_dir,
    uid_caption_path,
    batch_size=60,
    K=100, # 1 positive + K-1 negative captions for each (generated) image
    repeat=5):

    if model_name == 'gt':
        dataset = RetrievalDataset(
            is_gt=True,
            coco_img_dir=coco_image_dir,
            uid_caption_path=uid_caption_path,
        )
    else:
        dataset = RetrievalDataset(
            img_dir=model_image_dir,
            coco_img_dir=coco_image_dir,
            uid_caption_path=uid_caption_path,
            )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=4,
        collate_fn=dataset.collate_fn,
    )

    # fwd all samples
    image_features = []
    text_features = []
    for batch_idx, batch in enumerate(tqdm(dataloader)):
        images, texts = batch
        texts = texts.cuda()

        text_emb = model.encode_text(texts)  # embed with text encoder
        images = images.cuda()
        image_emb = model.encode_image(images)  # embed with image encoder

        text_features.append(text_emb.detach().cpu())
        image_features.append(image_emb.detach().cpu())

    image_features = torch.cat(image_features, 0)
    text_features = torch.cat(text_features, 0)
    print('Done forward')

    all_indices = list(range(len(image_features)))

    repeat_results = []

    avg_results = np.zeros((len(image_features), repeat))

    for j in range(repeat):
        print('repeat ', j)
        n_retrieved = 0

        for i, img_feat in enumerate(tqdm(image_features)):
            pos_text_feat = text_features[i]
            neg_text_indices = torch.LongTensor(
                random.choices(all_indices, k=K-1))
            neg_text_feats = text_features.index_select(0, neg_text_indices)

            text_candidates = torch.cat([
                pos_text_feat.view(1, -1),
                neg_text_feats
            ])

            sim_scores = img_feat @ text_candidates.T

            if 0 == sim_scores.argmax().item():
                n_retrieved += 1

                avg_results[i, j] = 1

        print(n_retrieved)
        R_precision = n_retrieved / len(image_features)
        print(R_precision)
        repeat_results.append(R_precision)

    avg_results = np.mean(avg_results, axis=1)

    print('repeat average:', (sum(repeat_results) / len(repeat_results)))
    return avg_results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--uid_caption_path', type=str, default='uid_caption.csv', help='path to the list of 30K image-caption pairs')
    parser.add_argument('--image_dir', type=str, default='dalle_small', help='path to generated image directory')
    parser.add_argument('--coco_image_dir', type=str, default='../../../datasets/COCO/images/val2014/',
                        help='path to coco val 2014 image dir')
    args = parser.parse_args()
    print(args)

    single_caption = True  # choose if evalating only using the first caption
    model_name = "ViT-B/32"  # "RN50" #"RN50x4" #"RN101" #

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(model_name, device=device)
    print(f"Loaded CLIP retrieval {model_name} at {device}")

    forward_calc_retrieval_score(
        args.image_dir,
        args.coco_image_dir,
        args.uid_caption_path)
