# ------------------------------------------------------------------------------------
# Minimal DALL-E
# Copyright (c) 2021 KakaoBrain. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------

import os
import sys
import argparse
# import clip
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import json
import pandas as pd
# import more_itertools
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dalle.models import Dalle
from dalle.utils.utils import set_seed, clip_score
from skill_dataset import SkillTextImageDataset
from coco_dataset import COCOTextImageDataset, COCO30KDataset
from social_dataset import SocialDataset

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--num_candidates', type=int, default=1)
    parser.add_argument('--prompt', type=str, default='A painting of a tree on the ocean')
    parser.add_argument('--softmax-temperature', type=float, default=1.0)
    parser.add_argument('--top-k', type=int, default=256)
    parser.add_argument('--top-p', type=float, default=None, help='0.0 <= top-p <= 1.0')
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--image_resolution', type=int, default=256)

    parser.add_argument('--dataset_dir', type=str, default=None)
    parser.add_argument('--skill_name', type=str, default=None)

    parser.add_argument('--split', type=str, default='val')

    parser.add_argument('--data', type=str, choices=['paintskills', 'COCO', 'COCO30K', 'social'], default='paintskills')

    parser.add_argument('--ckpt_path', type=str, default=None)

    parser.add_argument('--image_dump_dir', default='/../../../datasets/PaintSkills/minDALLE_inference')

    parser.add_argument('--batch_size', type=int, default=8)

    parser.add_argument('--proc_id', type=int, default=-1)
    parser.add_argument('--n_proc', type=int, default=-1)

    args = parser.parse_args()
    print(args)

    # Setup
    assert args.top_k <= 256, "It is recommended that top_k is set lower than 256."

    set_seed(args.seed)

    model = Dalle.from_pretrained('minDALL-E/1.3B')  # This will automatically download the pretrained model.

    device = 'cuda'

    # Load checkpoint
    if args.ckpt_path is not None:
        ckpt = torch.load(args.ckpt_path, map_location=device)
        model.load_state_dict(ckpt['state_dict'])
        model.to(device=device)
        print('Model loaded from {}'.format(args.ckpt_path))
    else:
        model.to(device=device)
        print('Model loaded from pretrained weights.')

    if args.data == 'paintskills':
        dataset_dir = Path(args.dataset_dir).resolve()

        skill = args.skill_name

        image_dir = dataset_dir.joinpath(f'{skill}/images')
        text_data_file = dataset_dir.joinpath(f'{skill}/scenes/{skill}_val.json')

        dataset = SkillTextImageDataset(
            image_dir=image_dir,
            text_data_file=text_data_file,
            image_resolution=args.image_resolution,
            tokenizer=model.tokenizer,
            load_image=False,
            )

        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=dataset.text_collate_fn,
            num_workers=4,
        )

        if args.ckpt_path is not None:
            run_name = 'CC'
        else:
            run_name = 'CCzero'

        output_dir = Path(args.image_dump_dir).joinpath(f'{args.skill_name}_{run_name}_{args.split}')

        if not output_dir.is_dir():
            output_dir.mkdir(parents=True)
        print('Dump images in ', output_dir)

        desc = f'{args.ckpt_path}-{args.skill_name}-{args.split}'

        for batch in tqdm(loader, desc=desc):
            text_tokens = batch['input_ids']
            text_tokens = text_tokens.to(device)

            B = len(text_tokens)

            # prompt = text_tokens

            images = model.sampling(prompt="",
                                    tokens=text_tokens,
                                    top_k=args.top_k,
                                    top_p=args.top_p,
                                    softmax_temperature=args.softmax_temperature,
                                    # num_candidates=args.num_candidates,
                                    num_candidates=B,
                                    device=device,
                                    is_tqdm=False,
                                    ).cpu().numpy()
            images = np.transpose(images, (0, 2, 3, 1))

            text_ids = batch['id']

            for i in range(len(images)):
                im = Image.fromarray((images[i]*255).astype(np.uint8))
                out_fname = output_dir.joinpath(f'{text_ids[i]}.png')
                im.save(out_fname)

    elif args.data == 'COCO':
        coco_dir = Path('../../../datasets/COCO/').resolve()

        ann_path = coco_dir.joinpath('dataset_coco.json')
        # ann_path = coco_dir.joinpath('dataset_coco_karpathy_ru.json')
        # print('Loading text from', ann_path)
        # cap_ann_data = json.load(open(ann_path))

        assert args.split == 'karpathy_test', args.split

        image_dump_dir = coco_dir.joinpath('mindalle_inference').joinpath(args.split)
        image_dump_dir.mkdir(exist_ok=True, parents=True)
        print('Image dump at: ', image_dump_dir)

        output_dir = image_dump_dir

        # COCO
        text_data_file = ann_path

        dataset = COCOTextImageDataset(
            text_data_file=text_data_file,
            image_resolution=args.image_resolution,
            tokenizer=model.tokenizer,
            load_image=False,
            proc_id=args.proc_id,
            n_proc=args.n_proc,
        )

        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=dataset.text_collate_fn,
            num_workers=4,
        )

        desc = f'Mindalle {ann_path}: {args.proc_id} out of {args.n_proc} procs'

        for batch in tqdm(loader, desc=desc):
            text_tokens = batch['input_ids']
            text_tokens = text_tokens.to(device)

            B = len(text_tokens)

            images = model.sampling(prompt="",
                                    tokens=text_tokens,
                                    top_k=args.top_k,
                                    top_p=args.top_p,
                                    softmax_temperature=args.softmax_temperature,
                                    # num_candidates=args.num_candidates,
                                    num_candidates=B,
                                    device=device,
                                    is_tqdm=False,
                                    ).cpu().numpy()
            images = np.transpose(images, (0, 2, 3, 1))

            filenames = batch['filenames']

            for i in range(len(images)):
                im = Image.fromarray((images[i]*255).astype(np.uint8))
                out_fname = output_dir.joinpath(filenames[i])
                im.save(out_fname)

    elif args.data == 'COCO30K':

        uid_caption_path = Path('../IS_FID/uid_caption.csv')

        dataset = COCO30KDataset(
            data_path = uid_caption_path,
            image_resolution=args.image_resolution,
            tokenizer=model.tokenizer,
            load_image=False,
            proc_id=args.proc_id,
            n_proc=args.n_proc,
        )

        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=dataset.text_collate_fn,
            num_workers=4,
        )

        image_dump_dir = Path('../IS_FID/COCO30K/minDALLE_zero')
        # image_dump_dir.mkdir(exist_ok=True, parents=True)
        assert image_dump_dir.exists(), image_dump_dir
        print('Image dump at: ', image_dump_dir)

        output_dir = image_dump_dir

        desc = f'Mindalle COCO 30K for IS/FID: {args.proc_id} out of {args.n_proc} procs'

        for batch in tqdm(loader, desc=desc):
            text_tokens = batch['input_ids']
            text_tokens = text_tokens.to(device)

            B = len(text_tokens)

            images = model.sampling(prompt="",
                                    tokens=text_tokens,
                                    top_k=args.top_k,
                                    top_p=args.top_p,
                                    softmax_temperature=args.softmax_temperature,
                                    # num_candidates=args.num_candidates,
                                    num_candidates=B,
                                    device=device,
                                    is_tqdm=False,
                                    ).cpu().numpy()
            images = np.transpose(images, (0, 2, 3, 1))

            filenames = batch['filenames']

            for i in range(len(images)):
                im = Image.fromarray((images[i]*255).astype(np.uint8))
                out_fname = output_dir.joinpath(filenames[i])
                im.save(out_fname)

    elif args.data == 'social':
        dataset = SocialDataset(
            tokenizer=model.tokenizer,
        )

        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=dataset.text_collate_fn,
            num_workers=4,
        )

        # image_dump_dir = Path('../social_bias_analysis/mindalle_output')
        image_dump_dir = Path('../social_bias_analysis/mindalle_output_person')
        # image_dump_dir.mkdir(exist_ok=True, parents=True)
        assert image_dump_dir.exists(), image_dump_dir
        print('Image dump at: ', image_dump_dir)

        output_dir = image_dump_dir

        for j in range(9):
            desc = f'Mindalle Social: {args.proc_id} out of {args.n_proc} procs - repeat{j}'

            for batch in tqdm(loader, desc=desc):
                text_tokens = batch['input_ids']
                text_tokens = text_tokens.to(device)

                B = len(text_tokens)

                images = model.sampling(prompt="",
                                        tokens=text_tokens,
                                        top_k=args.top_k,
                                        top_p=args.top_p,
                                        softmax_temperature=args.softmax_temperature,
                                        # num_candidates=args.num_candidates,
                                        num_candidates=B,
                                        device=device,
                                        is_tqdm=False,
                                        ).cpu().numpy()
                images = np.transpose(images, (0, 2, 3, 1))

                # filenames = batch['filenames']

                for i in range(len(images)):
                    im = Image.fromarray((images[i]*255).astype(np.uint8))

                    caption = batch['caption'][i]

                    caption_dir = image_dump_dir.joinpath(caption)

                    if not  caption_dir.is_dir():
                        caption_dir.mkdir()

                    fname = f"{j}.jpg"

                    out_fname = caption_dir.joinpath(fname)
                    im.save(out_fname)