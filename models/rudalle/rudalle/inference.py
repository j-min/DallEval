import argparse
import multiprocessing
import torch
from psutil import virtual_memory

import json
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

import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm
from dataclasses import dataclass, field
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from rudalle.pipelines import generate_images, show, super_resolution, cherry_pick_by_clip
from rudalle import get_rudalle_model, get_tokenizer, get_vae, get_realesrgan, get_ruclip
from rudalle.utils import seed_everything, torch_tensors_to_pil_list

from pathlib import Path

from rudalle_dataset import SkillTextImageDataset

def batch_generate_images(text, tokenizer, dalle, vae, top_k, top_p, images_num, image_prompts=None, temperature=1.0, bs=8,
                    seed=None, use_cache=True):
    # TODO docstring
    if seed is not None:
        seed_everything(seed)

    vocab_size = dalle.get_param('vocab_size')
    text_seq_length = dalle.get_param('text_seq_length')
    image_seq_length = dalle.get_param('image_seq_length')
    total_seq_length = dalle.get_param('total_seq_length')
    device = dalle.get_param('device')

    # text = text.lower().strip()
    # input_ids = tokenizer.encode_text(text, text_seq_length=text_seq_length)

    B = len(text)

    if type(text[0]) == str:
        text = [sent.lower().strip() for sent in text]
        input_ids = [tokenizer.encode_text(sent, text_seq_length=text_seq_length) for sent in text]
        input_ids = torch.stack(input_ids, 0).view(B, text_seq_length)
    else:
        input_ids = text

    pil_images, scores = [], []
    with torch.no_grad():
        attention_mask = torch.tril(torch.ones((B, 1, total_seq_length, total_seq_length), device=device))
        # out = input_ids.unsqueeze(0).repeat(chunk_bs, 1).to(device)

        # out = input_ids[start_i:end_i].to(device)
        out = input_ids.to(device)

        has_cache = False
        sample_scores = []
        # if image_prompts is not None:
        #     prompts_idx, prompts = image_prompts.image_prompts_idx, image_prompts.image_prompts
        #     prompts = prompts.repeat(B, 1)

        # for idx in tqdm(range(out.shape[1], total_seq_length)):
        for idx in range(out.shape[1], total_seq_length):
            idx -= text_seq_length
            # if image_prompts is not None and idx in prompts_idx:
            #     out = torch.cat((out, prompts[:, idx].unsqueeze(1)), dim=-1)
            # else:

            logits, has_cache = dalle(out, attention_mask,
                                        has_cache=has_cache, use_cache=use_cache, return_loss=False)
            logits = logits[:, -1, vocab_size:]
            logits /= temperature
            filtered_logits = transformers.top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
            probs = torch.nn.functional.softmax(filtered_logits, dim=-1)
            sample = torch.multinomial(probs, 1)
            sample_scores.append(probs[torch.arange(probs.size(0)), sample.transpose(0, 1)])
            out = torch.cat((out, sample), dim=-1)

        codebooks = out[:, -image_seq_length:]
        images = vae.decode(codebooks)
        pil_images += torch_tensors_to_pil_list(images)
        scores += torch.cat(sample_scores).sum(0).detach().cpu().numpy().tolist()

            # start_i += chunk_bs
            # end_i += chunk_bs

    return pil_images, scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--skill', type=str, default=None)
    parser.add_argument('--data', type=str, choices=['paintskills', 'COCO', 'COCO30K'], default='paintskills')

    parser.add_argument('--dataset_dir', type=str, default='.../datasets/PaintSkills/')
    parser.add_argument('--image_dump_dir', type=str, default='.../datasets/PaintSkills/rudalle_inference/')

    parser.add_argument('--split', type=str, default='karpathy_test')

    parser.add_argument('--batch_size', type=int, default=32)

    parser.add_argument('--proc_id', type=int, default=-1)
    parser.add_argument('--n_proc', type=int, default=-1)

    parser.add_argument('--ckpt_path', type=str, default=None)

    args = parser.parse_args()
    print(args)

    ram_gb = round(virtual_memory().total / 1024**3, 1)

    print('CPU:', multiprocessing.cpu_count())
    print('RAM GB:', ram_gb)
    print("PyTorch version:", torch.__version__)
    print("CUDA version:", torch.version.cuda)
    print("cuDNN version:", torch.backends.cudnn.version())
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print("device:", device.type)

    # class Args():

    #     def __init__(self):

    #         self.text_seq_length = model.get_param('text_seq_length')
    #         self.total_seq_length = model.get_param('total_seq_length')
    #         self.epochs = 1
    #         # self.save_path = 'checkpoints/'
    #         # self.model_name = 'awesomemodel_'
    #         # self.save_every = 2000
    #         # self.prefix_length = 10
    #         self.bs = 1
    #         self.clip = 0.24
    #         self.lr = 4e-5
    #         self.warmup_steps = 50
    #         self.wandb = True
    # args = Args()

    # if not os.path.exists(args.save_path):
    #     os.makedirs(args.save_path)

    device = 'cuda'
    print('Model loading...')
    model = get_rudalle_model(
        'Malevich', pretrained=True, fp16=True, device=device)
    vae = get_vae().to(device)
    tokenizer = get_tokenizer()
    model = model.eval()
    print('Model loaded')

    if args.ckpt_path is not None:
        # skill_checkpoint_path = './skill_checkpoints_backup/rudalle_skill_dalle_30000.pt'

        def load_state_dict(state_dict_path, loc='cpu'):
            state_dict = torch.load(state_dict_path, map_location=loc)
            # Change Multi GPU to single GPU
            original_keys = list(state_dict.keys())
            for key in original_keys:
                if key.startswith("module."):
                    new_key = key[len("module."):]
                    state_dict[new_key] = state_dict.pop(key)
            return state_dict

        ckpt_dict =  load_state_dict(args.ckpt_path, loc=device)
        # len(ckpt_dict)
        result = model.load_state_dict(ckpt_dict, strict=False)
        print(result)


    top_k, top_p, images_num = 2048, 0.995, 1

    if args.data == 'paintskills':
        if args.skill is None:
            skills = ['object', 'color', 'count', 'spatial']
        else:
            skills = [args.skill]
        # splits = ['val', 'train']

        # paintskill_dir = Path('../../../datasets/PaintSkills/').resolve()

        paintskill_dir = Path(args.dataset_dir).resolve()
        rudalle_inference_dir = paintskill_dir(args.imamge_dump_dir).resolve()

        split = 'val'
        for skill in skills:

            skill_dir = paintskill_dir.joinpath(f'{skill}')
            ru_scene_path = skill_dir .joinpath(f'scenes/{skill}_{split}_ru.json')

            with open(ru_scene_path) as f:
                scene_data = json.load(f)

            if args.ckpt_path is None:
                output_dir = rudalle_inference_dir.joinpath(f'{skill}_zero_{split}')
            else:
                output_dir = rudalle_inference_dir.joinpath(f'{skill}_{split}')
            output_dir.mkdir(exist_ok=True, parents=True)
            print('Image output dir:', output_dir)

            text_data_file = str(paintskill_dir / skill / 'scenes' / f"{skill}_{split}_ru.json")

            text_seq_length = model.get_param('text_seq_length')

            ds = SkillTextImageDataset(
                args=args,
                skill_name=skill,
                split=split,
                image_dir=None,
                text_data_file=text_data_file,
                text_len=text_seq_length,
                truncate_captions=True,
                tokenizer=tokenizer,
                shuffle=False,
                load_image=False,
            )

            from torch.utils.data import Dataset, DataLoader

            dataloader = DataLoader(
                ds,
                batch_size=args.batch_size,
                drop_last=False,
                num_workers=4,
                pin_memory=True,
                collate_fn=ds.text_collate_fn
            )

            for batch in tqdm(dataloader, desc=f'Inference: {skill}-{split}'):
                input_ids = batch['input_ids']

                pil_images, scores = batch_generate_images(
                    # text=ru_text_list,
                    text=input_ids,
                    tokenizer=tokenizer,
                    dalle=model,
                    vae=vae,
                    top_k=top_k,
                    images_num=images_num,
                    top_p=top_p,
                )

                for j, img in enumerate(pil_images):
                    filename = batch['id'][j]
                    out_fname = output_dir.joinpath(f'{filename}.png')
                    img.save(output_dir.joinpath(out_fname))


    elif args.data == 'COCO':
        coco_dir = Path('../../../datasets/COCO/').resolve()

        # ann_path = coco_dir.joinpath('dataset_coco.json')
        ann_path = coco_dir.joinpath('dataset_coco_karpathy_ru.json')
        print('Loading text from', ann_path)
        cap_ann_data = json.load(open(ann_path))

        image_dump_dir = coco_dir.joinpath('rudalle_inference').joinpath(args.split)
        image_dump_dir.mkdir(exist_ok=True, parents=True)
        print('Image dump at: ', image_dump_dir)

        split_rename = {
            'train': 'train',
            'restval': 'train',
            'val': 'val',
            'test': 'test'
        }

        karpathy_split_name = args.split.split('_')[-1]

        data = [datum for datum in cap_ann_data['images'] if split_rename[datum['split']] == karpathy_split_name]

        if args.n_proc:
            sampled_data = data[args.proc_id::args.n_proc]
        else:
            sampled_data = data

        # for datum in tqdm(split_data):

        for batch_datum in more_itertools.chunked(
            tqdm(
                sampled_data,
                desc=f'Rudalle imagen {ann_path}: {args.proc_id} out of {args.n_proc} procs',
                # disable=args.proc_id>0
                ),
            args.batch_size):

            ru_text_list = []
            filename_list = []
            for datum in batch_datum:
                ru_text_list.append(datum['ru_sentences'][0])
                filename_list.append(datum["filename"])

            pil_images, scores = batch_generate_images(
                text=ru_text_list,
                tokenizer=tokenizer,
                dalle=model,
                vae=vae,
                top_k=top_k,
                images_num=images_num,
                top_p=top_p,
            )


            for j, img in enumerate(pil_images):
                img.save(image_dump_dir.joinpath(filename_list[j]))


    elif args.data == 'COCO30K':

        uid_caption_path = Path('../IS_FID/uid_caption_ru.csv')
        assert uid_caption_path.is_file(), uid_caption_path
        print('Load from:', uid_caption_path)
        df = pd.read_csv(uid_caption_path)
        assert len(df) == 30000, len(df)
        captions = df['caption'].tolist()
        uids = df['uid'].tolist()

        if args.n_proc > 0:
            print('total ', len(captions), 'data')
            captions = captions[args.proc_id::args.n_proc]
            uids = uids[args.proc_id::args.n_proc]
            print('sampled ', len(captions), 'data')

        image_dump_dir = Path('../IS_FID/COCO30K/RUDALLE_zero')
        # image_dump_dir.mkdir(exist_ok=True, parents=True)
        assert image_dump_dir.exists(), image_dump_dir
        print('Image dump at: ', image_dump_dir)

        for batch_datum in more_itertools.chunked(
            tqdm(
                # cocodataset,
                zip(captions, uids),
                desc=f'Rudalle imagen - COCO 30K for IS/FID: {args.proc_id} out of {args.n_proc} procs',
                total=len(captions),
                # disable=args.proc_id>0
            ),
            args.batch_size):

            ru_text_list = []
            filename_list = []
            for ru_text, uid in batch_datum:
                ru_text_list.append(ru_text)

                filename = f'{uid}.jpg'
                filename_list.append(filename)

            pil_images, scores = batch_generate_images(
                text=ru_text_list,
                tokenizer=tokenizer,
                dalle=model,
                vae=vae,
                top_k=top_k,
                images_num=images_num,
                top_p=top_p,
            )

            for j, img in enumerate(pil_images):
                img.save(image_dump_dir.joinpath(filename_list[j]))
