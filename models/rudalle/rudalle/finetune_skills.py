#@title Доступные ресурсы
import argparse
from pynvml import nvmlInit, nvmlDeviceGetCount, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
import multiprocessing
import torch
from psutil import virtual_memory

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
# from torch.utils.data import Dataset
from tqdm import tqdm
from dataclasses import dataclass, field
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader

#@markdown import model
from rudalle.pipelines import generate_images, show, super_resolution, cherry_pick_by_clip
from rudalle import get_rudalle_model, get_tokenizer, get_vae, get_realesrgan, get_ruclip
from rudalle.utils import seed_everything

from rudalle_dataset import SkillTextImageDataset

from pathlib import Path
from random import randint, choice

import PIL

from torch.utils.data import Dataset
from torchvision import transforms as T

import json
from copy import deepcopy
import numpy as np
import torch

import wandb

import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast

import argparse

def get_parser():
    parser = argparse.ArgumentParser()
    # self.epochs = 1
    # self.save_path = 'skill_checkpoints/'
    # self.model_name = 'rudalle_skill_'
    # self.save_every = 2000
    # self.prefix_length = 10
    # self.bs = 1
    # self.clip = 0.24
    # self.lr = 4e-5
    # self.warmup_steps = 50
    # self.wandb = False
    # self.image_size = 256
    # self.resize_ratio = 0.75
    # self.is_shuffle = True

    parser.add_argument('--project_name', type=str, default='rudalle_skill_finetune')

    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--save_path', type=str, default='checkpoints/')
    parser.add_argument('--model_name', type=str, default='rudalle')
    parser.add_argument('--save_every', type=int, default=10000)
    parser.add_argument('--prefix_length', type=int, default=10)
    parser.add_argument('--bs', type=int, default=1)
    parser.add_argument('--clip', type=float, default=0.24)
    parser.add_argument('--lr', type=float, default=4e-5)
    parser.add_argument('--warmup_steps', type=int, default=50)
    parser.add_argument('--wandb', type=bool, default=True)
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--resize_ratio', type=float, default=0.75)
    parser.add_argument('--is_shuffle', type=bool, default=True)
    parser.add_argument('--dataset_dir', type=str, default='../../../datasets/PaintSkills/')
    parser.add_argument('--skill_name', type=str, default='object')

    parser.add_argument('--num_workers', type=int, default=4)

    parser.add_argument('--split', type=str, default='train')

    parser.add_argument('--distributed', action='store_true')
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--gpu', type=int, default=None)

    return parser

def freeze(
        model,
        freeze_emb=True,
        freeze_ln=False,
        freeze_attn=False,
        freeze_ff=True,
        freeze_other=True,
    ):
        for name, p in model.module.named_parameters():
            name = name.lower()
            if 'ln' in name or 'norm' in name:
                p.requires_grad = not freeze_ln
            elif 'embeddings' in name:
                p.requires_grad = not freeze_emb
            elif 'mlp' in name:
                p.requires_grad = not freeze_ff
            # elif 'attn' in name:
            elif 'attention' in name:
                p.requires_grad = not freeze_attn
            else:
                p.requires_grad = not freeze_other
        return model

#markdown Simple training loop
def train(args, model, vae, train_dataloader, optimizer, scheduler):
    """
    args - arguments for training

    train_dataloader - RuDalleDataset class with text - image pair in batch
    """
    loss_logs = []

    is_main_process = args.gpu in [0, -1]

    try:
        # if is_main_process:
        #     # progress = tqdm(total=len(train_dataloader), desc='finetuning goes brrr')
        #     progress = tqdm(total=len(train_dataloader),
        #                     desc=f'{args.skill_name}')
        save_counter = 0
        for epoch in range(args.epochs):

            if is_main_process:
                # progress = tqdm(total=len(train_dataloader), desc='finetuning goes brrr')
                progress = tqdm(total=len(train_dataloader),
                                desc=f'{args.skill_name} Epoch {epoch}')

            if args.distributed:
                train_dataloader.sampler.set_epoch(epoch)

            model.train();

            for text, images in train_dataloader:

                # print('loaded batch')

                device = f'cuda:{args.gpu}' if args.gpu is not None else 'cpu'

                # print(device)
                text = text.to(device)
                images = images.to(device)

                # print('batch to device')

                # device = model.get_param('device')
                save_counter+=1
                model.zero_grad()
                attention_mask = torch.tril(torch.ones((args.bs, 1, args.total_seq_length, args.total_seq_length), device=device))
                image_input_ids = vae.get_codebook_indices(images)

                # print('after vae')

                input_ids = torch.cat((text, image_input_ids), dim=1)
                if args.distributed:
                    loss, loss_values = model.module.forward(input_ids, attention_mask, return_loss=True)
                else:
                    loss, loss_values = model.forward(input_ids, attention_mask, return_loss=True)
                #train step
                loss.backward()

                # print('backprop')

                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                # print('after step')
                #save every here
                if is_main_process and save_counter % args.save_every == 0:
                    # print(f'Saveing checkpoint here {args.model_name}_dalle_{save_counter}.pt')
                    # print(f'Saveing checkpoint here {args.model_name}_{save_counter}.pt')

                    plt.plot(loss_logs)
                    plt.show()
                    # torch.save(
                    #         model.state_dict(),
                    #         # os.path.join(args.save_path,f"{args.model_name}_dalle_{save_counter}.pt")
                    #         os.path.join(args.save_path,f"{args.model_name}_{save_counter}.pt")
                    #         )
                if is_main_process and args.wandb:
                    wandb.log({"loss":  loss.item()})
                loss_logs += [loss.item()]

                if is_main_process:
                    progress.update()
                    progress.set_postfix({"loss": loss.item()})

            print (f'Finished epoch {epoch}')
            print('Saving checkpoint', f"{args.model_name}_epoch{epoch}.pt")
            torch.save(
                model.state_dict(),
                os.path.join(args.save_path,
                             f"{args.model_name}_epoch{epoch}.pt")
            )

        if is_main_process:
            # print(f'Complitly tuned and saved here  {args.model_name}__dalle_last.pt')
            print(f'Complitly tuned and saved here  {args.model_name}_last.pt')

            # plt.plot(loss_logs)
            # plt.show()

            torch.save(
                        model.state_dict(),
                        # os.path.join(args.save_path,f"{args.checkpoints}/{args.model_name}_dalle_last.pt")
                        os.path.join(args.save_path, f"{args.model_name}_last.pt")
                        )

            # with torch.no_grad():
            #     model.eval()

            #     pil_images = []


            # scores = []
            # text = 'Кроссовки, Nike, цвет: черный'

            # for top_k, top_p, images_num in [(2048, 0.995, 3)]:
            #     _pil_images, _scores = generate_images(
            #         text, tokenizer, model, vae, top_k=top_k, images_num=images_num, top_p=top_p)
            #     pil_images += _pil_images
            # show([pil_image for pil_image in pil_images], 3)

            # model.train()


    except KeyboardInterrupt:
        # if is_main_process:
        #     print(f'What for did you stopped? Please change model_path to /{args.checkpoints}/{args.model_name}_dalle_Failed_train.pt')
        #     plt.plot(loss_logs)
        #     plt.show()

        #     torch.save(
        #                 model.state_dict(),
        #                 os.path.join(args.save_path,f"{args.checkpoints}/{args.model_name}_dalle_Failed_train.pt")
        #                 )

        pass
    except Exception as err:
        print(f'Failed with {err}')



def main_worker(gpu, args):
    # GPU is assigned
    args.gpu = gpu
    args.rank = gpu
    print(f'Process Launching at GPU {gpu}')

    is_main_process = gpu in [0, -1]

    if args.distributed:
        torch.cuda.set_device(args.gpu)
        dist.init_process_group(backend='nccl')

    print(f'Building train loader at GPU {gpu}')
    # train_loader = get_loader(
    #     args,
    #     split=args.train, mode='train', batch_size=args.batch_size,
    #     distributed=args.distributed, gpu=args.gpu,
    #     workers=args.num_workers,
    #     topk=args.train_topk,
    # )

    device = f'cuda:{gpu}'
    model = get_rudalle_model('Malevich', pretrained=True, fp16=True, device=device)
    vae = get_vae().to(device)
    tokenizer = get_tokenizer()

    args.text_seq_length = model.get_param('text_seq_length')
    args.total_seq_length = model.get_param('total_seq_length')

    # paintskill_dir = Path(args.data'../../../datasets/PaintSkills/').resolve()
    dataset_dir = Path(args.dataset_dir).resolve()

    # st = RuDalleDataset(file_path='content/sneaks/', csv_path='data_desc.csv',
    # tokenizer=tokenizer,
    # text_seq_length=model.get_param('text_seq_length'),
    # device=device
    # )

    image_dir = str(dataset_dir / args.skill_name / 'images')
    # text_data_file = str(dataset_dir / args.skill_name / 'scenes' / args.text_file)
    text_data_file = str(dataset_dir / args.skill_name / 'scenes' / f"{args.skill_name}_{args.split}_ru.json")

    text_seq_length = model.get_param('text_seq_length')

    ds = SkillTextImageDataset(
        args=args,
        skill_name=args.skill_name,
        split=args.split,
        image_dir=image_dir,
        text_data_file=text_data_file,
        text_len=text_seq_length,
        image_size=args.image_size,
        resize_ratio=args.resize_ratio,
        truncate_captions=True,
        tokenizer=tokenizer,
        shuffle=args.is_shuffle,
        # paintskill_real=False
    )

    sampler = torch.utils.data.distributed.DistributedSampler(ds)

    train_dataloader = DataLoader(
        ds,
        batch_size=args.bs,
        drop_last=True,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True
        )

    from transformers import AdamW, get_linear_schedule_with_warmup
    model.train()
    optimizer = AdamW(model.parameters(), lr=args.lr)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr,
                                                    final_div_factor=500,
                                                    steps_per_epoch=len(train_dataloader), epochs=args.epochs)

    #@markdown You can unfreeze or freeze more parametrs, but it can
    model = freeze(model=model,
                   freeze_emb=False,
                   freeze_ln=False,
                   freeze_attn=True,
                   freeze_ff=True,
                   freeze_other=False)  # freeze params to

    model = DDP(model, device_ids=[args.gpu],
        find_unused_parameters=True
    )

    if is_main_process:
        project_name = args.project_name

        wandb.init(project=project_name)
        wandb.run.name = args.run_name
        wandb.config.update(args)
        wandb.watch(model)

        src_dir = Path(__file__).resolve().parent
        base_path = str(src_dir.parent)
        src_dir = str(src_dir)
        wandb.save(os.path.join(src_dir + "/*.py"), base_path=base_path)

    train(args, model, vae, train_dataloader, optimizer, scheduler)



if __name__ == '__main__':
    ram_gb = round(virtual_memory().total / 1024**3, 1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    nvmlInit()
    h = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(h)
    # if info.total > 16252636672:
    #     print('Everything is ok, you can begin')
    # else:
    #     print('We dont recomend to begin, you gonna get Out of memory')

    cudnn.benchmark = True
    parser = get_parser()
    args = parser.parse_args()

    ngpus_per_node = torch.cuda.device_count()
    args.world_size = ngpus_per_node
    if args.local_rank in [0, -1]:
        print('CPU:', multiprocessing.cpu_count())
        print('RAM GB:', ram_gb)
        print("PyTorch version:", torch.__version__)
        print("CUDA version:", torch.version.cuda)
        print("cuDNN version:", torch.backends.cudnn.version())
        print("device:", device.type)

        if info.total > 16252636672:
            print('Everything is ok, you can begin')
        else:
            print('We dont recomend to begin, you gonna get Out of memory')

        print(args)

        comments = []
        # if args.load is not None:
        #     ckpt_str = "_".join(args.load.split('/')[-3:])
        #     comments.append(ckpt_str)
        # if args.comment != '':
        #     comments.append(args.comment)
        comment = '_'.join(comments)

        from datetime import datetime
        current_time = datetime.now().strftime('%b%d_%H-%M')

        run_name = f'{current_time}_GPU{args.world_size}'
        if len(comments) > 0:
            run_name += f'_{comment}'
        run_name += f"_{args.skill_name}"

        args.run_name = run_name

    if args.distributed:
        main_worker(args.local_rank, args)
