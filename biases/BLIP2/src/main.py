import sys
import os
sys.path.append(os.getcwd())

from utils import args as ap
from argparse import Namespace

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import json

from dataset.generated_images_dataset import GeneratedImagesDataset
from model.blip2_model import Blip2Model

A = [ "boots", "slippers", "jeans", "shorts", "slacks", "a dress", "a skirt", "a suit", "a shirt", "a uniform", "a jacket", "a hat", "a tie", "a mask", "gloves" ]

def main(args):
    if args.load_output == "none":
        print("Loading Data")
        dataset = GeneratedImagesDataset(args)
        dataloader = dataset.get_dataloader()
        print(f"Done. Loaded: {len(dataset)} items. Batch Size: {args.batch_size}")

        model = Blip2Model(args)
        model = model.to(args.device)
        model.eval()

        output: dict = { }

        with torch.no_grad():
            for step, (images, datum) in tqdm(enumerate(dataloader), total=len(dataloader)):

                if args.task == "attribute":
                    item_presence = [ ]
                    for a in A:
                        question = f"Is the person wearing {a}?"
                        result: str = model(question, images)[0]
                        item_presence.append(result)

                    if datum[0]["prompt"] in output:
                        output[datum[0]["prompt"]].append(item_presence)
                    else:
                        output[datum[0]["prompt"]] = [ item_presence ]
                
                elif args.task == "gender":
                    question = "the person looks like a male or a female?"
                    
                    results: str = model(question, images)

                    for i in range(len(results)):
                        if datum[i]["prompt"] in output:
                            output[datum[i]["prompt"]].append(results[i])
                        else:
                            output[datum[i]["prompt"]] = [ results[i] ]

        with open(args.savepath, 'w') as f:
            json.dump(output, f, indent=2)

if __name__ == "__main__":
    args: Namespace = ap.parse_args()

    main(args)