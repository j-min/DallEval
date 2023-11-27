import torch
from torch.utils.data import Dataset, DataLoader
from argparse import Namespace
from tqdm import tqdm
import json
from PIL import Image
import numpy as np
import csv

class GeneratedImagesDataset(Dataset):
    
    def __init__(self, args: Namespace):
        self.args: Namespace = args
        
        self.image_path: str = self.args.image_dir
        metadata_file: str = self.args.metadata_file

        self.data = [ ]

        with open(metadata_file, 'r') as f:
            data = json.load(f)
            for prompt in data:

                if args.task == "attribute":
                    for i in range(9):
                        d = {
                            "image_path": f"{self.image_path}/{prompt.replace(' ', '_')}/{i}.png",
                            "idx": i,
                            "prompt": prompt
                        }
                        self.data.append(d)
                elif args.task == "gender":
                    if "A person " in prompt or prompt == "A person":
                        for i in range(9):
                            d = {
                                "image_path": f"{self.image_path}/{prompt.replace(' ', '_')}/{i}.png",
                                "idx": i,
                                "prompt": prompt
                            }
                            self.data.append(d)
    
    def __getitem__(self, index) -> tuple:
        datum: dict = self.data[index]

        image: Image = Image.open(datum["image_path"]).convert("RGB")

        return image, datum

    def __len__(self) -> int:
        return len(self.data)

    def get_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self, batch_size=self.args.batch_size, shuffle=False,
                            num_workers=self.args.num_workers, collate_fn=GeneratedImagesDataset.collate_fn)

    def collate_fn(batch):
        images: list = [ ]
        datums: list[dict] = [ ]

        for (image, datum) in batch:
            images.append(image)
            datums.append(datum)

        return images, datums