import torch
import torch.nn as nn
from torch import Tensor
from argparse import Namespace
from lavis.models import load_model_and_preprocess

class Blip2Model(nn.Module):
    def __init__(self, args: Namespace):
        super(Blip2Model, self).__init__()

        self.args = args

        self.model, self.vis_processors, _ = load_model_and_preprocess(name="blip2_t5",
                                                            model_type="pretrain_flant5xxl",
                                                            is_eval=True, device=args.device)
        
    def forward(self, q: str, image_input) -> list:
        question = f"Question: {q} Answer:"

        images = [ self.vis_processors["eval"](im) for im in image_input ]

        images = torch.stack(images).to(self.args.device)
        answer = self.model.generate({"image": images, "prompt": [question] * len(images)})
        
        return [ a.lower() for a in answer ]
