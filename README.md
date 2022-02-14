# DALL-Eval: Probing the Reasoning Skills and Social Biases of Text-to-Image Generative Transformers


* Authors: [Jaemin Cho](https://j-min.io), [Abhay Zala](https://aszala.com/), and [Mohit Bansal](https://www.cs.unc.edu/~mbansal/) (UNC Chapel Hill)
* [Paper](https://arxiv.org/abs/2202.04053)

<img src="./assets/teaser.png" alt="teaser image" width="500"/>

# Visual Reasoning

<img src="./assets/skills.png" alt="skill image" width="1200"/>

Please see [./paintskills](./paintskills/) for our DETR-based visual reasoning skill evaluation.

# Image Quality & Image-Text Alignment

<img src="./assets/alignment_quality.png" alt="alignment and quality image" width="500"/>

TBD.

# Social Bias

<img src="./assets/bias_experiment_flow.png" alt="bias exp image" width="500"/>

TBD.

# Models

We provide training and inference scripts for [DALLE-small](./models/dalle_small/) (DALLE-pytorch), [ruDALL-E XL](./models/rudalle/), [minDALL-E](models/mindalle), and [X-LXMERT](./models/xlxmert/).


# Acknowledgments
We thank the developers of [DETR](https://github.com/facebookresearch/detr), [DALLE-pytorch](https://github.com/lucidrains/DALLE-pytorch), [ruDALL-E](https://github.com/sberbank-ai/ru-dalle), [minDALL-E](https://github.com/kakaobrain/minDALL-E), and [X-LXMERT](https://github.com/allenai/x-lxmert), for their public code release.

# Reference
Please cite our paper if you use our dataset in your works:
```bibtex

@article{Cho2022DallEval,
  title         = {DALL-Eval: Probing the Reasoning Skills and Social Biases of Text-to-Image Generative Transformers},
  author        = {Jaemin Cho and Abhay Zala and Mohit Bansal},
  year          = {2022},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  eprint        = {2202.04053}
}
```