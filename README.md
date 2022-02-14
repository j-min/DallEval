# DALL-Eval: Probing the Reasoning Skills and Social Biases of Text-to-Image Generative Transformers


* Authors: [Jaemin Cho](https://j-min.io), [Abhay Zala](https://aszala.com/), and [Mohit Bansal](https://www.cs.unc.edu/~mbansal/) (UNC Chapel Hill)
* [Paper](https://arxiv.org/abs/2202.04053)

<img src="./assets/teaser.png" alt="teaser image" width="700"/>

# Visual Reasoning

Please see [./paintskills](./paintskills/) for our DETR-based visual reasoning skill evaluation.


<!-- # Image-Text Alignment

TBD.

# Image Quality

TBD.

# Social Bias

TBD. -->

# Models

We provide training and inference scripts for [DALLE-small](./models/dalle_small/), [X-LXMERT](./models/xlxmert/), [MinDALL-E](models/mindalle), and [RuDALL-E XL](./models/rudalle/).


<!-- # Acknowledgments
- We thank the developers of [DETR](https://github.com/facebookresearch/detr) for their public code release. -->

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