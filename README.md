# DALL-Eval: Probing the Reasoning Skills and Social Biases of Text-to-Image Generative Transformers


* Authors: [Jaemin Cho](https://j-min.io), [Abhay Zala](https://aszala.com/), and [Mohit Bansal](https://www.cs.unc.edu/~mbansal/) (UNC Chapel Hill)
* [Paper](https://arxiv.org/abs/2202.04053)

<img src="./assets/teaser.png" alt="teaser image" width="700"/>


# PaintSkills - Visual Reasoning

## Dataset

* Download four skill data: `object.zip`, `count.zip`, `color.zip` and `spatial.zip` from [the Google Drive link](https://drive.google.com/drive/folders/1Bza2zyvHLvComohZ9PAGyykY7sm7JoIH) and unzip.
```bash
unzip object.zip
unzip count.zip
unzip color.zip
unzip spatial.zip
```


* Each skill directory has hierarchy as below:
```bash
{skill}/        # skill name (i.e.., object, count, color, and spatial)
    # Images
    images/

    # Scene configuration
    scenes/
        {skill}_train.json
        {skill}_val.json

    # Bounding box annotations - only needed for DETR
    {skill}_train_bounding_boxes.json
    {skill}_val_bounding_boxes.json
```


## Evaluation

Please see [./paintskills/detr/README.md](./paintskills/detr/README.md) for our DETR-based visual reasoning skill evaluation.


# Acknowledgements
- We thank the developers of [DETR](https://github.com/facebookresearch/detr) for their public code release.

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