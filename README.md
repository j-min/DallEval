# DALL-Eval: Probing the Reasoning Skills and Social Biases of Text-to-Image Generative Transformer



* Authors: [Jaemin Cho](https://j-min.io), [Abhay Zala](https://aszala.com/), and [Mohit Bansal](https://www.cs.unc.edu/~mbansal/)
* [Paper](https://arxiv.org/abs/2202.04053)


## PaintSkills - Visual Reasoning

### Dataset

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


### Evaluation

Please see [./paintskills/detr/README.md](./paintskills/detr/README.md) for our DETR-based visual reasoning skill evaluation.


* Evaluation scripts for other aspects will be updated.

#  Acknowledgements
- We thank the developers of [DETR](https://github.com/facebookresearch/detr) for public releases of their code.

# Reference
Please cite our paper if you use our models in your works:
```bibtex

@article{cho2022dalleval,
  title         = {DALL-Eval: Probing the Reasoning Skills and Social Biases of Text-to-Image Generative Transformer},
  author        = {Jaemin Cho and Abhay Zala and Mohit Bansal},
  year          = {2022},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  eprint        = {2202.04053}
}
```