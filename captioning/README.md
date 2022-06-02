# Image-text Alignment Evaluation - Captioning

We provide captioning-based evaluation with VL-T5.


## Setup

1) Download `karpathy_test_text.json` from [google drive](https://drive.google.com/drive/folders/1eNeXzzhB7q9XCD_CNvNI0QZJINlUsE04).


```bash
# karpathy_test_text.json
gdown 1nxbCbRA0c7pPGJbT8tCfqKOxMgrQ6hsA
```

The `karpathy_test_text.json` file contains 5000 items of image id - caption pair (a caption is sampled from one of 5 reference captions) that correspond to Karpathy test split of COCO. Below is the first few lines of the file.

```json
[
    {
        "img_id": "COCO_val2014_000000391895",
        "targets": "A man with a red helmet on a small moped on a dirt road."
    },
    {
        "img_id": "COCO_val2014_000000060623",
        "targets": "A young girl inhales with the intent of blowing out a candle."
    },
    {
        "img_id": "COCO_val2014_000000483108",
        "targets": "A man on a bicycle riding next to a train"
...
```

2) Generate images from each caption, and save it in `$image_dir`.

```bash
./image_dir/
    COCO_val2014_000000391895.jpg # Generated from "A man with a red helmet on a small moped on a dirt road."
    COCO_val2014_000000060623.jpg # Generated from "A young girl inhales with the intent of blowing out a candle."
    ...
```

3) Setup VL-T5 captioning model

```bash
git clone https://github.com/j-min/VL-T5
cd VL-T5
pip install -r requirements.txt
pip install opencv-python
python -c "import language_evaluation; language_evaluation.download('coco')"
```

### Difference in visual feature between VL-T5 and this VL-T5 implementation
The FRCNN used in this repo is adapted from [Hugginface LXMERT demo](https://github.com/huggingface/transformers/tree/main/examples/research_projects/lxmert).
While this Hugginface FRCNN implementation is easy to work with custom images, we found that the Huggingface FRCNN provides slightly different features from the FRCNN features used in LXMERT and VL-T5.
Therefore, we finetune VL-T5 with this new FRCNN and provide this checkpoint for consistency. We used this checkpoint for our caption based evaluations.
The change of visual encoder made slight drop in the captioning performance for VL-T5 (e.g., BLUE@4: 34 -> 31 for Karpathy test split).

Download `dataset_coco.json` and `VLT5_HF_FRCNN_COCOCaption.pth` from [google drive](https://drive.google.com/drive/folders/1eNeXzzhB7q9XCD_CNvNI0QZJINlUsE04).


```bash
# dataset_coco.json
gdown 1dGVf6dCpddpvHT85TWnHiHOaQ9p_6Xuq

# VLT5_HF_FRCNN_COCOCaption.pth
gdown 1jDi6spmY892eO2AWvvzESixX-YZXISiz
```

4) Run evaluation script - It takes around 10 mins on single RTX 2080 Ti GPU

```bash
python run_caption_evaluation.py

...
Eval results # based on GT COCO Images
{'Bleu_1': 0.7237392516848571,
 'Bleu_2': 0.5633070870571786,
 'Bleu_3': 0.42804852307296365
 'Bleu_4': 0.325427941604873,
 'CIDEr': 1.0826244658301367,
 'METEOR': 0.2748594629606107,
 'ROUGE_L': 0.5527750831742165
 'SPICE': 0.20432732656681163}
...
```
<small> *In [the arxiv v1 pdf](https://arxiv.org/pdf/2202.04053v1.pdf) Table 7, there is a typo of GT (Up. bound) row SPICE = 55.3 which actually is the ROUGE-L value. We will update this in the later version.
</small>