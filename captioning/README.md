# Image-text Alignment Evaluation - Captioning

We provide captioning-based evaluation with VL-T5.


## Setup

1) Download `karpathy_test_text.json` from [google drive](https://drive.google.com/drive/folders/1eNeXzzhB7q9XCD_CNvNI0QZJINlUsE04).


```bash
# karpathy_test_text.jso
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

4) Run evaluation script
```bash
python run_caption_evaluation.py

...
Eval results # based on GT COCO Images
{
    'Bleu_1': 0.737227164947394,
    'Bleu_2': 0.5677760033468644,
    'Bleu_3': 0.42301410683524954,
    'Bleu_4': 0.3125204158572636,
    'METEOR': 0.2639331643036797,
    'ROUGE_L': 0.5439489553069495,
    'CIDEr': 1.0464655726928305,
    'SPICE': 0.19540839342910868
}
...
```
