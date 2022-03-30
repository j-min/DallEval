# Image-text Alignment Evaluation

We provide CLIP-based R-precision evaluation.


## Setup
Download 30K image-caption pairs.

```bash
python google_drive.py 1au9DI9tr-dcfGMFxFkFrJt0p_htojxH2 ./uid_caption.csv
```

The `uid_caption.csv` file consists of 30K image-caption pairs. The images are sampled from COCO val2014 split. For each image, a caption out of 5 paired captions is randomly sampled. The file has two keys:
* `uid`: `{COCO img id}_mscoco_{caption index}`
* `caption`: paired caption

Below are the first few lines of `uid_caption.csv`:
```
uid,caption
346904_mscoco_0,A bus driving down a road by a building.
416478_mscoco_2,A woman and two men looking at a laptop screen.
155051_mscoco_0,a close of up a clock that makes the moon look small next to it
135161_mscoco_3,A bathroom being renovated featuring a toilette and shower.
280036_mscoco_2,A boy in black sweater standing on beach flying a kite.
439969_mscoco_0,A rusted pink fire hydrant in the grass
...
```


## Calculate R-precision

1) Generate 30K images from the captions of `uid_caption.csv` in a directory `$image_dir`. The images should be either `.jpg` or `.png` format.

2) Download COCO val 2014 images from http://images.cocodataset.org/zips/val2014.zip at `$coco_image_dir`.

3) Calculate R-precision
```bash
python rprecision.py \
    --uid_caption_path uid_caption.csv \
    --image_dir $image_dir \
    --coco_image_dir $coco_image_dir
```