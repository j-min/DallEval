# Image Quality Evaluation

We provide FID evaluation based on [DM-GAN repo](https://github.com/MinfengZhu/DM-GAN/tree/master/eval/FID).


## Setup
Download 30K COCO val 2014 images

```bash
python google_drive.py 1au9DI9tr-dcfGMFxFkFrJt0p_htojxH2 ./uid_caption.csv
```

The `uid_caption.csv` file consists of 30K images-caption pairs. The images are sampled from COCO val2014 split. For each image, a caption out of 5 paired captions is randomly sampled. The file has two keys:
* `uid`: `{COCO img id}_mscoco_{caption index}`
* `caption`: pared caption

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


## Calculate FID

1) Download pre-computed COCO statistics for FID [from DM-GAN repo](https://drive.google.com/file/d/10NYi4XU3_bLjPEAg5KQal-l8A_d8lnL5).
```bash
python google_drive.py 10NYi4XU3_bLjPEAg5KQal-l8A_d8lnL5 FID/coco_val.npz
```

1) Generate 30K images from the captions of `uid_caption.csv` in a directory `image_dir`. The images should be either `.jpg` or `.png` format.


2) Calculate FID
```bash
cd FID
python fid_score.py \
    --batch-size 100 \
    --gpu 0 \
    --path1 coco_val.npz
    --path2 image_dir
```