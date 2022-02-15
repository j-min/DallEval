# ruDALL-E XL

## Based on https://github.com/sberbank-ai/ru-dalle

## Setup

```bash
pip install rudalle==0.0.1rc6
```


## Training on PaintSkills
```bash
cd rudalle/
python -m torch.distributed.launch \
    --nproc_per_node=$n_gpus \
    finetune_skills.py \
    --distributed \
    --clip 0.05 \
    --bs 1 \
    --skill $skill_name \
    --dataset_dir $dataset_dir \
    --save_path "checkpoints/"$skill_name

e.g.,
skill_name='object'
paintskills_dir='../../../../../datasets/PaintSkills'
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    finetune_skills.py \
    --distributed \
    --clip 0.05 \
    --bs 1 \
    --skill_name $skill_name \
    --dataset_dir $paintskills_dir \
    --save_path "checkpoints/"$skill_name
```

## Inference on PaintSkills
```bash
cd rudalle/
python inference.py \
    --data 'paintskills' \
    --skill $skill \
    --ckpt_path $ckpt_path \
    --dataset_dir $paintskills_dir
    --image_dump_dir $image_dump_dir


e.g.,
paintskills_dir='../../../../../datasets/PaintSkills'
image_dump_dir='../../../../../datasets/PaintSkills/rudalle_inference'
python inference.py \
    --data 'paintskills' \
    --skill 'spatial' \
    --ckpt_path "checkpoints/spatial/rudalle_epoch4.pt"
    --dataset_dir $paintskills_dir
    --image_dump_dir $image_dump_dir
```