# minDALL-E

## Based on https://github.com/kakaobrain/minDALL-E

## Setup

```bash
# Download the CC checkpoint
mkdir cache
cd cache
wget https://arena.kakaocdn.net/brainrepo/models/minDALL-E/57b008f02ceaa02b779c8b7463143315/1.3B.tar.gz
tar -xvf 1.3B.tar.gz

# Install dependencies
cd minDALL-E/
pip install -r requirements.txt
```


## Training on PaintSkills
```bash
cd minDALL-E/
python examples/finetuning_mindalle_skill.py \
    --result_path './checkpoints/' \
    --skill_name $skill \
    --dataset_dir $paintskills_dir \
    --config-downstream './configs/dalle-skill.yaml' \
    --n_gpus $n_gpus

e.g.,
paintskills_dir='../../../../../datasets/PaintSkills'
python examples/finetuning_mindalle_skill.py \
    --result_path './checkpoints/' \
    --skill_name 'object' \
    --dataset_dir $paintskills_dir \
    --config-downstream './configs/dalle-skill.yaml' \
    --n_gpus 3
```

## Inference on PaintSkills
```bash
cd minDALL-E/
python examples/inference_mindalle.py \
    --ckpt_path $checkpoint_path \
    --data 'paintskills'
    --split 'val' \
    --skill_name $skill \
    --dataset_dir $paintskills_dir  \
    --image_dump_dir $image_dump_dir

e.g.,
python examples/inference_mindalle.py \
    --ckpt_path './checkpoints/object_04012022_193926/ckpt/last.ckpt' \
    --data 'paintskills'
    --split 'val' \
    --skill_name 'object' \
    --dataset_dir '../../../datasets/PaintSkills/' \
    --image_dump_dir '../../../datasets/PaintSkills/minDALLE_inference'
```