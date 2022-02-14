# DALLE-Small

## Modified from https://github.com/lucidrains/DALLE-pytorch

## Setup (modified from this [colab](https://colab.research.google.com/drive/1b8va5g852hq3p7yro7xWY3Cc-bd2CRdv) from [robvanvolt/DALLE-models](https://github.com/robvanvolt/DALLE-models))

```bash
sudo apt-get -y install llvm-9-dev cmake

pip install -r requirements.txt

# Deepspeed
git clone https://github.com/microsoft/DeepSpeed.git /tmp/Deepspeed
cd /tmp/Deepspeed
git checkout tags/v0.5.1
DS_BUILD_SPARSE_ATTN=1 ./install.sh -r

# DALLE-pytorch
cd DALLE-pytorch/
python3 setup.py install


# Download checkpoint trained on CC
wget https://github.com/johnpaulbin/DALLE-models/releases/download/model/16L_64HD_8H_512I_128T_cc12m_cc3m_3E.pt
mkdir dalle_CC
mv 16L_64HD_8H_512I_128T_cc12m_cc3m_3E.pt dalle_CC/auxiliary.pt

wget https://www.dropbox.com/s/15mhdhy57y6qttf/vqgan.1024.model.ckpt
wget https://www.dropbox.com/s/q8nayimg4skf0pl/vqgan.1024.config.yml

cp "vqgan.1024.model.ckpt" ~/.cache/dalle
cp "vqgan.1024.config.yml" ~/.cache/dalle
```

## Training on PaintSkills
```bash
bash scripts/finetune_skill_CCPT.sh $skill $paintskills_dir

e.g.,
paintskills_dir='../../../../../datasets/PaintSkills'
bash scripts/finetune_skill_CCPT.sh 'object' $paintskills_dir
bash scripts/finetune_skill_CCPT.sh 'count' $paintskills_dir
bash scripts/finetune_skill_CCPT.sh 'color' $paintskills_dir
bash scripts/finetune_skill_CCPT.sh 'spatial' $paintskills_dir
```

## Inference on PaintSkills
```bash
bash scripts/inference_skill_CCPT_FT.sh $skill $paintskills_dir $image_dump_dir

e.g.,
paintskills_dir='../../../../../datasets/PaintSkills'
image_dump_dir='../../../../../datasets/PaintSkills/DalleSmall_FT_object'
bash scripts/finetune_skill_CCPT.sh 'object' $paintskills_dir $image_dump_dir
```