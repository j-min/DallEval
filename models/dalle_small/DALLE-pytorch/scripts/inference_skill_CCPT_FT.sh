

skill_name=$1
dataset_dir=$2
# dataset_dir='../../../../../datasets/PaintSkills'
image_dump_dir$3
# image_dump_dir='../../../../../datasets/PaintSkills/DalleSmall_inference'

split='val'



dalle_path="DALLE_CC_"$skill_name".pt"

python inference_skill_dalle.py \
    --dalle_path $dalle_path \
    --batch_size 10 \
    --text_seq_len 128 \
    --truncate_captions \
    --dataset_dir $dataset_dir \
    --image_dump_dir $image_dump_dir \
    --skill_name $skill_name \
    --split $split \
    --text_file $skill_name'_'$split'.json'