

skill_name=$1
# skill_name='object'
dataset_dir=$2
# dataset_dir='../../../../../datasets/PaintSkills'

split='train'
dalle_path='dalle_CC.pt'

deepspeed finetune_dalle.py \
    --dalle_path $dalle_path \
    --taming \
    --deepspeed \
    --batch_size 40 \
    --epochs 10 \
    --text_seq_len 128 \
    --heads 8 \
    --dim_head 64 \
    --reversible \
    --loss_img_weight 7 \
    --attn_types 'sparse' \
    --dataset_dir $dataset_dir \
    --skill_name $skill_name \
    --split $split \
    --text_file $skill_name'_'$split'.json' \
    --save_every_n_steps -100 \
    --dalle_output_file_name "DALLE_CC_"$skill_name \
    --wandb_name "PaintSkills-"$skill_name \
    --wandb_run_name "DALLE CC12M+3M Pretrained" \
    ${@:2}