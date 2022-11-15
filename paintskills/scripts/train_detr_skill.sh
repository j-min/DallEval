
n_gpus=$1

python -m torch.distributed.launch \
    --nproc_per_node=$n_gpus \
    --use_env \
    --master_port=23456 \
    detr/main.py \
    --backbone 'resnet101' --dilation \
    --resume './detr/checkpoints/detr-r101-dc5-a2e86def.pth' \
    --dataset_file 'skill' \
    --output_dir './detr/output/' \
    --batch_size 2 \
    --epochs 10 \
    --lr 1e-5 \
    --lr_backbone 1e-6 \
    --num_classes 91 \
    ${@:2}