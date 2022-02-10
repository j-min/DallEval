
n_gpus=$1

python -m torch.distributed.launch \
    --nproc_per_node=$n_gpus \
    --use_env \
    --master_port=23456 \
    detr/main.py \
    --resume './detr/checkpoints/detr-r50-e632da11.pth' \
    --dataset_file 'skill' \
    --output_dir './detr/output/' \
    --batch_size 16 \
    --epochs 10 \
    --lr 1e-5 \
    --lr_backbone 1e-6 \
    --num_classes 21 \
    ${@:2}