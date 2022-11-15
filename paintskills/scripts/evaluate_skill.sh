
python detr/evaluate_skill.py \
    --backbone 'resnet101' --dilation \
    --batch_size 10 \
    --num_classes 91 \
    --FT \
    ${@:1}