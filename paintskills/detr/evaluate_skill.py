
from pathlib import Path
import json
import torch

from models import build_model
from main import get_args_parser

if __name__ == '__main__':
    parser = get_args_parser()

    parser.add_argument('--gt_data_eval', action='store_true', help="whether to evaluate on the ground truth data")

    parser.add_argument('--gen_model', type=str, default=None)

    parser.add_argument('--FT', action='store_true')

    parser.add_argument('--p_threshold', type=float, default=0.7)

    parser.add_argument('--result_dump_path', type=str, default=None, help="path to dump the evaluation results")

    parser.add_argument('--image_dir', type=str, default=None)

    parser.add_argument('--ignore_other_classes', action='store_true')

    parser.add_argument('--split', type=str, default='val')


    args = parser.parse_args()

    skill_name = args.skill_name
    split = args.split

    if args.result_dump_path is not None:
        result_dump_path = Path(args.result_dump_path)
        result_dump_dir = result_dump_path.parent
        assert result_dump_dir.is_dir(), result_dump_dir
        print("Result will be dumped at:", result_dump_path)

    # Load DETR checkpoint
    if args.resume == '':
        if args.FT:
            if args.resume == '':
                args.resume = '/nas-ssd/jmincho/datasets/PaintSkillsNEW/checkpoints/DETR_skills/allskills/checkpoint.pth'
                # args.resume = f'./output/{skill_name}/checkpoint.pth'
        else:
            if args.backbone == 'resnet50':
                args.resume = './checkpoints/detr-r50-e632da11.pth'
            else:
                args.resume = './checkpoints/detr-r101-dc5-a2e86def.pth'

    print(args)
    print('Building model')
    model, criterion, postprocessors = build_model(args)
    # model_without_ddp = model
    model.eval()

    print('Loading checkpoint from', args.resume)
    checkpoint = torch.load(args.resume, map_location='cpu')

    # if args.num_classes and checkpoint['model']['class_embed.weight'].size(0) != args.num_classes + 1:
    #     checkpoint['model'].pop('class_embed.weight')
    #     checkpoint['model'].pop('class_embed.bias')
    load_result = model.load_state_dict(checkpoint['model'], strict=False)
    print(load_result)

    assert args.paintskills_dir is not None, f"Please specify --paintskills_dir, current: {args.paintskills_dir}"

    paintskills_dir = Path(args.paintskills_dir)
    print('paintskill_dir:', paintskills_dir)

    metadata_path = paintskills_dir.joinpath('metadata.json')
    print('metadata:', metadata_path)

    if args.image_dir is None:
        if args.gt_data_eval:
            image_dir = paintskills_dir.joinpath(f'{skill_name}/images/')
        else:
            print("Please specify --image_dir")
            exit(1)

    else:
        image_dir = Path(args.image_dir)

    print('Loading images from', image_dir)

    ann_path = paintskills_dir.joinpath(f'{skill_name}/scenes/{skill_name}_{split}.json')

    if skill_name == 'object':
        from datasets.skill_eval import ObjectDataset, eval_object

        dataset = ObjectDataset(
            image_dir=image_dir,
            ann_path=ann_path,
            metadata_path=metadata_path,
            args=args,
        )

        results = eval_object(dataset, model, args)

    elif skill_name == 'count':
        from datasets.skill_eval import CountDataset, eval_count

        dataset = CountDataset(
            image_dir=image_dir,
            ann_path=ann_path,
            metadata_path=metadata_path,
            args=args,
        )

        results = eval_count(dataset, model, args)

    elif skill_name == 'spatial':
        from datasets.skill_eval import SpatialDataset, eval_spatial

        dataset = SpatialDataset(
            image_dir=image_dir,
            ann_path=ann_path,
            metadata_path=metadata_path,
            args=args,
        )

        results = eval_spatial(dataset, model, args)

    if args.result_dump_path is not None:
        json.dump(results, open(args.result_dump_path, 'w'), indent=4)
