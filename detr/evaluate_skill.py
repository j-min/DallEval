
from pathlib import Path
import json
import torch

from models import build_model
from main import get_args_parser

if __name__ == '__main__':
    parser = get_args_parser()

    parser.add_argument('--paintskills_dir', type=str, default='../datasets/PaintSkills')

    parser.add_argument('--gt_data_eval', action='store_true', help="whether to evaluate on the ground truth data")

    parser.add_argument('--gen_model', type=str, default='dalle_small', help="which model images to be evaluated")
    parser.add_argument('--FT', action='store_true', help="whether to use the finetuned model images")

    parser.add_argument('--p_threshold', type=float, default=0.7)

    parser.add_argument('--result_dump_path', type=str, default=None, help="path to dump the evaluation results")



    args = parser.parse_args()

    args.split = 'val'
    skill_name = args.skill_name
    split = args.split

    if args.result_dump_path is not None:
        result_dump_path = Path(args.result_dump_path)
        result_dump_dir = result_dump_path.parent
        assert result_dump_dir.is_dir(), result_dump_dir
        print("Result will be dumped at:", result_dump_path)

    # Load DETR checkpoint
    args.resume = f'./output/{skill_name}/checkpoint.pth'

    if skill_name == 'color':
        args.num_colors = 6

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

    if args.gt_data_eval:
        image_dir = paintskills_dir.joinpath(f'{skill_name}/images/')
    else:
        if 'dalle_small' in args.gen_model:
            if args.FT:
                dump_name = f'{skill_name}_CC_{split}'
            else:
                dump_name = f'{skill_name}_CCzero_{split}'
            image_dir = paintskills_dir.joinpath(f'DALLE_inference/{dump_name}')

        elif 'rudalle' in args.gen_model:
            if args.FT:
                dump_name = f'{skill_name}_{split}'
            else:
                dump_name = f'{skill_name}_zero_{split}'

            image_dir = paintskills_dir.joinpath(f'rudalle_inference/{dump_name}')

        elif 'mindalle' in args.gen_model:
            if args.FT:
                dump_name = f'{skill_name}_CC_{split}'
            else:
                dump_name = f'{skill_name}_CCzero_{split}'

            image_dir = paintskills_dir.joinpath(f'minDALLE_inference/{dump_name}')

        elif 'xlxmert' in args.gen_model:
            assert args.FT is False
            dump_name = f'{skill_name}_{split}'
            image_dir = paintskills_dir.joinpath(f'XLXMERT_inference/{dump_name}')


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

    elif skill_name == 'color':
        from datasets.skill_eval import ColorDataset, eval_color

        dataset = ColorDataset(
            image_dir=image_dir,
            ann_path=ann_path,
            metadata_path=metadata_path,
            args=args,
        )

        results = eval_color(dataset, model, args)

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
        json.dump(results, open(args.result_dump_path, 'w'))
