
from pathlib import Path
import json
import torch

from models import build_model
from main import get_args_parser

if __name__ == '__main__':
    parser = get_args_parser()

    parser.add_argument('--split', type=str, default='val')
    parser.add_argument('--image_dir', type=str, default=None)
    parser.add_argument('--gt_data_eval', action='store_true', help="whether to evaluate on the ground truth data")
    parser.add_argument('--result_dump_path', type=str, default=None, help="path to dump the evaluation results")
    parser.add_argument('--viz_save_dir', type=str, default=None, help="path to dump the visualization results; ignored if None")
    parser.add_argument('--ignore_other_classes', action='store_true', help="whether to ignore other classes")
    parser.add_argument('--shuffle_baseline', action='store_true', help="baseline - shuffle the images - should score very low accuracy")
    parser.add_argument('--p_threshold', type=float, default=0.7)

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
                if Path('./checkpoints/paintskills_detr_r50.pth').exists():
                    pass
                else:
                    from huggingface_hub import hf_hub_download
                    hf_hub_download(repo_id="j-min/PaintSkills-DETR-R101-DC5", filename="paintskills_detr_r101_dc5.pth", local_dir='./checkpoints/')
                args.resume = './checkpoints/paintskills_detr_r50.pth'

    print('arguments')
    print(args)
    print('='*30)

    print('='*30)
    print('Building model')
    print('='*30)
    model, criterion, postprocessors = build_model(args)
    model.eval()

    print('='*30)
    print('Loading checkpoint from', args.resume)
    print('='*30)
    checkpoint = torch.load(args.resume, map_location='cpu')
    load_result = model.load_state_dict(checkpoint, strict=False)
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

    print('='*30)
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

        results = eval_object(dataset, model, postprocessors, args)

    elif skill_name == 'count':
        from datasets.skill_eval import CountDataset, eval_count

        dataset = CountDataset(
            image_dir=image_dir,
            ann_path=ann_path,
            metadata_path=metadata_path,
            args=args,
        )

        results = eval_count(dataset, model, postprocessors, args)

    elif skill_name == 'spatial':
        from datasets.skill_eval import SpatialDataset, eval_spatial

        dataset = SpatialDataset(
            image_dir=image_dir,
            ann_path=ann_path,
            metadata_path=metadata_path,
            args=args,
        )

        results = eval_spatial(dataset, model, postprocessors, args)

    if args.result_dump_path is not None:
        json.dump(results, open(args.result_dump_path, 'w'), indent=4)