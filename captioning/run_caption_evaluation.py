import sys
from tqdm import tqdm
from pathlib import Path
import json
import torch
from pprint import pprint


sys.path.append('./VL-T5/VL-T5/src')
from param import parse_args
from caption import Trainer


# Faster RCNN traineed on VG (Please check README.md for more details)
sys.path.append('./VL-T5')
from inference.processing_image import Preprocess
from inference.modeling_frcnn import GeneralizedRCNN
from inference.utils import Config


if __name__ == '__main__':
    args = parse_args(
        parse=False,
    )
    args.gpu = 0
    print(args)

    args.ckpt_path = 'VLT5_HF_FRCNN_COCOCaption.pth'
    args.image_dir = "image_dir"
    args.eval_results_path = "eval_results.json"

    trainer = Trainer(args, train=False)

    trainer.load_checkpoint(args.ckpt_path)
    trainer.model.eval();
    print('Loaded VL-T5 captioning model!')

    frcnn_cfg = Config.from_pretrained("unc-nlp/frcnn-vg-finetuned")
    frcnn_cfg.model.device = f'cuda:{args.gpu}'
    frcnn = GeneralizedRCNN.from_pretrained("unc-nlp/frcnn-vg-finetuned", config=frcnn_cfg)
    image_preprocess = Preprocess(frcnn_cfg)
    print('Loaded FRCNN!')


    import language_evaluation
    class COCOCaptionEvaluator:
        def __init__(self):
            self.evaluator = language_evaluation.CocoEvaluator(verbose=False)

        def evaluate(self, predicts, answers):
            results = self.evaluator.run_evaluation(predicts, answers)
            return results

    evaluator = COCOCaptionEvaluator()
    print('Loaded language evaluator!')

    karpathy_data = json.load(open('./dataset_coco.json'))

    split_rename = {
        'train': 'train',
        'restval': 'train',
        'val': 'val',
        'test': 'test'
    }

    source = 'test'

    data = []
    for datum in karpathy_data['images']:
        re_split = split_rename[datum['split']]
        if re_split != source:
            continue

        if re_split == 'train':
            for d in datum['sentences']:
                img_id = datum['filename'].split('.')[0]
                new_datum = {
                    'img_id': img_id,
                    'sent': d['raw'].strip(),
                    'targets': [d['raw'].strip() for d in datum['sentences']],
                    'is_train': True,
                }
                data.append(new_datum)
        else:
            img_id = datum['filename'].split('.')[0]
            new_datum = {
                'img_id': img_id,
                # 'sent': d['raw'],
                'targets': [d['raw'].strip() for d in datum['sentences']],
                'is_train': False,
            }
            data.append(new_datum)

    imgid2targets = {}
    max_n_target = 0
    for datum in data:
        imgid2targets[datum['img_id']] = datum['targets']
        max_n_target = max(max_n_target, len(datum['targets']))
    print('Loaded Karpathy test')


    print('results will be saved at', args.eval_results_path)

    predictions = []
    targets = []

    model_img_dir = Path(args.image_dir)
    fname_list = list(model_img_dir.glob('*.jpg'))

    for i, fname in enumerate(tqdm(fname_list, desc='Generating captions...')):
        with torch.no_grad():

            images, sizes, scales_yx = image_preprocess(str(fname))

            output_dict = frcnn(
                images,
                sizes,
                scales_yx=scales_yx,
                padding='max_detections',
                max_detections=frcnn_cfg.max_detections,
                return_tensors='pt'
            )

            normalized_boxes = output_dict.get("normalized_boxes")
            features = output_dict.get("roi_features")

            batch = {}
            batch['vis_feats'] = features
            batch['input_ids'] = trainer.tokenizer(['caption:'], return_tensors='pt').input_ids
            batch['boxes'] = normalized_boxes

            gen_caption = trainer.model.test_step(
                batch
            )['pred'][0]

        predictions.append(gen_caption)

        img_id = fname.stem
        targets.append(imgid2targets[img_id])

    eval_results = evaluator.evaluate(predictions, targets)
    print('Eval results')
    pprint(eval_results)

    results = {
        'predictions': predictions,
        'targets': targets,
        'img_ids': list(str(fname.name) for fname in fname_list),
        'metrics': eval_results
    }

    with open(args.eval_results_path, 'w') as f:
        json.dump(results, f, indent=4)