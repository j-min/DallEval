import sys
from tqdm import tqdm
from pathlib import Path
import json
import torch
from more_itertools import chunked
from pprint import pprint


sys.path.append('./VL-T5/VL-T5/src')
from param import parse_args
from caption import Trainer


# Faster RCNN trained on VG (Please check README.md for more details)
sys.path.append('./VL-T5')
from inference.processing_image import Preprocess
from inference.modeling_frcnn import GeneralizedRCNN
from inference.utils import Config


if __name__ == '__main__':
    args = parse_args(
        parse=False,
        num_beams=5, # Don't change this
        batch_size=100, # You may change this to adjust memory/inference speed
    )
    args.gpu = 0
    print(args)

    image_dir = "image_dir" # UPDATE_THIS_PATH

    ckpt_path = 'VLT5_HF_FRCNN_COCOCaption.pth'
    eval_results_path = "eval_results.json"

    trainer = Trainer(args, train=False)

    trainer.load_checkpoint(ckpt_path)
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


    print('results will be saved at', eval_results_path)

    predictions = []
    targets = []

    fname_list = list(Path(image_dir).glob('*.jpg'))

    B = args.batch_size

    for batch_fnames in chunked(tqdm(fname_list, 'Generating captions...'), n=B):
        with torch.no_grad():

            feats = torch.zeros(B, 36, 2048)
            boxes = torch.zeros(B, 36, 4)
            for j, fname in enumerate(batch_fnames):
                img_id = fname.stem
                targets.append(imgid2targets[img_id])

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
                normalized_boxes.clamp_(min=0.0, max=1.0)
                features = output_dict.get("roi_features")

                feats[j] = features
                boxes[j] = normalized_boxes

            input_ids = trainer.tokenizer(['caption:'], return_tensors='pt').input_ids
            input_ids = input_ids.view(1, -1)
            input_ids = input_ids.expand(B, -1)

            batch = {}
            batch['input_ids'] = input_ids
            batch['vis_feats'] = feats
            batch['boxes'] = boxes

            gen_kwargs = {}
            gen_kwargs['num_beams'] = args.num_beams
            gen_kwargs['max_length'] = args.gen_max_length

            gen_caption = trainer.model.test_step(
                batch,
                **gen_kwargs
            )['pred']

        predictions.extend(gen_caption)

    eval_results = evaluator.evaluate(predictions, targets)
    print('Eval results')
    pprint(eval_results)

    results = {
        'predictions': predictions,
        'targets': targets,
        'img_ids': list(str(fname.name) for fname in fname_list),
        'metrics': eval_results
    }

    with open(eval_results_path, 'w') as f:
        json.dump(results, f, indent=4)
