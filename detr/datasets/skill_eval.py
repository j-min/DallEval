# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Skill evaluator that works in distributed mode.

Mostly copy-paste from https://github.com/pytorch/vision/blob/edfd5a7/references/detection/skill_eval.py
The difference is that there is less copy-pasting from pyskilltools
in the end of the file, as python3 can suppress prints with contextlib
"""
import os
import contextlib
import copy
import numpy as np
import torch

import json

from torch.utils.data import Dataset, DataLoader
import json
from pathlib import Path
from copy import deepcopy
from PIL import Image
from tqdm import tqdm

from torchvision import transforms as T


def parseShape(shape):
    if "human" in shape:
        return "person"
    elif "van" in shape:
        return "car"
    elif "fireHydrant" in shape:
        return "fire hydrant"
    elif "trafficLight" in shape:
        return "traffic light"
    elif "diningTable" in shape:
        return "dining table"
    elif "stopSign" in shape:
        return "stop sign"
    elif "pottedPlant" in shape:
        return "potted plant"
    elif "bike" in shape:
        return "bicycle"
    else:
        return shape


class ObjectDataset(Dataset):
    def __init__(self, image_dir, ann_path, metadata_path, args=None):
        self.ann_data = json.load(open(ann_path))

        self.args = args

        self.image_dir = image_dir

        self.img_transform = T.Compose([
            # T.Resize(800),
            T.Resize((800, 800)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.metadata = json.load(open(metadata_path))

        self.shape_to_ix = {shape: i for i, shape in enumerate(self.metadata['Shape'])}

    def __len__(self):
        return len(self.ann_data['data'])

    def __getitem__(self, ix):
        datum = self.ann_data['data'][ix]

        out = deepcopy(datum)

        if self.args.gt_data_eval:
            fname = f"image_{datum['id']}"
            img_path = self.image_dir.joinpath(fname).with_suffix('.png')
        else:
            fname = datum['id']
            img_path = self.image_dir.joinpath(fname).with_suffix('.png')

        img = Image.open(img_path)

        out['img_id'] = str(img_path)

        out['width'] = img.width
        out['height'] = img.height

        img_tensor = self.img_transform(img)
        out['img_tensor'] = img_tensor

        return out

    def collate_fn(self, batch):
        out = deepcopy(batch[0])

        out['img_tensors'] = [x['img_tensor'] for x in batch]
        out['img_tensors'] = torch.stack(out['img_tensors'], 0)

        out['target'] = []
        out['img_ids'] = []
        for datum in batch:
            shape = datum['objects'][0]['shape']
            if 'human' in shape:
                shape = 'human'
            shape_id = self.shape_to_ix[shape]
            out['target'].append(shape_id)
            out['img_ids'].append(datum['img_id'])

        out['target'] = torch.LongTensor(out['target'])



        return out


def eval_object(dataset, model, args):
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=dataset.collate_fn
    )

    total = 0
    total_correct = 0

    pbar = tqdm(total=len(dataloader))

    device = 'cuda'

    model = model.to(device)

    results = []

    for batch in dataloader:
        inputs = batch['img_tensors']
        inputs = inputs.to(device)

        B = inputs.size(0)

        with torch.no_grad():
            outputs = model(inputs)

        # [B, n_queries, num_classes]
        all_probas = outputs['pred_logits'].softmax(-1)[:, :, :-1]

        # [B, num_classes]
        probas_max_query = all_probas.max(1).values

        # pred_id = probas.max(1).values.max(-1).indices

        # [B]
        pred_prob = probas_max_query.max(-1).values
        pred_id = probas_max_query.max(-1).indices

        # keep only predictions with confidence
        # pred_id[pred_prob < 0.8] = -1

        target = batch["target"].to(device)

        # print('probas:', probas)
        # print('pred_id:', pred_id)
        # print('target:', target)

        # break

        correct = pred_id == target
        # correct = correct.float()

        total += B
        total_correct += correct.sum().item()

        for j in range(B):
            results.append({
                'img_id': batch['img_ids'][j],
                'correct': correct[j].item(),
                'pred_object': 'NA' if pred_id[j].item() == -1 else dataset.metadata['Shape'][pred_id[j].item()],
                'pred_confidence': pred_prob[j].item(),
                'target_object': dataset.metadata['Shape'][target[j].item()]
            })

        acc = total_correct / total

        desc = f'Acc: {acc * 100:.2f}%'

        pbar.set_description(desc)
        pbar.update(1)

    pbar.close()

    acc = total_correct / total

    print('Total:', total)
    print('correct:', total_correct)
    print(f'Acc: {acc * 100:.2f}%')

    return results


class ColorDataset(Dataset):
    def __init__(self, image_dir, ann_path, metadata_path, args=None):
        self.ann_data = json.load(open(ann_path))

        self.args = args

        self.image_dir = image_dir

        self.img_transform = T.Compose([
            # T.Resize(800),
            T.Resize((800, 800)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.metadata = json.load(open(metadata_path))

        self.shape_to_ix = {shape: i for i, shape in enumerate(self.metadata['Shape'])}
        self.color_to_ix = {color: i for i, color in enumerate(self.metadata['Color'])}

        self.ix_to_shape = {i: shape for shape, i in self.shape_to_ix.items()}
        self.ix_to_color = {i: color for color, i in self.color_to_ix.items()}


    def __len__(self):
        return len(self.ann_data['data'])

    def __getitem__(self, ix):
        datum = self.ann_data['data'][ix]

        out = deepcopy(datum)

        if self.args.gt_data_eval:
            fname = f"image_{datum['id']}"
            img_path = self.image_dir.joinpath(fname).with_suffix('.png')
        else:
            fname = datum['id']
            img_path = self.image_dir.joinpath(fname).with_suffix('.png')

        img = Image.open(img_path)

        out['img_id'] = str(img_path)

        out['img'] = img
        out['width'] = img.width
        out['height'] = img.height

        img_tensor = self.img_transform(img)
        out['img_tensor'] = img_tensor

        return out

    def collate_fn(self, batch):
        out = deepcopy(batch[0])

        out['img_tensors'] = [x['img_tensor'] for x in batch]
        out['img_tensors'] = torch.stack(out['img_tensors'], 0)

        out['target'] = []
        out['color_target'] = []
        out['target_names'] = []
        out['color_target_names'] = []
        out['img_ids'] = []
        out['imgs'] = []
        for datum in batch:
            target_names = []
            color_target_names = []


            shape = datum['objects'][0]['shape']
            if 'human' in shape:
                shape = 'human'

            shape_id = self.shape_to_ix[shape]
            out['target'].append(shape_id)
            out['img_ids'].append(datum['img_id'])

            color = datum['objects'][0]['color']
            color_id = self.color_to_ix[color]
            out['color_target'].append(color_id)

            target_names.append(shape)
            color_target_names.append(color)

            out['target_names'].append(target_names)
            out['color_target_names'].append(color_target_names)

            out['imgs'].append(datum['img'])

        out['target'] = torch.LongTensor(out['target'])
        out['color_target'] = torch.LongTensor(out['color_target'])

        return out

from matplotlib import pyplot as plt

def eval_color(dataset, model, args):
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=dataset.collate_fn
    )

    total = 0
    total_correct = 0
    total_object_correct = 0
    total_color_correct = 0

    pbar = tqdm(total=len(dataloader))

    device = 'cuda'

    model = model.to(device)

    results = []

    for batch in dataloader:
        inputs = batch['img_tensors']
        inputs = inputs.to(device)

        B = inputs.size(0)

        with torch.no_grad():
            outputs = model(inputs)

        # [B, n_queries, num_classes]
        all_probas = outputs['pred_logits'].softmax(-1)[:, :, :-1]

        # [B, num_classes]
        probas_max_query = all_probas.max(1).values

        # pred_id = probas.max(1).values.max(-1).indices

        # [B]
        pred_prob = probas_max_query.max(-1).values
        pred_id = probas_max_query.max(-1).indices

        # pred_id[pred_prob < 0.8] = -1

        # probas = outputs['pred_logits'].softmax(-1)[:, :, :-1]  # [B, 100, n_class]
        # pred_id = probas.max(1).values.max(-1).indices  # [B, 100, n_class] -> [B, n_class] -> [B]
        target = batch["target"].to(device)
        object_correct = pred_id == target

        color_probas = outputs['pred_colors'].softmax(-1)[:, :, :-1]
        # pred_color_id = color_probas.max(1).values.max(-1).indices
        pred_obj_query_ids = all_probas.max(2).values.max(-1).indices

        # pred_color_id = color_probas.max(2).indices.gather(1, pred_obj_query_ids.view(1, B)).squeeze(0)

        color_on_queries = color_probas.max(2).indices
        pred_color_id = torch.stack([color_on_queries[b_i, query_id] for b_i, query_id in enumerate(pred_obj_query_ids.flatten())]).to(device)

        color_on_queries_prob = color_probas.max(2).values
        pred_color_prob = torch.stack([color_on_queries_prob[b_i, query_id]
                                      for b_i, query_id in enumerate(pred_obj_query_ids.flatten())]).to(device)

        color_target = batch["color_target"].to(device)
        color_correct = pred_color_id == color_target

        correct = object_correct * color_correct

        # break

        # correct = correct.float()

        for j in range(B):
            results.append({
                'img_id': batch['img_ids'][j],
                'target_names': batch['target_names'][j],
                'color_target_names': batch['color_target_names'][j],
                # 'pred_names': [dataset.ix_to_shape[x] for x in pred_id[j].view(-1).tolist()],
                'pred_object': 'NA' if pred_id[j].item() == -1 else dataset.metadata['Shape'][pred_id[j].item()],
                'pred_color_names': [dataset.ix_to_color[x] for x in pred_color_id[j].view(-1).tolist()],
                'pred_object_confidence': pred_prob[j].item(),
                'pred_color_confidence': pred_color_prob[j].item(),
                'correct': correct[j].item(),
            })

        total += B
        total_correct += correct.sum().item()
        total_object_correct += object_correct.sum().item()
        total_color_correct += color_correct.sum().item()

        acc = total_correct / total
        object_acc = total_object_correct / total
        color_acc = total_color_correct / total

        desc = f'Acc: {acc * 100:.2f}% | Object Acc: {object_acc * 100:.2f}% | Color Acc: {color_acc * 100:.2f}%'

        pbar.set_description(desc)
        pbar.update(1)

    pbar.close()

    acc = total_correct / total

    print('Total:', total)
    print('# correct:', total_correct)
    print('# object correct:', total_object_correct)
    print('# color correct:', total_color_correct)
    print(f'Acc: {acc * 100:.2f}%')
    print(f'Object Acc: {object_acc * 100:.2f}%')
    print(f'Color Acc: {color_acc * 100:.2f}%')

    return results


class CountDataset(Dataset):
    def __init__(self, image_dir, ann_path, metadata_path, args=None):
        self.ann_data = json.load(open(ann_path))

        self.args = args

        self.image_dir = image_dir

        self.img_transform = T.Compose([
            # T.Resize(800),
            T.Resize((800, 800)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.metadata = json.load(open(metadata_path))

        self.shape_to_ix = {shape: i for i, shape in enumerate(self.metadata['Shape'])}

    def __len__(self):
        # return len(self.ann_data['data']) - 1
        return len(self.ann_data['data'])

    def __getitem__(self, ix):
        # datum = self.ann_data['data'][ix + 1]
        datum = self.ann_data['data'][ix]

        out = deepcopy(datum)


        if self.args.gt_data_eval:
            fname = f"image_{datum['id']}"
            img_path = self.image_dir.joinpath(fname).with_suffix('.png')
        else:
            fname = datum['id']
            img_path = self.image_dir.joinpath(fname).with_suffix('.png')

        img = Image.open(img_path)

        out['img_id'] = str(img_path)

        out['width'] = img.width
        out['height'] = img.height

        img_tensor = self.img_transform(img)
        out['img_tensor'] = img_tensor

        return out

    def collate_fn(self, batch):
        out = deepcopy(batch[0])

        out['img_tensors'] = [x['img_tensor'] for x in batch]
        out['img_tensors'] = torch.stack(out['img_tensors'], 0)

        out['target'] = []
        out['count_target'] = []
        out['img_ids'] = []

        for i, datum in enumerate(batch):
            out['img_ids'].append(datum['img_id'])

            for j, obj in enumerate(datum['objects']):
                shape = obj['shape']
                if 'human' in shape:
                    shape = 'human'
                shape_id = self.shape_to_ix[shape]

                if j == 0:
                    out['target'].append(shape_id)
                    out['count_target'].append(len(datum['objects']))
                else:
                    break

        out['target'] = torch.LongTensor(out['target'])
        out['count_target'] = torch.LongTensor(out['count_target'])

        return out


def eval_count(dataset, model, args):
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=dataset.collate_fn
    )

    total = 0
    total_correct = 0
    total_top_object_correct = 0
    total_count_correct = 0

    pbar = tqdm(total=len(dataloader))

    device = 'cuda'

    model = model.to(device)

    results = []

    for batch in dataloader:
        inputs = batch['img_tensors']
        inputs = inputs.to(device)

        B = inputs.size(0)

        with torch.no_grad():
            outputs = model(inputs)

        # [B, num_queries, num_classes]
        probas = outputs['pred_logits'].softmax(-1)[:, :, :-1]

        # [B, num_queries]
        max_prob = probas.max(-1).values

        assert max_prob.shape == (B, args.num_queries), max_prob.shape

        # [B]
        if args.p_threshold is not None:
            pred_n = (max_prob > args.p_threshold).sum(1)

        n_pred_objs_batch = pred_n.clamp(min=1)

        count_correct = pred_n == batch['count_target'].to(device)

        correct = torch.zeros(B).to(device)
        for i in range(B):
            gt_n = batch['count_target'][i].item()

            target_class = batch['target'][i].item()

            n_pred_objs = n_pred_objs_batch[i].item()

            # [num_queries]
            # top_n_indices = probas[i, :, target_class].topk(gt_n).indices
            top_n_indices = probas[i, :, target_class].topk(n_pred_objs).indices
            assert top_n_indices.shape == (n_pred_objs,), top_n_indices.shape

            # [n_pred_objs num_classes]
            probas_n_objs = probas[i][top_n_indices]
            assert probas_n_objs.shape == (n_pred_objs, args.num_classes)

            probas_n_objs_id = probas_n_objs.max(1).indices
            assert probas_n_objs_id.shape == (n_pred_objs,)

            # all predicted obj class should be GT obj classes
            correct_n = probas_n_objs_id == target_class
            correct[i] = correct_n.sum().item() == gt_n


        for j in range(B):
            results.append({
                'img_id': batch['img_ids'][j],
                'correct': correct[j].item(),
                'pred_count': n_pred_objs_batch[j].item(),
                'count_target': batch['count_target'][j].item(),
                'count_correct': count_correct[j].item(),
            })

        total += B
        total_correct += correct.sum().item()
        total_count_correct += count_correct.sum().item()

        acc = total_correct / total
        count_acc = total_count_correct / total
        # color_acc = total_color_correct / total

        desc = f'Acc: {acc * 100:.2f}% | count Acc: {count_acc * 100:.2f}%'

        pbar.set_description(desc)
        pbar.update(1)

    pbar.close()

    acc = total_correct / total

    print('Total:', total)
    print('# correct:', total_correct)
    print('# count correct:', total_count_correct)
    print(f'Acc: {acc * 100:.2f}%')
    print(f'count Acc: {count_acc * 100:.2f}%')

    return results


class SpatialDataset(Dataset):
    def __init__(self, image_dir, ann_path, metadata_path, args=None):
        self.ann_data = json.load(open(ann_path))

        self.args = args

        self.image_dir = image_dir

        self.img_transform = T.Compose([
            # T.Resize(800),
            T.Resize((800, 800)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.metadata = json.load(open(metadata_path))


        self.shape_to_ix = {shape: i for i, shape in enumerate(self.metadata['Shape'])}
        self.ix_to_shape = {i: shape for shape, i in self.shape_to_ix.items()}

    def __len__(self):
        return len(self.ann_data['data'])

    def __getitem__(self, ix):
        datum = self.ann_data['data'][ix]

        out = deepcopy(datum)

        if self.args.gt_data_eval:
            fname = f"image_{datum['id']}"
            img_path = self.image_dir.joinpath(fname).with_suffix('.png')
        else:
            fname = datum['id']
            img_path = self.image_dir.joinpath(fname).with_suffix('.png')

        img = Image.open(img_path)

        out['img_id'] = str(img_path)
        out['img'] = img

        out['width'] = img.width
        out['height'] = img.height

        img_tensor = self.img_transform(img)
        out['img_tensor'] = img_tensor

        return out

    def collate_fn(self, batch):
        out = deepcopy(batch[0])

        out['img_tensors'] = [x['img_tensor'] for x in batch]
        out['img_tensors'] = torch.stack(out['img_tensors'], 0)

        out['objA'] = []
        out['objB'] = []
        out['GT_objs'] = []
        out['relations_target'] = []

        out['img_ids'] = []
        out['imgs'] = []

        for i, datum in enumerate(batch):
            out['img_ids'].append(datum['img_id'])
            out['imgs'].append(datum['img'])

            for j, obj in enumerate(datum['objects']):
                shape = obj['shape']
                if 'human' in shape:
                    shape = 'human'
                shape_id = self.shape_to_ix[shape]

                out['GT_objs'].append(shape_id)

                if j == 0:
                    out['objA'].append(shape_id)
                    # out['count_target'].append(len(datum['objects']))
                elif j == 1:
                    out['objB'].append(shape_id)
                    relation = obj['relation']

                    assert relation is not None

                    # left_0
                    relation, relative_idx = relation.split('_')
                    assert relative_idx == '0'
                    assert relation in ['above', 'below', 'left', 'right']

                    out['relations_target'].append(relation)
                    # break

        out['objA'] = torch.LongTensor(out['objA'])
        out['objB'] = torch.LongTensor(out['objB'])
        out['GT_objs'] = torch.LongTensor(out['GT_objs'])

        return out


def eval_spatial(dataset, model, args):
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=dataset.collate_fn
    )

    total = 0
    total_correct = 0
    total_obj_correct = 0
    # total_top_object_correct = 0
    total_spatial_correct = 0

    pbar = tqdm(total=len(dataloader))

    device = 'cuda'

    model = model.to(device)

    results = []

    for batch in dataloader:
        inputs = batch['img_tensors']
        inputs = inputs.to(device)

        B = inputs.size(0)

        with torch.no_grad():
            outputs = model(inputs)

        # [B, num_queries, num_classes]
        probas = outputs['pred_logits'].softmax(-1)[:, :, :-1]

        # [B, num_queries]
        max_prob = probas.max(2).values
        class_id = probas.max(2).indices

        assert max_prob.shape == (B, args.num_queries), max_prob.shape

        # [B, 2]
        top2_query_indices = max_prob.topk(2, dim=1).indices.tolist()
        top2_query_probs = max_prob.topk(2, dim=1).values.tolist()

        assert len(batch['relations_target']) == B


        correct = torch.ones(B).to(device) * -1
        obj_correct = torch.ones(B).to(device) * -1

        pred_objs_ids = []
        gt_objs_ids = []
        pred_rels = []

        pred_obj_locs = []
        pred_obj_bbox = []

        for i in range(B):

            assert batch['relations_target'][i] in ['left', 'right', 'above', 'below']

            objA_class = batch['objA'][i].item()
            objB_class = batch['objB'][i].item()

            # pred_n_i = pred_n[i].item()

            # [num_queries]
            # top_n_indices = probas[i, :, target_class].topk(gt_n).indices
            # [2, num_classes]
            # top_n_indices = probas[i, :, :].topk(2, dim=0).indices

            assert len(top2_query_indices[i]) == 2, top2_query_indices[i]
            obj_C_query, obj_D_query = top2_query_indices[i]

            pred_obj_C = class_id[i, obj_C_query].item()
            pred_obj_D = class_id[i, obj_D_query].item()

            gt_objs_ids.append([objA_class, objB_class])
            pred_objs_ids.append([pred_obj_C, pred_obj_D])

            pred_obj_C_bbox = outputs['pred_boxes'][i, obj_C_query].tolist()
            pred_obj_D_bbox = outputs['pred_boxes'][i, obj_D_query].tolist()

            assert len(pred_obj_C_bbox) == 4, pred_obj_C_bbox

            # print('objA, objB', objA_class, objB_class)
            # print('objC, objD', pred_obj_C, pred_obj_D)

            # x_c, y_c, w, h = pred_obj_C_bbox
            x_C, y_C = pred_obj_C_bbox[:2]

            # x_c, y_c, w, h = pred_obj_D_bbox
            x_D, y_D = pred_obj_D_bbox[:2]

            x_diff = x_C - x_D
            y_diff = y_C - y_D

            pred_obj_bbox.append([pred_obj_C_bbox, pred_obj_D_bbox])
            pred_obj_locs.append([(x_C, y_C), (x_D, y_D)])

            if objA_class == objB_class:
                if objA_class == pred_obj_C and pred_obj_C == pred_obj_D:

                    # left/right
                    if abs(x_diff) > abs(y_diff):
                        if batch['relations_target'][i] in ['left', 'right']:
                            pred_rel = batch['relations_target'][i]
                            correct[i] = 1
                        else:
                            correct[i] = 0
                    # above/below
                    else:
                        if batch['relations_target'][i] in ['above', 'below']:
                            pred_rel = batch['relations_target'][i]
                            correct[i] = 1
                        else:
                            correct[i] = 0
                    obj_correct[i] = 1
                else:
                    pred_rel = 'obj_relation_incorrect'

                    correct[i] = 0
                    obj_correct[i] = 0

            else:
                if (objA_class, objB_class) == (pred_obj_C, pred_obj_D):
                    obj_correct[i] = 1

                    # left/right
                    if abs(x_diff) > abs(y_diff):
                        if x_C < x_D:
                            pred_rel = 'right'
                        else:
                            pred_rel = 'left'
                    # above/below
                    else:
                        if y_C > y_D:
                            pred_rel = 'above'
                        else:
                            pred_rel = 'below'

                    if pred_rel == batch['relations_target'][i]:
                        correct[i] = 1
                    else:
                        correct[i] = 0
                    # obj_correct[i] = 1

                    # import ipdb; ipdb.set_trace()

                elif (objA_class, objB_class) == (pred_obj_D, pred_obj_C):

                    obj_correct[i] = 1

                    # left/right
                    if abs(x_diff) > abs(y_diff):
                        if x_C < x_D:
                            pred_rel = 'left'
                        else:
                            pred_rel = 'right'

                    # above/below
                    else:
                        if y_C > y_D:
                            pred_rel = 'below'
                        else:
                            pred_rel = 'above'

                    if pred_rel == batch['relations_target'][i]:
                        correct[i] = 1
                    else:
                        correct[i] = 0
                    # obj_correct[i] = 1

                else:
                    pred_rel = 'obj_not_matching'
                    correct[i] = 0
                    obj_correct[i] = 0

                assert correct[i].min().item() in [0,1], correct[i]

            pred_rels.append(pred_rel)

        # # break
        # for j in range(B):
        #     results.append({
        #         'img_id': batch['img_ids'][j],
        #         'correct': correct[j].item(),
        #     })

        for j in range(B):
            results.append({
                'img_id': batch['img_ids'][j],
                'GT_objs': (dataset.ix_to_shape[gt_objs_ids[j][0]], dataset.ix_to_shape[gt_objs_ids[j][1]]),
                'pred_objs': (dataset.ix_to_shape[pred_objs_ids[j][0]], dataset.ix_to_shape[pred_objs_ids[j][1]]),
                'pred_obj_bbox': pred_obj_bbox[j],
                'pred_obj_locs': pred_obj_locs[j],
                'GT_rel': batch['relations_target'][j],
                'pred_rel': pred_rels[j],
                'correct': bool(correct[j].item()),
                'obj_correct': bool(obj_correct[j].item()),
            })

        total += B
        total_correct += correct.sum().item()
        total_obj_correct += obj_correct.sum().item()

        acc = total_correct / total
        obj_acc = total_obj_correct / total
        # color_acc = total_color_correct / total

        desc = f'Acc: {acc * 100: .2f} % | Obj Acc: {obj_acc * 100: .2f}%'

        pbar.set_description(desc)
        pbar.update(1)

    pbar.close()

    acc = total_correct / total

    print('Total:', total)
    print('# correct:', total_correct)
    # print('# count correct:', total_count_correct)
    print(f'Acc: {acc * 100:.2f}%')
    print(f'Obj Acc: {obj_acc * 100:.2f}%')

    return results
