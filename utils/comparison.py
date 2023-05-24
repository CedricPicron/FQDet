"""
Collection of comparison utilities.
"""
from collections import defaultdict
import json
from pathlib import Path
from zipfile import ZipFile

import pycocotools.mask as maskUtils
import torch

from datasets.coco.coco import SPLITS as COCO_SPLITS


def compare_models(dataset, eval_split, res_file_name1, res_file_name2, output_dir=None, save_name=None):
    """
    Compares two models w.r.t. the desired ground-truth annotations based on the two given result files.

    Args:
        dataset (str): String containing the dataset name for which result files were obtained.
        eval_split (str): String containing the evaluation split for which result files were obtained.
        res_file_name1 (str): String containing the path to the first result file.
        res_file_name2 (str): String containing the path to the second result file.
        output_dir (Path): Path to directory to save the comparison results (default=None).
        save_name (str): String containing the name of the file saving the comparison results (default=None).

    Raises:
        ValueError: Error when an unsupported or invalid dataset is provided.
        ValueError: Error when the zipped result file contains multiple files.
        ValueError: Error when the zipped result file contains file that does not have json extension.
        ValueError: Error when input result file does not have zip or json extension.
        ValueError: Error when the results dictionaries contain neither the 'segmentation' nor the 'bbox' key.
    """

    # Get ground-truth images, categories and annotations
    if dataset == 'coco':
        gt_file_name = Path() / 'datasets' / COCO_SPLITS['coco'][f'coco_{eval_split}'][1]

        with open(gt_file_name, 'r') as gt_file:
            gt_data = json.load(gt_file)

            img_ids = [img_dict['id'] for img_dict in gt_data['images']]
            cat_ids = [cat_dict['id'] for cat_dict in gt_data['categories']]

            gt_ann_dicts = [ann_dict for ann_dict in gt_data['annotations'] if not ann_dict.get('iscrowd', False)]
            imgs_dict = {img_dict['id']: img_dict for img_dict in gt_data['images']}

    else:
        error_msg = f'Unsupported or invalid dataset for results comparison (got {dataset}).'
        raise ValueError(error_msg)

    # Get list with result dictionaries from each result file
    res_dicts_list = []

    for res_file_name in (res_file_name1, res_file_name2):
        extension = res_file_name.split('.')[-1]

        if extension == 'zip':
            with ZipFile(res_file_name, 'r') as zip_file:
                file_names = zip_file.namelist()
                extension = file_names[0].split('.')[-1]

                if len(file_names) > 1:
                    num_files = len(file_names)
                    error_msg = f"The zipped result file should only contain a single file, but got {num_files}."
                    raise ValueError(error_msg)

                elif extension != 'json':
                    error_msg = f"The zipped result file should have json extension, but got {extension} extension."

                with zip_file.open(file_names[0], 'r') as json_file:
                    res_dicts = json.load(json_file)
                    res_dicts_list.append(res_dicts)

        elif extension == 'json':
            with open(res_file_name, 'r') as json_file:
                res_dicts = json.load(json_file)
                res_dicts_list.append(res_dicts)

        else:
            error_msg = f"Input result file should have zip or json extension, but got {extension} extension instead."
            raise ValueError(error_msg)

    # Get IoU type
    if 'segmentation' in res_dicts_list[0][0]:
        iou_type = 'segm'

    elif 'bbox' in res_dicts_list[0][0]:
        iou_type = 'bbox'

    else:
        error_msg = "Result dictionaries should contain the 'segmentation' or 'bbox' key, but both are missing."
        raise ValueError(error_msg)

    # Make sure segmentations are in desired format
    if iou_type == 'segm':
        ann_dicts = gt_ann_dicts + res_dicts_list[0] + res_dicts_list[1]

        for ann_dict in ann_dicts:
            seg_ann = ann_dict['segmentation']

            if type(seg_ann) == list or type(seg_ann['counts']) == list:
                img_dict = imgs_dict[ann_dict['image_id']]
                height, width = img_dict['height'], img_dict['width']

                rle = maskUtils.frPyObjects(seg_ann, height, width)
                rle = maskUtils.merge(rle) if type(seg_ann) == list else rle
                ann_dict['segmentation'] = rle

    # Collect annotations per image and category
    ann_dicts_list = [gt_ann_dicts, *res_dicts_list]
    img_cat_dicts_list = [defaultdict(list), defaultdict(list), defaultdict(list)]

    for ann_dicts, img_cat_dicts in zip(ann_dicts_list, img_cat_dicts_list):
        for ann_dict in ann_dicts:
            img_id = ann_dict['image_id']
            cat_id = ann_dict['category_id']
            img_cat_dicts[img_id, cat_id].append(ann_dict)

    gt_img_cat_dicts = img_cat_dicts_list[0]
    res_img_cat_dicts_list = [img_cat_dicts_list[1], img_cat_dicts_list[2]]

    # Get some additional settings depending on dataset
    if dataset == 'coco':
        max_dets = 100
        iou_thr = 0.5

    # Compare results
    one_better = 0
    two_better = 0
    both_matched = 0
    one_missed = 0
    two_missed = 0
    both_missed = 0
    total = 0

    for img_id in img_ids:
        for cat_id in cat_ids:
            gt_dicts = gt_img_cat_dicts[img_id, cat_id]
            num_gts = len(gt_dicts)

            if num_gts == 0:
                continue

            gt_matched = torch.zeros(num_gts, 2, dtype=torch.bool)
            gt_ious = torch.zeros(num_gts, 2, dtype=torch.float)

            for res_id in range(2):
                res_dicts = res_img_cat_dicts_list[res_id][img_id, cat_id]
                res_dicts = sorted(res_dicts, key=lambda d: d['score'], reverse=True)
                res_dicts = res_dicts[:max_dets]

                if iou_type == 'segm':
                    preds = [res_dict['segmentation'] for res_dict in res_dicts]
                    targets = [gt_dict['segmentation'] for gt_dict in gt_dicts]

                elif iou_type == 'bbox':
                    preds = [res_dict['bbox'] for res_dict in res_dicts]
                    targets = [gt_dict['bbox'] for gt_dict in gt_dicts]

                iscrowd = [0] * len(targets)
                ious = maskUtils.iou(preds, targets, iscrowd)

                for iou_pred in ious:
                    iou_to_beat = iou_thr
                    matched_gt_id = -1

                    for gt_id, iou_pred_gt in enumerate(iou_pred):
                        if (not gt_matched[gt_id, res_id].item()) & (iou_pred_gt > iou_to_beat):
                            iou_to_beat = iou_pred_gt
                            matched_gt_id = gt_id

                    if matched_gt_id != -1:
                        gt_matched[matched_gt_id, res_id] = True
                        gt_ious[matched_gt_id, res_id] = iou_to_beat

            both_matched_mask = gt_matched.all(dim=1)
            gt_ious = gt_ious[both_matched_mask, :]

            one_better += (gt_ious[:, 0] > gt_ious[:, 1]).sum().item()
            two_better += (gt_ious[:, 1] > gt_ious[:, 0]).sum().item()

            both_matched += len(gt_ious)
            one_missed += (~gt_matched[:, 0] & gt_matched[:, 1]).sum().item()
            two_missed += (gt_matched[:, 0] & ~gt_matched[:, 1]).sum().item()
            both_missed += (~gt_matched).all(dim=1).sum().item()
            total += num_gts

    # Get final comparison results
    one_better = 100 * one_better / total
    two_better = 100 * two_better / total

    one_better_matched = 100 * one_better / (one_better + two_better)
    two_better_matched = 100 - one_better_matched

    both_matched = 100 * both_matched / total
    one_missed = 100 * one_missed / total
    two_missed = 100 * two_missed / total
    both_missed = 100 * both_missed / total

    # Get string with comparison results
    comp_str = "\n================================\n"
    comp_str += "|  Model comparison (results)  |\n"
    comp_str += "================================\n\n"

    comp_str += f"One better: {one_better:5.2f} ({one_better_matched:5.2f})\n"
    comp_str += f"Two better: {two_better:5.2f} ({two_better_matched:5.2f})\n\n"

    comp_str += f"Both matched: {both_matched:5.2f}\n"
    comp_str += f"One missed:   {one_missed:5.2f}\n"
    comp_str += f"Two missed:   {two_missed:5.2f}\n"
    comp_str += f"Both missed:  {both_missed:5.2f}\n"

    # Print comparison results
    print(comp_str)

    # Save comparison results if output directory is provided
    if output_dir is not None:
        save_name = f'{save_name}.txt' if save_name else 'model_comparison.txt'

        with (output_dir / save_name).open('w') as comp_file:
            comp_file.write(comp_str)
