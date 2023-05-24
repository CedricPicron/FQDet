"""
Cityscapes dataset/evaluator and build function.
"""
import glob
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import boundary_iou.cityscapes_instance_api.evalInstanceLevelSemanticLabeling as cityscapes_eval
from cityscapesscripts.helpers.labels import name2label
from detectron2.data import MetadataCatalog
from detectron2.data.datasets.cityscapes import load_cityscapes_instances
from detectron2.layers import batched_nms
import numpy as np
import pandas as pd
from PIL import Image
from pycocotools.mask import decode as rle_to_mask
import torch
from torch.utils.data import Dataset

from datasets.coco.coco import CocoDataset
from datasets.transforms import get_train_transforms, get_eval_transforms
from structures.boxes import Boxes
from structures.images import Images
from structures.masks import mask_inv_transform, mask_to_rle
import utils.distributed as distributed


class CityscapesDataset(Dataset):
    """
    Class implementing the CityscapesDataset dataset.

    Attributes:
        root (Path): Path to main directory from which datasets directory can be accessed.
        transforms (List): List [num_transforms] of transforms applied to image (and targets if available).
        metadata (detectron2.data.Metadata): Metadata instance containing additional dataset information.
        requires_masks (bool): Boolean indicating whether target dictionaries require masks.
        image_dicts (List): List [num_images] of dictionaries containing image-specific information.
        has_gt_anns (bool): Boolean indicating whether annotations are available.
    """

    def __init__(self, image_dir, info_dir, transforms, metadata, requires_masks=False):
        """
        Initializes the Cityscapes dataset.

        Args:
            image_dir (Path): Path to directory with Cityscapes images.
            info_dir (Path): Path to directory with information about Cityscapes images.
            transforms (List): List [num_transforms] of transforms applied to image (and targets if available).
            metadata (detectron2.data.Metadata): Metadata instance containing additional dataset information.
            requires_masks (bool): Boolean indicating whether target dictionaries require masks (default=False).
        """

        # Set base attributes
        self.root = Path()
        self.transforms = transforms
        self.metadata = metadata
        self.requires_masks = requires_masks

        # Get image dictionaries containing information about each image
        self.image_dicts = load_cityscapes_instances(image_dir, info_dir)

        # Set attribute indicating whether annotations are available
        self.has_gt_anns = False

        for image_dict in self.image_dicts:
            if len(image_dict['annotations']) > 0:
                self.has_gt_anns = True
                break

        # Filter image dictionaries if needed
        if self.has_gt_anns:
            image_dicts = []

            for image_dict in self.image_dicts:
                image_dict['annotations'] = [ann for ann in image_dict['annotations'] if not ann['iscrowd']]
                image_dicts.append(image_dict) if len(image_dict['annotations']) > 0 else None

            self.image_dicts = image_dicts

    def __getitem__(self, index):
        """
        Implements the __getitem__ method of the CityscapesDataset dataset.

        Args:
            index (int): Index selecting one of the dataset (image, transform) combinations.

        Returns:
            image (Images): Structure containing the image tensor after data augmentation.

            tgt_dict (Dict): Target dictionary potentially containing following keys (empty when no annotations):
                - labels (LongTensor): tensor of shape [num_targets] containing the class indices;
                - boxes (Boxes): structure containing axis-aligned bounding boxes of size [num_targets];
                - masks (BoolTensor): segmentation masks of shape [num_targets, iH, iW].
        """

        # Load image and place it into Images structure
        contiguous_img_id = index // len(self.transforms)
        image_dict = self.image_dicts[contiguous_img_id]
        image_path = self.root / image_dict['file_name']

        image = Image.open(image_path).convert('RGB')
        image = Images(image, contiguous_img_id)

        # Initialize empty target dictionary
        tgt_dict = {}

        # Add targets to target dictionary if annotations are provided
        if self.has_gt_anns:

            # Get annotations
            annotations = image_dict['annotations']

            # Get object class labels
            labels = [annotation['category_id'] for annotation in annotations]
            labels = torch.tensor(labels, dtype=torch.int64)

            # Get object boxes in (left, top, right, bottom) format
            boxes = [annotation['bbox'] for annotation in annotations]
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            boxes = Boxes(boxes, format='xyxy')

            # Clip boxes and only keep targets with well-defined boxes
            boxes, well_defined = boxes.clip(image.size())
            tgt_dict['labels'] = labels[well_defined]
            tgt_dict['boxes'] = boxes[well_defined]

            # Get segmentation masks if required and add to target dictionary
            if self.requires_masks:
                iW, iH = image.size()
                masks = CocoDataset.get_masks(annotations, iH, iW)
                tgt_dict['masks'] = masks[well_defined]

        # Perform image and target dictionary transformations
        transform = self.transforms[index % len(self.transforms)]
        image, tgt_dict = transform(image, tgt_dict)

        # Only keep targets with well-defined boxes
        if 'boxes' in tgt_dict:
            well_defined = tgt_dict['boxes'].well_defined()

            for key in tgt_dict.keys():
                if key in ['labels', 'boxes', 'masks']:
                    tgt_dict[key] = tgt_dict[key][well_defined]

        return image, tgt_dict

    def __len__(self):
        """
        Implements the __len__ method of the CityscapesDataset dataset.

        Returns:
            dataset_length (int): Dataset length measured as the number of images times the number of transforms.
        """

        # Get dataset length
        dataset_length = len(self.image_dicts) * len(self.transforms)

        return dataset_length


class CityscapesEvaluator(object):
    """
    Class implementing the CityscapesEvaluator evaluator.

    Attributes:
        file_names (List): List [num_images] containing the filenames of the evaluation images.
        image_ids (List): List [num_eval_images] containing the image indices of evaluated images.
        metadata (detectron2.data.Metadata): Metadata instance containing additional dataset information.
        metrics (List): List [num_metrics] with strings specifying the evaluation metrics.
        result_dicts (List): List [num_preds] with prediction-specific result dictionaries.

        eval_nms (bool): Boolean indicating whether to perform NMS during evaluation.
        nms_thr (float): IoU threshold used during evaluation NMS to remove duplicate detections.

        has_gt_anns (bool): Boolean indicating whether evaluation dataset has ground-truth annotations.
    """

    def __init__(self, eval_dataset, metrics=None, nms_thr=0.5):
        """
        Initializes the CityscapesEvaluator evaluator.

        Args:
            eval_dataset (CityscapesDataset): The evaluation dataset.
            metrics (List): List [num_metrics] with strings specifying the evaluation metrics (default=None).
            nms_thr (float): IoU threshold used during evaluation NMS to remove duplicate detections (default=0.5).
        """

        # Set base attributes
        self.file_names = [image_dict['file_name'] for image_dict in eval_dataset.image_dicts]
        self.image_ids = []
        self.metadata = eval_dataset.metadata
        self.metrics = metrics if metrics is not None else []
        self.result_dicts = []

        # Set NMS attributes
        self.eval_nms = len(eval_dataset.transforms) > 1
        self.nms_thr = nms_thr

        # Set attributes related to ground-truth annotations
        self.has_gt_anns = eval_dataset.has_gt_anns

    def add_metrics(self, metrics):
        """
        Adds valid evaluation metrics to the CityscapesEvaluator evaluator and resets.

        Args:
            metrics (List): List with strings containing the evaluation metrics to be added.
        """

        # Add valid evalutation metrics
        metrics = [metric for metric in metrics if metric != 'bbox']
        self.metrics.extend(metrics)

    def reset(self):
        """
        Resets the image_ids and result_dicts attributes of the CityscapesEvaluator evaluator.
        """

        # Reset image_ids and result_dicts attributes
        self.image_ids = []
        self.result_dicts = []

    def update(self, images, pred_dict):
        """
        Updates result dictionaries of the evaluator object based on the given images and corresponding predictions.

        Args:
            images (Images): Images structure containing the batched images.

            pred_dict (Dict): Prediction dictionary potentially containing following keys:
                - labels (LongTensor): predicted class indices of shape [num_preds];
                - boxes (Boxes): structure containing axis-aligned bounding boxes of size [num_preds];
                - masks (BoolTensor): predicted segmentation masks of shape [num_preds, mH, mW];
                - scores (FloatTensor): normalized prediction scores of shape [num_preds];
                - batch_ids (LongTensor): batch indices of predictions of shape [num_preds].

        Raises:
            ValueError: Error when no box predictions are provided when using evaluation NMS.
        """

        # Update the image_ids attribute
        self.image_ids.extend(images.image_ids)

        # Extract predictions from prediction dictionary
        labels = pred_dict['labels']
        boxes = pred_dict.get('boxes', None)
        segms = pred_dict.get('masks', None)
        scores = pred_dict['scores']
        batch_ids = pred_dict['batch_ids']

        # Transform boxes to original image space and convert them to desired format
        if boxes is not None:
            boxes, well_defined = boxes.transform(images, inverse=True)
            boxes = boxes[well_defined].to_format('xyxy')

            labels = labels[well_defined]
            segms = segms[well_defined] if segms is not None else None
            scores = scores[well_defined]
            batch_ids = batch_ids[well_defined]

        # Transform segmentation masks to original image space
        if segms is not None:
            segms = mask_inv_transform(segms, images, batch_ids)
            segms = mask_to_rle(segms)

        # Get image indices corresponding to predictions
        image_ids = torch.as_tensor(images.image_ids)
        image_ids = image_ids[batch_ids.cpu()]

        # Convert labels to original non-contiguous id space
        orig_ids = [name2label[name].id for name in self.metadata.thing_classes]
        orig_ids = labels.new_tensor(orig_ids)
        labels = orig_ids[labels]

        # Convert desired tensors to lists
        image_ids = image_ids.tolist()
        labels = labels.tolist()
        boxes = boxes.boxes.tolist() if boxes is not None else None
        scores = scores.tolist()

        # Get result dictionary
        result_dict = {}
        result_dict['image_id'] = image_ids
        result_dict['category_id'] = labels
        result_dict['score'] = scores

        if self.eval_nms:
            if boxes is not None:
                result_dict['bbox'] = boxes
            else:
                error_msg = "Box predictions must be provided when using evaluation NMS."
                raise ValueError(error_msg)

        if segms is not None:
            result_dict['segmentation'] = segms

        # Get list of result dictionaries
        result_dicts = pd.DataFrame(result_dict).to_dict(orient='records')

        # Update result_dicts attribute
        self.result_dicts.extend(result_dicts)

    def evaluate(self, device='cpu', output_dir=None, save_results=False, save_name='results'):
        """
        Perform evaluation by finalizing the result dictionaries and by comparing with ground-truth (if available).

        Args:
            device (str): String containing the type of device used during NMS (default='cpu').
            output_dir (Path): Path to output directory to save result dictionaries (default=None).
            save_results (bool): Boolean indicating whether to save the results (default=False).
            save_name (str) String containing the name of the saved result file (default='results').

        Returns:
            eval_dict (Dict): Dictionary with evaluation results for each metric (if annotations are available).
        """

        # Synchronize image indices and make them unique
        gathered_image_ids = distributed.all_gather(self.image_ids)
        self.image_ids = [image_id for list in gathered_image_ids for image_id in list]
        self.image_ids = list(np.unique(self.image_ids))

        # Synchronize result dictionaries
        gathered_result_dicts = distributed.all_gather(self.result_dicts)
        self.result_dicts = [result_dict for list in gathered_result_dicts for result_dict in list]

        # Return if not main process
        if not distributed.is_main_process():
            return

        # Peform NMS if requested
        if self.eval_nms:
            boxes = {image_id: [] for image_id in self.image_ids}
            scores = {image_id: [] for image_id in self.image_ids}
            labels = {image_id: [] for image_id in self.image_ids}
            result_ids = {image_id: [] for image_id in self.image_ids}
            keep_result_ids = []

            for result_id, result_dict in enumerate(self.result_dicts):
                image_id = result_dict['image_id']

                boxes[image_id].append(result_dict['bbox'])
                scores[image_id].append(result_dict['score'])
                labels[image_id].append(result_dict['category_id'])
                result_ids[image_id].append(result_id)

            for image_id in self.image_ids:
                boxes_i = torch.tensor(boxes[image_id], device=device)
                scores_i = torch.tensor(scores[image_id], device=device)
                labels_i = torch.tensor(labels[image_id], device=device)
                result_ids_i = torch.tensor(result_ids[image_id], device=device)

                keep_ids = batched_nms(boxes_i, scores_i, labels_i, iou_threshold=self.nms_thr)[:100]
                keep_result_ids_i = result_ids_i[keep_ids].tolist()
                keep_result_ids.extend(keep_result_ids_i)

            result_ids = set(range(len(self.result_dicts)))
            keep_result_ids = set(keep_result_ids)
            drop_result_ids = result_ids - keep_result_ids

            data = pd.DataFrame(self.result_dicts)
            data.drop(index=drop_result_ids, inplace=True)
            self.result_dicts = data.to_dict('records')

        # Get result files
        temp_dir = TemporaryDirectory()
        temp_dir_name = Path(temp_dir.name)

        base_names_dict = {}
        pred_txts = []

        if self.has_gt_anns:
            gt_dir = self.metadata.gt_dir[:-1]
            gt_imgs = []

        for image_id in self.image_ids:
            file_name = self.file_names[image_id]

            base_name = file_name.split('/')[-1]
            base_name = base_name.split('.')[0]
            base_name = base_name.split('_')

            if self.has_gt_anns:
                city_name = base_name[0]

            base_name = '_'.join(base_name[:3])
            base_names_dict[image_id] = base_name

            if self.has_gt_anns:
                gt_img = f'{gt_dir}/{city_name}/{base_name}_gtFine_instanceIds.png'
                gt_imgs.append(gt_img)

            pred_txt = f'{temp_dir_name}/{base_name}_pred.txt'
            open(pred_txt, 'w').close()
            pred_txts.append(pred_txt)

        for result_id, result_dict in enumerate(self.result_dicts):
            image_id = result_dict['image_id']
            label = result_dict['category_id']
            segm = result_dict['segmentation']
            score = result_dict['score']

            base_name = base_names_dict[image_id]
            pred_img = f'{base_name}_{label}_{result_id}.png'
            pred_txt = f'{base_name}_pred.txt'

            segm = rle_to_mask(segm)
            Image.fromarray(segm * 255).save(temp_dir_name / pred_img)

            with open(temp_dir_name / pred_txt, 'a') as pred_file:
                pred_file.write(f"{pred_img} {label} {score}\n")

        # Save result files if needed
        if output_dir is not None:
            if save_results or not self.has_gt_anns:
                res_files = [*glob.glob(str(temp_dir_name / '*.png')), *pred_txts]

                with ZipFile(output_dir / f'{save_name}.zip', 'w') as zip_file:
                    for res_file in res_files:
                        zip_file.write(res_file)

        # Clean up and return if no ground-truth annotations are available
        if not self.has_gt_anns:
            temp_dir.cleanup()
            return

        # Set arguments of Cityscapes evaluation API
        cityscapes_eval.args.colorized = False
        cityscapes_eval.args.gtInstancesFile = temp_dir_name / 'gtInstances.json'
        cityscapes_eval.args.JSONOutput = False
        cityscapes_eval.args.predictionPath = str(temp_dir_name)
        cityscapes_eval.args.predictionWalk = None
        cityscapes_eval.args.quiet = True

        # Get and print evaluation results for each metric
        eval_dict = {}

        for metric in self.metrics:
            cityscapes_eval.args.iou_type = metric
            results = cityscapes_eval.evaluateImgLists(pred_txts, gt_imgs, cityscapes_eval.args)['averages']
            eval_dict[metric] = list(results.values())

            print(f"Evaluation metric: {metric}")
            cityscapes_eval.printResults(results, cityscapes_eval.args)

        # Clean up temporary directory
        temp_dir.cleanup()

        return eval_dict


def build_cityscapes(args):
    """
    Build Cityscapes datasets and evaluator from command-line arguments.

    Args:
        args (argparse.Namespace): Command-line arguments.

    Returns:
        datasets (Dict): Dictionary of datasets potentially containing following keys:
            - train (CityscapesDataset): the training dataset (only present during training);
            - eval (CityscapesDataset): the evaluation dataset (always present).

        evaluator (CityscapesEvaluator): Object capable of computing evaluations from predictions and storing them.
    """

    # Get root directory containing datasets
    root = Path() / 'datasets'

    # Initialize empty datasets dictionary
    datasets = {}

    # Get training dataset if needed
    if not args.eval:
        image_dir = root / 'cityscapes/leftImg8bit/train'
        info_dir = root / 'cityscapes/gtFine/train'
        train_transforms = get_train_transforms(f'cityscapes_{args.train_transforms_type}')
        metadata = MetadataCatalog.get('cityscapes_fine_instance_seg_train')
        datasets['train'] = CityscapesDataset(image_dir, info_dir, train_transforms, metadata)

    # Get evaluation dataset
    image_dir = root / f'cityscapes/leftImg8bit/{args.eval_split}'
    info_dir = root / f'cityscapes/gtFine/{args.eval_split}'
    eval_transforms = get_eval_transforms(f'cityscapes_{args.eval_transforms_type}')
    metadata = MetadataCatalog.get(f'cityscapes_fine_instance_seg_{args.eval_split}')
    datasets['eval'] = CityscapesDataset(image_dir, info_dir, eval_transforms, metadata)

    # Get evaluator
    evaluator = CityscapesEvaluator(datasets['eval'], nms_thr=args.eval_nms_thr)

    return datasets, evaluator
