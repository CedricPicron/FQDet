"""
Objects365 dataset/evaluator and build function.
"""
import contextlib
import json
import os
from pathlib import Path
from zipfile import ZipFile

from boundary_iou.coco_instance_api.coco import COCO
from boundary_iou.coco_instance_api.cocoeval import COCOeval
from detectron2.data import MetadataCatalog
from detectron2.layers import batched_nms
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

from datasets.transforms import get_train_transforms, get_eval_transforms
from structures.boxes import Boxes
from structures.images import Images
import utils.distributed as distributed


# Add Objects365 entries to MetadataCatalog
thing_classes = (
    'Person', 'Sneakers', 'Chair', 'Other Shoes', 'Hat', 'Car', 'Lamp',
    'Glasses', 'Bottle', 'Desk', 'Cup', 'Street Lights', 'Cabinet/shelf',
    'Handbag/Satchel', 'Bracelet', 'Plate', 'Picture/Frame', 'Helmet',
    'Book', 'Gloves', 'Storage box', 'Boat', 'Leather Shoes', 'Flower',
    'Bench', 'Potted Plant', 'Bowl/Basin', 'Flag', 'Pillow', 'Boots',
    'Vase', 'Microphone', 'Necklace', 'Ring', 'SUV', 'Wine Glass', 'Belt',
    'Moniter/TV', 'Backpack', 'Umbrella', 'Traffic Light', 'Speaker',
    'Watch', 'Tie', 'Trash bin Can', 'Slippers', 'Bicycle', 'Stool',
    'Barrel/bucket', 'Van', 'Couch', 'Sandals', 'Bakset', 'Drum',
    'Pen/Pencil', 'Bus', 'Wild Bird', 'High Heels', 'Motorcycle',
    'Guitar', 'Carpet', 'Cell Phone', 'Bread', 'Camera', 'Canned',
    'Truck', 'Traffic cone', 'Cymbal', 'Lifesaver', 'Towel',
    'Stuffed Toy', 'Candle', 'Sailboat', 'Laptop', 'Awning', 'Bed',
    'Faucet', 'Tent', 'Horse', 'Mirror', 'Power outlet', 'Sink', 'Apple',
    'Air Conditioner', 'Knife', 'Hockey Stick', 'Paddle', 'Pickup Truck',
    'Fork', 'Traffic Sign', 'Ballon', 'Tripod', 'Dog', 'Spoon', 'Clock',
    'Pot', 'Cow', 'Cake', 'Dinning Table', 'Sheep', 'Hanger',
    'Blackboard/Whiteboard', 'Napkin', 'Other Fish', 'Orange/Tangerine',
    'Toiletry', 'Keyboard', 'Tomato', 'Lantern', 'Machinery Vehicle',
    'Fan', 'Green Vegetables', 'Banana', 'Baseball Glove', 'Airplane',
    'Mouse', 'Train', 'Pumpkin', 'Soccer', 'Skiboard', 'Luggage',
    'Nightstand', 'Tea pot', 'Telephone', 'Trolley', 'Head Phone',
    'Sports Car', 'Stop Sign', 'Dessert', 'Scooter', 'Stroller', 'Crane',
    'Remote', 'Refrigerator', 'Oven', 'Lemon', 'Duck', 'Baseball Bat',
    'Surveillance Camera', 'Cat', 'Jug', 'Broccoli', 'Piano', 'Pizza',
    'Elephant', 'Skateboard', 'Surfboard', 'Gun',
    'Skating and Skiing shoes', 'Gas stove', 'Donut', 'Bow Tie', 'Carrot',
    'Toilet', 'Kite', 'Strawberry', 'Other Balls', 'Shovel', 'Pepper',
    'Computer Box', 'Toilet Paper', 'Cleaning Products', 'Chopsticks',
    'Microwave', 'Pigeon', 'Baseball', 'Cutting/chopping Board',
    'Coffee Table', 'Side Table', 'Scissors', 'Marker', 'Pie', 'Ladder',
    'Snowboard', 'Cookies', 'Radiator', 'Fire Hydrant', 'Basketball',
    'Zebra', 'Grape', 'Giraffe', 'Potato', 'Sausage', 'Tricycle',
    'Violin', 'Egg', 'Fire Extinguisher', 'Candy', 'Fire Truck',
    'Billards', 'Converter', 'Bathtub', 'Wheelchair', 'Golf Club',
    'Briefcase', 'Cucumber', 'Cigar/Cigarette ', 'Paint Brush', 'Pear',
    'Heavy Truck', 'Hamburger', 'Extractor', 'Extention Cord', 'Tong',
    'Tennis Racket', 'Folder', 'American Football', 'earphone', 'Mask',
    'Kettle', 'Tennis', 'Ship', 'Swing', 'Coffee Machine', 'Slide',
    'Carriage', 'Onion', 'Green beans', 'Projector', 'Frisbee',
    'Washing Machine/Drying Machine', 'Chicken', 'Printer', 'Watermelon',
    'Saxophone', 'Tissue', 'Toothbrush', 'Ice cream', 'Hotair ballon',
    'Cello', 'French Fries', 'Scale', 'Trophy', 'Cabbage', 'Hot dog',
    'Blender', 'Peach', 'Rice', 'Wallet/Purse', 'Volleyball', 'Deer',
    'Goose', 'Tape', 'Tablet', 'Cosmetics', 'Trumpet', 'Pineapple',
    'Golf Ball', 'Ambulance', 'Parking meter', 'Mango', 'Key', 'Hurdle',
    'Fishing Rod', 'Medal', 'Flute', 'Brush', 'Penguin', 'Megaphone',
    'Corn', 'Lettuce', 'Garlic', 'Swan', 'Helicopter', 'Green Onion',
    'Sandwich', 'Nuts', 'Speed Limit Sign', 'Induction Cooker', 'Broom',
    'Trombone', 'Plum', 'Rickshaw', 'Goldfish', 'Kiwi fruit',
    'Router/modem', 'Poker Card', 'Toaster', 'Shrimp', 'Sushi', 'Cheese',
    'Notepaper', 'Cherry', 'Pliers', 'CD', 'Pasta', 'Hammer', 'Cue',
    'Avocado', 'Hamimelon', 'Flask', 'Mushroon', 'Screwdriver', 'Soap',
    'Recorder', 'Bear', 'Eggplant', 'Board Eraser', 'Coconut',
    'Tape Measur/ Ruler', 'Pig', 'Showerhead', 'Globe', 'Chips', 'Steak',
    'Crosswalk Sign', 'Stapler', 'Campel', 'Formula 1 ', 'Pomegranate',
    'Dishwasher', 'Crab', 'Hoverboard', 'Meat ball', 'Rice Cooker',
    'Tuba', 'Calculator', 'Papaya', 'Antelope', 'Parrot', 'Seal',
    'Buttefly', 'Dumbbell', 'Donkey', 'Lion', 'Urinal', 'Dolphin',
    'Electric Drill', 'Hair Dryer', 'Egg tart', 'Jellyfish', 'Treadmill',
    'Lighter', 'Grapefruit', 'Game board', 'Mop', 'Radish', 'Baozi',
    'Target', 'French', 'Spring Rolls', 'Monkey', 'Rabbit', 'Pencil Case',
    'Yak', 'Red Cabbage', 'Binoculars', 'Asparagus', 'Barbell', 'Scallop',
    'Noddles', 'Comb', 'Dumpling', 'Oyster', 'Table Tennis paddle',
    'Cosmetics Brush/Eyeliner Pencil', 'Chainsaw', 'Eraser', 'Lobster',
    'Durian', 'Okra', 'Lipstick', 'Cosmetics Mirror', 'Curling',
    'Table Tennis ',
)

for name in ('objects365_train', 'objects365_val'):
    MetadataCatalog.get(name).thing_classes = thing_classes

# Get tuple of missing images
MISSING_IMAGES = (
    'patch6/objects365_v1_00320532.jpg',
    'patch6/objects365_v1_00320534.jpg',
    'patch16/objects365_v2_00908726.jpg',
)


class Objects365Dataset(Dataset):
    """
    Class implementing the Objects365Dataset dataset.

    Attributes:
        image_dicts (List): List [num_images] of dictionaries containing information about the dataset images.
        ann_dicts (List): List [num_annotations] of dictionaries containing information about the dataset annotations.
        image_dir (Path): Path to directory with Objects365 images.
        transforms (List): List [num_transforms] of transforms applied to image and targets.
        metadata (detectron2.data.Metadata): Metadata instance containing additional dataset information.
    """

    def __init__(self, annotation_file, image_dir, transforms, metadata):
        """
        Initializes the Objects365Dataset dataset.

        Args:
            annotation_file (Path): Path to annotation file with Objects365 annotations.
            image_dir (Path): Path to directory with Objects365 images.
            transforms (List): List [num_transforms] of transforms applied to image and targets.
            metadata (detectron2.data.Metadata): Metadata instance containing additional dataset information.
        """

        # Process annotation file
        with open(annotation_file) as json_file:
            annotations = json.load(json_file)

        self.image_dicts = annotations['images']
        self.ann_dicts = annotations['annotations']
        img_id_dict = {}

        for contiguous_img_id, image_dict in enumerate(self.image_dicts):
            image_dict['ann_ids'] = []
            image_dict['file_name'] = image_dict['file_name'][10:]
            img_id_dict[image_dict['id']] = contiguous_img_id

        for ann_id, ann_dict in enumerate(self.ann_dicts):
            ann_dict['category_id'] = ann_dict['category_id'] - 1

            img_id = ann_dict['image_id']
            contiguous_img_id = img_id_dict[img_id]
            self.image_dicts[contiguous_img_id]['ann_ids'].append(ann_id)

        # Remove image dictionaries for which image file is missing
        self.image_dicts = [img_dict for img_dict in self.image_dicts if img_dict['file_name'] not in MISSING_IMAGES]

        # Set remaining attributes
        self.image_dir = image_dir
        self.transforms = transforms
        self.metadata = metadata

    def __getitem__(self, index):
        """
        Implements the __getitem__ method of the Objects365Dataset dataset.

        Args:
            index (int): Index selecting one of the dataset (image, transform) combinations.

        Returns:
            image (Images): Structure containing the image tensor after data augmentation.

            tgt_dict (Dict): Target dictionary potentially containing following keys (empty when no annotations):
                - labels (LongTensor): tensor of shape [num_targets] containing the class indices;
                - boxes (Boxes): structure containing axis-aligned bounding boxes of size [num_targets].
        """

        # Load image and place it into Images structure
        contiguous_img_id = index // len(self.transforms)
        image_dict = self.image_dicts[contiguous_img_id]

        image_path = self.image_dir / image_dict['file_name']
        image = Image.open(image_path).convert('RGB')

        img_id = image_dict['id']
        image = Images(image, img_id)

        # Load annotations and remove crowd annotations
        anns = [self.ann_dicts[ann_id] for ann_id in image_dict['ann_ids']]
        anns = [ann for ann in anns if ann.get('iscrowd', 0) == 0]

        # Get object class labels
        labels = [ann['category_id'] for ann in anns]
        labels = torch.tensor(labels, dtype=torch.int64)

        # Get object boxes in (left, top, right, bottom) format
        boxes = [ann['bbox'] for ann in anns]
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes = Boxes(boxes, format='xywh').to_format('xyxy')

        # Clip boxes and only keep targets with well-defined boxes
        boxes, well_defined = boxes.clip(image.size())
        labels = labels[well_defined]
        boxes = boxes[well_defined]

        # Get target dictionary with target labels and boxes
        tgt_dict = {'labels': labels, 'boxes': boxes}

        # Perform image and target dictionary transformations
        transform = self.transforms[index % len(self.transforms)]
        image, tgt_dict = transform(image, tgt_dict)

        # Only keep targets with well-defined boxes
        if 'boxes' in tgt_dict:
            well_defined = tgt_dict['boxes'].well_defined()

            for key in tgt_dict.keys():
                if key in ['labels', 'boxes']:
                    tgt_dict[key] = tgt_dict[key][well_defined]

        return image, tgt_dict

    def __len__(self):
        """
        Implements the __len__ method of the Objects365Dataset dataset.

        Returns:
            dataset_length (int): Dataset length measured as the number of images times the number of transforms.
        """

        # Get dataset length
        dataset_length = len(self.image_dicts) * len(self.transforms)

        return dataset_length


class Objects365Evaluator(object):
    """
    Class implementing the Objects365Evaluator evaluator.

    Attributes:
        coco_gt (COCO): COCO object containing the ground-truth Objects365 annotations.

        image_ids (List): List of evaluated image ids.
        metadata (detectron2.data.Metadata): Metadata instance containing additional dataset information.
        metrics (List): List with strings containing the evaluation metrics to be used.
        result_dicts (Dict): Dictionary of lists containing prediction results in COCO results format for each metric.

        eval_nms (bool): Boolean indicating whether to perform NMS during evaluation.
        nms_thr (float): IoU threshold used during evaluation NMS to remove duplicate detections.
    """

    def __init__(self, annotation_file, eval_dataset, metrics=None, nms_thr=0.5):
        """
        Initializes the Objects365Evaluator evaluator.

        Args:
            annotation_file (Path): Path to annotation file with Objects365 annotations.
            eval_dataset (Objects365Dataset): The evaluation dataset.
            metrics (List): List with strings containing the evaluation metrics to be used (default=None).
            nms_thr (float): IoU threshold used during evaluation NMS to remove duplicate detections (default=0.5).
        """

        # Get COCO object containing the ground-truth annotations
        with open(os.devnull, 'w') as devnull:
            with contextlib.redirect_stdout(devnull):
                self.coco_gt = COCO(annotation_file)

        # Set base attributes
        self.image_ids = []
        self.metadata = eval_dataset.metadata
        self.metrics = metrics if metrics is not None else []
        self.result_dicts = []

        # Set NMS attributes
        self.eval_nms = len(eval_dataset.transforms) > 1
        self.nms_thr = nms_thr

    def add_metrics(self, metrics):
        """
        Adds given evaluation metrics to the Objects365Evaluator evaluator.

        Args:
            metrics (List): List with strings containing the evaluation metrics to be added.
        """

        # Add evalutation metrics
        self.metrics.extend(metrics)

    def reset(self):
        """
        Resets the image_ids and result_dicts attributes of the Objects365Evaluator evaluator.
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
                - scores (FloatTensor): normalized prediction scores of shape [num_preds];
                - batch_ids (LongTensor): batch indices of predictions of shape [num_preds].
        """

        # Update the image_ids attribute
        self.image_ids.extend(images.image_ids)

        # Extract predictions from prediction dictionary
        labels = pred_dict['labels']
        boxes = pred_dict['boxes']
        scores = pred_dict['scores']
        batch_ids = pred_dict['batch_ids']

        # Transform boxes to original image space and convert them to desired format
        boxes, well_defined = boxes.transform(images, inverse=True)
        boxes = boxes[well_defined].to_format('xywh')

        labels = labels[well_defined]
        scores = scores[well_defined]
        batch_ids = batch_ids[well_defined]

        # Get image indices corresponding to predictions
        image_ids = torch.as_tensor(images.image_ids)
        image_ids = image_ids[batch_ids.cpu()]

        # Transform labels back to original space
        labels = labels + 1

        # Convert desired tensors to lists
        image_ids = image_ids.tolist()
        labels = labels.tolist()
        boxes = boxes.boxes.tolist()
        scores = scores.tolist()

        # Get result dictionary
        result_dict = {}
        result_dict['image_id'] = image_ids
        result_dict['category_id'] = labels
        result_dict['bbox'] = boxes
        result_dict['score'] = scores

        # Get list of result dictionaries
        result_dicts = pd.DataFrame(result_dict).to_dict(orient='records')

        # Make sure there is at least one result dictionary per image
        for image_id in images.image_ids:
            if image_id not in image_ids:
                result_dict = {}
                result_dict['image_id'] = image_id
                result_dict['category_id'] = 0
                result_dict['bbox'] = [0.0, 0.0, 0.0, 0.0]
                result_dict['score'] = 0.0
                result_dicts.append(result_dict)

        # Update result_dicts attribute
        self.result_dicts.extend(result_dicts)

    def evaluate(self, device='cpu', output_dir=None, save_results=False, save_name='results'):
        """
        Perform evaluation by finalizing the result dictionaries and by comparing with the ground-truth.

        Args:
            device (str): String containing the type of device used during NMS (default='cpu').
            output_dir (Path): Path to output directory to save result dictionaries (default=None).
            save_results (bool): Boolean indicating whether to save the results (default=False).
            save_name (str) String containing the name of the saved result file (default='results').

        Returns:
            eval_dict (Dict): Dictionary with evaluation results for each metric.
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
                label = result_dict['category_id']
                box = result_dict['bbox'].copy()
                score = result_dict['score']

                box[2] = box[0] + box[2]
                box[3] = box[1] + box[3]

                boxes[image_id].append(box)
                scores[image_id].append(score)
                labels[image_id].append(label)
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

        # Save result dictionaries if needed
        if output_dir is not None:
            if save_results:
                json_file_name = output_dir / f'{save_name}.json'
                zip_file_name = output_dir / f'{save_name}.zip'

                with open(json_file_name, 'w') as json_file:
                    json.dump(self.result_dicts, json_file)

                with ZipFile(zip_file_name, 'w') as zip_file:
                    zip_file.write(json_file_name, arcname=f'{save_name}.json')
                    os.remove(json_file_name)

        # Compare predictions with ground-truth annotations
        eval_dict = {}

        with open(os.devnull, 'w') as devnull:
            with contextlib.redirect_stdout(devnull):
                coco_res = self.coco_gt.loadRes(self.result_dicts)

        for metric in self.metrics:
            with open(os.devnull, 'w') as devnull:
                with contextlib.redirect_stdout(devnull):
                    sub_evaluator = COCOeval(self.coco_gt, coco_res, iouType=metric)
                    sub_evaluator.params.imgIds = self.image_ids
                    sub_evaluator.evaluate()
                    sub_evaluator.accumulate()

            print(f"Evaluation metric: {metric}")
            sub_evaluator.summarize()
            eval_dict[metric] = sub_evaluator.stats.tolist()

        return eval_dict


def build_objects365(args):
    """
    Build Objects365 datasets and evaluator from command-line arguments.

    Args:
        args (argparse.Namespace): Command-line arguments.

    Returns:
        datasets (Dict): Dictionary of datasets potentially containing following keys:
            - train (Objects365Dataset): the training dataset (only present during training);
            - eval (Objects365Dataset): the evaluation dataset (always present).

        evaluator (Objects365Evaluator): Object capable of computing evaluations from predictions and storing them.
    """

    # Get root directory containing datasets
    root = Path() / 'datasets'

    # Initialize empty datasets dictionary
    datasets = {}

    # Get training dataset if needed
    if not args.eval:
        annotation_file = root / 'objects365' / 'annotations' / 'train.json'
        image_dir = root / 'objects365' / 'train'
        train_transforms = get_train_transforms(f'objects365_{args.train_transforms_type}')
        metadata = MetadataCatalog.get(f'objects365_{args.train_split}')
        datasets['train'] = Objects365Dataset(annotation_file, image_dir, train_transforms, metadata)

    # Get evaluation dataset
    image_dir = root / 'objects365' / 'val'
    annotation_file = root / 'objects365' / 'annotations' / 'val.json'
    eval_transforms = get_eval_transforms(f'objects365_{args.eval_transforms_type}')
    metadata = MetadataCatalog.get(f'objects365_{args.eval_split}')
    datasets['eval'] = Objects365Dataset(annotation_file, image_dir, eval_transforms, metadata)

    # Get evaluator
    evaluator = Objects365Evaluator(annotation_file, datasets['eval'], nms_thr=args.eval_nms_thr)

    return datasets, evaluator
