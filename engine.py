"""
Collection of training, evaluation and saving functions.
"""
from copy import deepcopy
import json
import math
from pathlib import Path
import sys

from PIL import Image
import torch
from torch.nn.utils import clip_grad_norm_

from utils.logging import MetricLogger
import utils.distributed as distributed


def train(model, dataloader, optimizer, epoch, max_grad_norm=-1, print_freq=10):
    """
    Trains model for one epoch on data from the given dataloader.

    Args:
        model (nn.Module): Module computing predictions from images.
        dataloader (torch.utils.data.Dataloader): Training dataloader.
        optimizer (torch.optim.Optimizer): Optimizer updating the model parameters during training.
        epoch (int): Integer containing the current training epoch.
        max_grad_norm (float): Maximum gradient norm of parameters throughout model (default=-1).
        print_freq (int): Integer containing the logger print frequency (default=10).

    Returns:
        train_stats (Dict): Dictionary containing the epoch training statistics.
    """

    # Get device and set model in training mode
    device = next(model.parameters()).device
    model.train()

    # Initialize metric logger
    metric_logger = MetricLogger(delimiter="  ")
    header = f"Train epoch {epoch}:"

    # Iterate over training images
    for images, tgt_dict in metric_logger.log_every(dataloader, print_freq, header):

        # Place images and target dictionary on correct device
        images = images.to(device)
        tgt_dict = {k: v.to(device) for k, v in tgt_dict.items()}

        # Get loss and analysis dictionaries
        loss_dict, analysis_dict = model(images, tgt_dict)

        # Update model parameters
        optimizer.zero_grad(set_to_none=True)

        loss = sum(loss_dict.values())
        loss.backward()

        if max_grad_norm > 0:
            clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()

        # Average analysis and loss dictionaries over all GPUs for logging purposes
        analysis_dict = distributed.reduce_dict(analysis_dict)
        loss_dict = distributed.reduce_dict(loss_dict)
        loss = sum(loss_dict.values()).item()

        # Check whether loss is finite
        if not math.isfinite(loss):
            print(f"Loss dictionary: {loss_dict}")
            print(f"Loss is {loss}, stopping training.")
            sys.exit(1)

        # Update logger
        metric_logger.update(**analysis_dict, **loss_dict, loss=loss)

    # Get epoch training statistics
    metric_logger.synchronize_between_processes()
    train_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    # Add epoch learning rates of different parameter groups to training statistics
    for i, param_group_name in enumerate(optimizer.param_group_names):
        train_stats[f'lr_{param_group_name}'] = optimizer.param_groups[i]['lr']

    return train_stats


@torch.no_grad()
def evaluate(model, dataloader, evaluator=None, eval_with_bnd=False, epoch=None, output_dir=None, print_freq=10,
             save_stats=False, save_results=False, save_tag='single_scale', visualize=False, vis_score_thr=0.4):
    """
    Evaluates model on data from given dataloader. It additionally computes visualizations if 'visualize' is set.

    Args:
        model (nn.Module): Module to be evaluated computing predictions from images.
        dataloader (torch.utils.data.Dataloader): Evaluation dataloader.
        evaluator (object): Object computing evaluations from predictions and storing them (default=None).
        eval_with_bnd (bool): Boolean indicating whether to evaluate segmentations with boundary IoU (default=False).
        epoch (int): Integer containing the number of training epochs completed (default=None).
        output_dir (Path): Path to directory to save evaluations and potentially visualizations (default=None).
        print_freq (int): Integer containing the logger print frequency (default=10).
        save_stats (bool): Boolean indicating whether to save evaluation statistics (default=False).
        save_results (bool): Boolean indicating whether to save results (default=False).
        save_tag (str): String containing tag used at the end of evaluation file names (default='single_scale').
        visualize (bool): Boolean indicating whether visualizations should be computed (default=False).
        vis_score_thr (float): Threshold indicating the minimum score for a detection to be drawn (default=0.4).

    Returns:
        eval_stats (Dict): Dictionary containing the evaluation statistics.
    """

    # Get device, set model in evaluation mode and initialize evaluators
    device = next(model.parameters()).device
    model.eval()
    evaluators = None

    # Get one evaluator per prediction dictionary
    if evaluator is not None:
        sample_images = next(iter(dataloader))[0]
        pred_dicts = model(sample_images.to(device))[0]
        evaluators = []

        for pred_dict in pred_dicts:
            evaluator_i = deepcopy(evaluator)
            evaluator_i.metrics = []

            if 'masks' in pred_dict:
                metrics = ['bbox', 'segm', 'boundary'] if eval_with_bnd else ['bbox', 'segm']
                evaluator_i.add_metrics(metrics)

            elif 'boxes' in pred_dict:
                evaluator_i.add_metrics(['bbox'])

            if len(evaluator_i.metrics) > 0:
                evaluator_i.reset()
            else:
                evaluator_i = None

            evaluators.append(evaluator_i)

    # Make visualization directory within output directory if needed
    if visualize and output_dir is not None:
        vis_dir = output_dir / 'visualization'
        vis_dir.mkdir(exist_ok=True)

    # Get fixed model keyword arguments
    model_kwargs = {'extended_analysis': True, 'visualize': visualize, 'vis_score_thr': vis_score_thr}

    # Initialize metric logger
    window_size = 1 if visualize else 20
    metric_logger = MetricLogger(delimiter="  ", window_size=window_size)
    header = "Evaluation:" if epoch is None else f"Eval epoch {epoch}:"

    # Iterate over evaluation images
    for images, tgt_dict in metric_logger.log_every(dataloader, print_freq, header):

        # Place images and target dictionary on correct device
        images = images.to(device)
        tgt_dict = {k: v.to(device) for k, v in tgt_dict.items()}

        # Get prediction, loss and analysis dictionaries
        tgt_dict = tgt_dict if tgt_dict else None
        output_dicts = model(images, tgt_dict=tgt_dict, **model_kwargs)

        if tgt_dict is not None:
            pred_dicts, loss_dict, analysis_dict = output_dicts[:3]
        else:
            pred_dicts, analysis_dict = output_dicts[:2]
            loss_dict = {}

        # Average analysis and loss dictionaries over all GPUs for logging purposes
        analysis_dict = distributed.reduce_dict(analysis_dict)
        loss_dict = distributed.reduce_dict(loss_dict) if loss_dict else loss_dict

        # Update logger
        metric_logger.update(**analysis_dict, **loss_dict)

        if loss_dict:
            loss = sum(loss_dict.values()).item()
            metric_logger.update(loss=loss)

        # Update evaluators
        if evaluators is not None:
            for evaluator, pred_dict in zip(evaluators, pred_dicts):
                if evaluator is not None:
                    evaluator.update(images, pred_dict)

        # Save visualizations to visualization directory
        if visualize and output_dir is not None:
            images_dict = output_dicts[-1]

            for key, image in images_dict.items():
                key_parts = key.split('_')
                image_id = images.image_ids[int(key_parts[-1])]

                filename = ('_').join([str(image_id), *key_parts[:-1]])
                Image.fromarray(image).save(f'{vis_dir / filename}.png')

    # Get epoch evaluation statistics
    metric_logger.synchronize_between_processes()
    eval_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    # Perform evaluations
    if evaluators is not None:
        for i, evaluator in enumerate(evaluators, 1):

            if evaluator is not None:
                eval_kwargs = {'device': device, 'output_dir': output_dir, 'save_results': save_results}
                eval_kwargs['save_name'] = f'results_{i}_{save_tag}'
                eval_dict = evaluator.evaluate(**eval_kwargs)

                if eval_dict is not None:
                    for metric in eval_dict.keys():
                        eval_stats[f'eval_{i}_{metric}'] = eval_dict[metric]

    # Save evaluations to output directory
    if distributed.is_main_process() and output_dir is not None and save_stats:
        with (output_dir / f'eval_{save_tag}.txt').open('w') as eval_file:
            eval_file.write(json.dumps(eval_stats) + "\n")

    return eval_stats


def save_checkpoint(args, epoch, model, optimizer, scheduler):
    """
    Function used for checkpoint saving.

    No checkpoints are saved when args.output_dir is an empty string.

    Args:
        args (argparse.Namespace): Command-line arguments.
        epoch (int): Training epochs completed.
        model (nn.Module): Model module to be saved.
        optimizer (torch.optim.Optimizer): Optimizer to be saved.
        scheduler (torch.optim.lr_scheduler): Scheduler to be saved.
    """

    if args.output_dir and distributed.is_main_process():
        output_dir = Path(args.output_dir)
        checkpoint_paths = [output_dir / 'checkpoint.pth']

        # Extra checkpoint before LR drop
        if epoch in args.lr_drops:
            checkpoint_paths.append(output_dir / f'checkpoint{epoch}.pth')

        # Checkpoint saving
        for checkpoint_path in checkpoint_paths:
            checkpoint = {'args': args, 'epoch': epoch}
            checkpoint['model'] = model.state_dict()
            checkpoint['optimizer'] = optimizer.state_dict()
            checkpoint['scheduler'] = scheduler.state_dict()
            torch.save(checkpoint, checkpoint_path)


def save_log(output_dir, epoch, train_stats, eval_stats=None):
    """
    Function used for log saving.

    No logs are saved when output_dir is an empty string.

    Args:
        output_dir (str): String containing the path to the output directory used for saving.
        epoch (int): Training epochs completed.
        train_stats (Dict): Dictionary containing the training statistics.
        eval_stats (Dict): Dictionary containing the evaluation statistics (default=None).
    """

    if output_dir and distributed.is_main_process():
        output_dir = Path(output_dir)

        log_dict = {'epoch': epoch}
        log_dict.update({f'train_{k}': v for k, v in train_stats.items()})

        if eval_stats is not None:
            log_dict.update({f'eval_{k}': v for k, v in eval_stats.items()})

        with (output_dir / 'log.txt').open('a') as log_file:
            log_file.write(json.dumps(log_dict) + "\n")
