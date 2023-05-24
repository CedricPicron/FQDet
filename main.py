"""
Main program script.
"""
import argparse
import datetime
from pathlib import Path
import time

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler, SequentialSampler

from datasets.build import build_dataset
from engine import evaluate, save_checkpoint, save_log, train
from models.archs.build import build_arch
from utils.analysis import analyze_model
from utils.comparison import compare_models
from utils.data import collate_fn, SubsetSampler
import utils.distributed as distributed
from utils.profiling import profile_model


def get_parser():
    parser = argparse.ArgumentParser(add_help=False)

    # General
    parser.add_argument('--device', default='cuda', type=str, help='name of device to use')
    parser.add_argument('--checkpoint_full', default='', type=str, help='path to checkpoint to fully resume from')
    parser.add_argument('--checkpoint_part', default='', type=str, help='path to checkpoint to partly resume from')
    parser.add_argument('--disable_strict_loading', action='store_true', help='disable strict model loading')
    parser.add_argument('--output_dir', default='', type=str, help='path to output directory')

    # Distributed
    parser.add_argument('--dist_url', default='env://', type=str, help='url used to set up distributed training')

    # Dataset
    parser.add_argument('--dataset', default='coco', type=str, help='name of dataset')
    parser.add_argument('--train_split', default='2017_train', type=str, help='name of the training split')
    parser.add_argument('--eval_split', default='2017_val', type=str, help='name of the evaluation split')
    parser.add_argument('--eval_nms_thr', default=0.5, type=float, help='IoU threshold during evaluation NMS')

    # Transforms
    parser.add_argument('--train_transforms_type', default='multi_scale', type=str, help='training transforms type')
    parser.add_argument('--eval_transforms_type', default='single_scale', type=str, help='evaluation transforms type')

    # Data loading
    parser.add_argument('--batch_size', default=2, type=int, help='batch size per device')
    parser.add_argument('--num_workers', default=2, type=int, help='number of subprocesses to use for data loading')

    # Training
    parser.add_argument('--eval_freq', default=1, type=int, help='evaluation frequency during training')

    # Evaluation
    parser.add_argument('--eval', action='store_true', help='perform evaluation task instead of training')
    parser.add_argument('--eval_task', default='performance', type=str, help='name of the evaluation task')
    parser.add_argument('--eval_with_bnd', action='store_true', help='also evaluate segmentations with boundary IoU')

    # * Analysis
    parser.add_argument('--analysis_samples', default=100, type=int, help='number of samples used for model analysis')
    parser.add_argument('--analysis_train_only', action='store_true', help='only analyze model in training mode')
    parser.add_argument('--analysis_inf_only', action='store_true', help='only analyze model in inference mode')
    parser.add_argument('--analysis_save', action='store_true', help='whether to save analysis results')

    # * Comparison
    parser.add_argument('--comp_res_file1', default='', type=str, help='path to first result file to be compared')
    parser.add_argument('--comp_res_file2', default='', type=str, help='path to second result file to be compared')
    parser.add_argument('--comp_save_name', default='', type=str, help='name of file saving comparison results')

    # * Performance
    parser.add_argument('--perf_save_res', action='store_true', help='save results even when having annotations')
    parser.add_argument('--perf_save_tag', default='single_scale', type=str, help='tag used in evaluation file names')
    parser.add_argument('--perf_with_vis', action='store_true', help='also gather visualizations during evaluation')

    # * Profiling
    parser.add_argument('--profile_samples', default=10, type=int, help='number of samples used for model profiling')
    parser.add_argument('--profile_train_only', action='store_true', help='only profile model in training mode')
    parser.add_argument('--profile_inf_only', action='store_true', help='only profile model in inference mode')

    # * Visualization
    parser.add_argument('--num_images', default=10, type=int, help='number of images to be visualized')
    parser.add_argument('--image_offset', default=0, type=int, help='image id of first image to be visualized')
    parser.add_argument('--random_offset', action='store_true', help='generate random image offset')
    parser.add_argument('--vis_score_thr', default=0.4, type=float, help='score threshold used during visualization')

    # Architecture
    parser.add_argument('--arch_type', default='bch', type=str, help='type of architecture module')

    # * MMDetection architecture
    parser.add_argument('--mmdet_arch_cfg_path', default='', type=str, help='path to MMDetection architecture config')

    # Backbone
    parser.add_argument('--backbone_type', default='resnet', type=str, help='type of backbone module')
    parser.add_argument('--backbone_cfg_path', default='', type=str, help='path to the backbone config')

    # * MMDetection backbone
    parser.add_argument('--mmdet_backbone_cfg_path', default='', type=str, help='path to MMDetection backbone config')

    # * ResNet
    parser.add_argument('--resnet_name', default='resnet50', type=str, help='full name of the desired ResNet model')
    parser.add_argument('--resnet_out_ids', nargs='*', default=[3, 4, 5], type=int, help='ResNet output map indices')
    parser.add_argument('--resnet_dilation', action='store_true', help='whether to use dilation for last ResNet layer')

    # Core
    parser.add_argument('--cores', nargs='*', default='', type=str, help='names of desired cores')
    parser.add_argument('--core_cfg_paths', nargs='*', default='', type=str, help='paths to core configs')
    parser.add_argument('--core_ids', nargs='*', default=[3, 4, 5, 6, 7], type=int, help='core feature map indices')

    # * GC (Generalized Core)
    parser.add_argument('--gc_yaml', default='', type=str, help='path to yaml-file with GC specification')

    # * MMDetection core
    parser.add_argument('--mmdet_core_cfg_path', default='', type=str, help='path to MMDetection core config')

    # Heads
    parser.add_argument('--heads', nargs='*', default='', type=str, help='names of desired heads')
    parser.add_argument('--head_cfg_paths', nargs='*', default='', type=str, help='paths to head configs')

    # Optimizer
    parser.add_argument('--max_grad_norm', default=-1, type=float, help='maximum gradient norm during training')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='L2 weight decay coefficient')

    # * Learning rates (General)
    parser.add_argument('--lr_default', default=1e-4, type=float, help='default learning rate')
    parser.add_argument('--lr_backbone', default=1e-5, type=float, help='backbone learning rate')

    # * Learning rates (BCH)
    parser.add_argument('--lr_core', default=1e-4, type=float, help='core learning rate')
    parser.add_argument('--lr_heads', default=1e-4, type=float, help='heads learning rate')

    # * Learning rates (Deformable DETR)
    parser.add_argument('--lr_reference_points', default=1e-5, type=float, help='reference points learning rate')

    # * Learning rates (MMDetArch)
    parser.add_argument('--lr_neck', default=1e-4, type=float, help='neck learning rate')

    # * Learning rates (MSDA)
    parser.add_argument('--lr_offset', default=1e-5, type=float, help='learning rate of deformable offsets')

    # Scheduler
    parser.add_argument('--epochs', default=12, type=int, help='total number of training epochs')
    parser.add_argument('--lr_drops', nargs='*', default=[9], type=int, help='epochs of learning rate drops')

    return parser


def main(args):
    """
    Function containing the main program script.

    Args:
        args (argparse.Namespace): Command-line arguments.

    Raises:
        ValueError: Error when in distributed mode for the 'analysis' evaluation task.
        ValueError: Error when in distributed mode for the 'profile' evaluation task.
        ValueError: Error when no output directory could be determined for the 'visualize' evaluation task.
        ValueError: Error when in distributed mode for the 'visualize' evaluation task.
        ValueError: Error when an unknown evaluation task is provided.
    """

    # Compare two models if requested and return
    if args.eval_task == 'comparison':
        output_dir = Path(('/').join(args.comp_res_file1.split('/')[:-1]))
        save_name = args.comp_save_name
        compare_models(args.dataset, args.eval_split, args.comp_res_file1, args.comp_res_file2, output_dir, save_name)
        return

    # Initialize distributed mode if needed
    distributed.init_distributed_mode(args)
    print(args)

    # Get datasets and evaluator
    datasets, evaluator = build_dataset(args)

    # Build model and place it on correct device
    device = torch.device(args.device)
    model = build_arch(args)
    model = model.to(device)

    # Try loading checkpoint
    checkpoint_path = ''

    if args.checkpoint_full:
        try:
            checkpoint = torch.load(args.checkpoint_full, map_location='cpu')
            checkpoint_path = args.checkpoint_full
            change_scheduler = False
        except FileNotFoundError:
            pass

    if not checkpoint_path and args.checkpoint_part:
        checkpoint = torch.load(args.checkpoint_part, map_location='cpu')
        checkpoint_path = args.checkpoint_part
        change_scheduler = True

    # Load model from checkpoint if present
    if checkpoint_path:
        strict = not args.disable_strict_loading
        model.load_state_dict(checkpoint['model'], strict=strict)

    # Wrap model into DistributedDataParallel (DDP) if needed
    if args.distributed:
        model = DistributedDataParallel(model, device_ids=[args.gpu])

    # Set 'requires_masks' attribute of datasets
    for dataset in datasets.values():
        dataset.requires_masks = args.requires_masks

    # Get shared dataloader keyword arguments
    dataloader_kwargs = {'collate_fn': collate_fn, 'num_workers': args.num_workers, 'pin_memory': True}

    # Get training dataloader if needed
    if not args.eval:
        train_dataset = datasets['train']
        train_sampler = DistributedSampler(train_dataset) if args.distributed else RandomSampler(train_dataset)
        train_batch_sampler = torch.utils.data.BatchSampler(train_sampler, args.batch_size, drop_last=True)
        train_dataloader = DataLoader(train_dataset, batch_sampler=train_batch_sampler, **dataloader_kwargs)

    # Get evaluation dataloader
    eval_dataset = datasets['eval']

    if args.eval and args.eval_task == 'visualize':
        eval_sampler = SubsetSampler(eval_dataset, args.num_images, args.image_offset, args.random_offset)
    elif args.distributed:
        eval_sampler = DistributedSampler(eval_dataset, shuffle=False)
    else:
        eval_sampler = SequentialSampler(eval_dataset)

    eval_dataloader = DataLoader(eval_dataset, args.batch_size, sampler=eval_sampler, **dataloader_kwargs)

    # Get default optimizer and scheduler
    param_families = model.module.get_param_families() if args.distributed else model.get_param_families()
    param_families = ['offset', 'reference_points', *param_families, 'default']
    param_dicts = {family: {'params': [], 'lr': getattr(args, f'lr_{family}')} for family in param_families}

    for param_name, parameter in model.named_parameters():
        if parameter.requires_grad:
            for family_name in param_families:
                if family_name in param_name or family_name == 'default':
                    param_dicts[family_name]['params'].append(parameter)
                    break

    for family_name in param_families:
        if len(param_dicts[family_name]['params']) == 0:
            param_dicts.pop(family_name)

        elif param_dicts[family_name]['lr'] <= 0:
            for parameter in param_dicts[family_name]['params']:
                parameter.requires_grad_(False)

            param_dicts.pop(family_name)

    optimizer = torch.optim.AdamW(param_dicts.values(), weight_decay=args.weight_decay)
    optimizer.param_group_names = list(param_dicts.keys())
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.lr_drops)

    # Update default optimizer and/or scheduler based on checkpoint
    if checkpoint_path:
        if change_scheduler:
            for param_group in checkpoint['optimizer']['param_groups']:
                param_group['lr'] = param_group['initial_lr']

            optimizer.load_state_dict(checkpoint['optimizer'])
            optimizer.step()

            for _ in range(checkpoint['epoch']):
                scheduler.step()

        else:
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])

    # Get output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif checkpoint_path:
        output_dir = Path(checkpoint_path).parent
    else:
        output_dir = None

    # Perform evaluation task if requested
    if args.eval:

        # Perform model analysis and return
        if args.eval_task == 'analysis':

            if args.distributed:
                error_msg = "Distributed mode is not supported for the 'analysis' evaluation task."
                raise ValueError(error_msg)

            analyze_train = args.analysis_train_only or not args.analysis_inf_only
            analyze_inf = args.analysis_inf_only or not args.analysis_train_only
            output_dir = output_dir if args.analysis_save else None

            analysis_kwargs = {'num_samples': args.analysis_samples, 'analyze_train': analyze_train}
            analysis_kwargs = {**analysis_kwargs, 'analyze_inf': analyze_inf, 'output_dir': output_dir}
            analyze_model(model, eval_dataloader, optimizer, max_grad_norm=args.max_grad_norm, **analysis_kwargs)
            return

        # Evaluate model performance and return
        elif args.eval_task == 'performance':
            perf_kwargs = {'eval_with_bnd': args.eval_with_bnd, 'save_stats': True, 'save_results': args.perf_save_res}
            perf_kwargs = {**perf_kwargs, 'save_tag': f'{args.eval_split}_{args.perf_save_tag}'}
            perf_kwargs = {**perf_kwargs, 'visualize': args.perf_with_vis, 'vis_score_thr': args.vis_score_thr}
            evaluate(model, eval_dataloader, evaluator=evaluator, output_dir=output_dir, **perf_kwargs)
            return

        # Perform model profiling and return
        if args.eval_task == 'profile':

            if args.distributed:
                error_msg = "Distributed mode is not supported for the 'profile' evaluation task."
                raise ValueError(error_msg)

            profile_train = args.profile_train_only or not args.profile_inf_only
            profile_inf = args.profile_inf_only or not args.profile_train_only

            profile_kwargs = {'num_samples': args.profile_samples, 'profile_train': profile_train}
            profile_kwargs = {**profile_kwargs, 'profile_inf': profile_inf, 'output_dir': output_dir}
            profile_model(model, eval_dataloader, optimizer, max_grad_norm=args.max_grad_norm, **profile_kwargs)
            return

        # Visualize model predictions and return
        elif args.eval_task == 'visualize':
            if output_dir is None:
                error_msg = "The 'visualize' evaluation task requires an output directory, but no output directory "
                error_msg += "was given or could be derived from checkpoint."
                raise ValueError(error_msg)

            if args.distributed:
                error_msg = "Distributed mode is not supported for the 'visualize' evaluation task."
                raise ValueError(error_msg)

            if args.batch_size > 1:
                msg = "It's recommended to use 'batch_size=1' for the 'visualize' evaluation task, so that the printed "
                msg += "loss and analysis dictionaries are image-specific."
                print(msg)

            vis_kwargs = {'print_freq': 1, 'visualize': True, 'vis_score_thr': args.vis_score_thr}
            evaluate(model, eval_dataloader, output_dir=output_dir, **vis_kwargs)
            return

        # Raise error
        else:
            error_msg = f"An unknown evaluation task was provided (got '{args.eval_task}')."
            raise ValueError(error_msg)

    # Get start epoch
    start_epoch = checkpoint['epoch']+1 if checkpoint_path else 1

    # Start training timer
    start_time = time.time()

    # Main training loop
    for epoch in range(start_epoch, args.epochs+1):
        train_sampler.set_epoch(epoch) if args.distributed else None
        train_stats = train(model, train_dataloader, optimizer, epoch, max_grad_norm=args.max_grad_norm)
        scheduler.step()

        checkpoint_model = model.module if args.distributed else model
        save_checkpoint(args, epoch, checkpoint_model, optimizer, scheduler)

        if epoch % args.eval_freq == 0:
            eval_stats = evaluate(model, eval_dataloader, evaluator=evaluator, epoch=epoch)
        else:
            eval_stats = None

        save_log(args.output_dir, epoch, train_stats, eval_stats)
        distributed.synchronize()

    # End training timer and report total training time
    total_time = time.time() - start_time
    total_time = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training {args.epochs+1 - start_epoch} epochs finished after {total_time}")


if __name__ == '__main__':
    prog = "python main.py"
    description = "Main training and evaluation script"
    fmt = argparse.MetavarTypeHelpFormatter

    parser = argparse.ArgumentParser(prog=prog, description=description, parents=[get_parser()], formatter_class=fmt)
    main(parser.parse_args())
