"""
General build function for core modules.
"""
from collections import OrderedDict

from mmengine.config import Config

from .gc import GC
from .mmdet import MMDetCore
from models.build import build_model
from models.modules.container import Sequential


def build_core(args):
    """
    Build core module from command-line arguments.

    Args:
        args (argparse.Namespace): Command-line arguments.

    Returns:
        core (nn.Module): The specified core module.

    Raises:
        ValueError: Error when an unknown core type was provided.
    """

    # Initialize empty dictionary of core modules
    cores = OrderedDict()

    # Get input indices and input sizes
    in_ids = args.backbone_out_ids
    in_sizes = args.backbone_out_sizes

    # Initialize configuration index
    cfg_id = 0

    # Build core modules
    for core_type in args.cores:

        if core_type == 'cfg':
            core_cfg_path = args.core_cfg_paths[cfg_id]
            core_cfg = Config.fromfile(core_cfg_path)
            cfg_id += 1

            core_name = list(core_cfg.keys())[0]
            core_cfg = getattr(core_cfg, core_name)

            core_out_ids = core_cfg.pop('out_ids')
            core_out_sizes = core_cfg.pop('out_sizes')

            core = build_model(core_cfg)
            cores[core_name] = core

            core.out_ids = core_out_ids
            core.out_sizes = core_out_sizes

        elif core_type == 'gc':
            core_ids = args.core_ids
            yaml_file = args.gc_yaml

            core = GC(in_ids, in_sizes, core_ids, yaml_file)
            cores[core_type] = core

        elif core_type == 'mmdet':
            core = MMDetCore(args.mmdet_core_cfg_path)
            cores[core_type] = core

        else:
            raise ValueError(f"Unknown core type '{core_type}' was provided.")

        # Update input indices and input sizes
        in_ids = core.out_ids
        in_sizes = core.out_sizes

    # Add core output indices and output feature sizes to args
    args.core_out_ids = core.out_ids
    args.core_out_sizes = core.out_sizes

    # Get final core module from dictionary of core modules
    core = list(cores.values())[0] if len(cores) == 1 else Sequential(cores)

    return core
