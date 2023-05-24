"""
General build function for architecture modules.
"""

from .bch import BCH
from .mmdet import MMDetArch

from models.backbones.build import build_backbone
from models.cores.build import build_core
from models.heads.build import build_heads


def build_arch(args):
    """
    Build architecture module from command-line arguments.

    Args:
        args (argparse.Namespace): Command-line arguments.

    Returns:
        arch (nn.Module): The specified architecture module.

    Raises:
        ValueError: Error when unknown architecture type was provided.
    """

    # Build architecture module
    if args.arch_type == 'bch':
        backbone = build_backbone(args)
        core = build_core(args)
        heads = build_heads(args)
        arch = BCH(backbone, core, heads)

    elif args.arch_type == 'mmdet':
        backbone = build_backbone(args)
        core = build_core(args)
        arch = MMDetArch(args.mmdet_arch_cfg_path, backbone, core)
        args.requires_masks = arch.requires_masks

    else:
        raise ValueError(f"Unknown architecture type {args.arch_type} was provided.")

    return arch
