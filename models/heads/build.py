"""
General build function for heads.
"""

from mmengine.config import Config

from models.build import build_model


def build_heads(args):
    """
    Build head modules from command-line arguments.

    Args:
        args (argparse.Namespace): Command-line arguments.

    Returns:
        heads (Dict): Dictionary with specified head modules.

    Raises:
        ValueError: Error when an unknown head type was provided.
    """

    # Initialize empty dictionary of head modules
    heads = {}

    # Initialize 'requires_masks' command-line argument
    args.requires_masks = False

    # Initialize configuration index
    cfg_id = 0

    # Build head modules
    for head_type in args.heads:

        if head_type == 'cfg':
            head_cfg_path = args.head_cfg_paths[cfg_id]
            head_cfg = Config.fromfile(head_cfg_path)

            head_name = head_cfg.model.pop('name', str(cfg_id))
            args.requires_masks |= head_cfg.model.pop('requires_masks', False)

            heads[head_name] = build_model(head_cfg.model, metadata=args.metadata)
            cfg_id += 1

        else:
            raise ValueError(f"Unknown head type '{head_type}' was provided.")

    return heads
