"""
Distributed utilities.
"""
import builtins
import datetime
import os
import pickle

import torch


def init_distributed_mode(args):
    """
    Initializes distributed mode.
    """

    if 'RANK' not in os.environ or 'WORLD_SIZE' not in os.environ:
        print("Not using distributed mode.")
        args.distributed = False
        return

    args.distributed = True
    args.rank = int(os.environ['RANK'])
    args.world_size = int(os.environ['WORLD_SIZE'])

    args.gpu = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(args.gpu)

    kwargs = {'backend': 'nccl', 'init_method': args.dist_url, 'world_size': args.world_size, 'rank': args.rank}
    kwargs['timeout'] = datetime.timedelta(hours=1)
    torch.distributed.init_process_group(**kwargs)

    print(f"| distributed init (rank {args.rank}): {args.dist_url}", flush=True)
    set_master_only_printing(args.rank == 0)


def set_master_only_printing(is_master):
    """
    Disables printing when not in master process.

    Args:
        is_master (bool): Boolean indicating whether process is master process.
    """

    builtin_print = builtins.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    builtins.print = print


def is_dist_avail_and_initialized():
    """
    Checks whether the distributed package is available and whether the default process group is initialized.

    Returns:
        Boolean indicating distributed package availability and default process group initialization.
    """

    if not torch.distributed.is_available():
        return False
    if not torch.distributed.is_initialized():
        return False
    return True


def get_world_size():
    """
    Gets world size (i.e. size of process group).

    Returns:
        Integer containing the world size.
    """

    if not is_dist_avail_and_initialized():
        return 1
    return torch.distributed.get_world_size()


def get_rank():
    """
    Gets process rank (i.e. process rank within process group).

    Returns:
        Integer containing the process rank.
    """

    if not is_dist_avail_and_initialized():
        return 0
    return torch.distributed.get_rank()


def synchronize():
    """
    Synchronizes all processes from the process group.
    """

    if is_dist_avail_and_initialized():
        torch.distributed.barrier()


def is_main_process():
    """
    Checks whether process corresponds to main process of process group.

    Returns:
        Boolean indicating whether process is main process or not.
    """

    return get_rank() == 0


@torch.no_grad()
def reduce_dict(input_dict, average=True):
    """
    Reduce dictionary values across different processes.

    Args:
        input_dict (Dict): Input dictionary for which values will be reduced.
        average (bool): If true, average values, else simply take the sum.

    Returns:
        reduced_dict (Dict): Dictionary with reduced values for all input keys.
    """

    if not is_dist_avail_and_initialized() or not input_dict:
        return input_dict

    keys = sorted(input_dict.keys())
    values = torch.cat([input_dict[key].view(1) for key in keys], dim=0)
    torch.distributed.all_reduce(values)

    values = values/get_world_size() if average else values
    reduced_dict = {k: v for k, v in zip(keys, values)}

    return reduced_dict


def all_gather(input_data):
    """
    Gathers a list of arbitrary picklable data (not necessarily tensors) across different processes.

    Args:
        input_data (object): Any picklable object to be gathered across processes.

    Returns:
        gathered_data (List): List of data gathered from each process.
    """

    # Get world size
    world_size = get_world_size()

    # Return if world size is one
    if world_size == 1:
        return [input_data]

    # Transform serialized object to byte tensor
    buffer = pickle.dumps(input_data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to('cuda')

    # Obtain max tensor size
    local_size = torch.tensor([tensor.numel()], device='cuda')
    size_list = [torch.tensor([0], device='cuda') for _ in range(world_size)]
    torch.distributed.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # Pad tensors as only tensors of same shape can be gathered
    if local_size != max_size:
        padding = torch.empty(size=(max_size - local_size,), dtype=torch.uint8, device='cuda')
        tensor = torch.cat((tensor, padding), dim=0)

    # Gather padded tensors in tensor_list
    tensor_list = [torch.empty((max_size,), dtype=torch.uint8, device='cuda') for _ in size_list]
    torch.distributed.all_gather(tensor_list, tensor)

    # Post-processing with padding removal and reserialization
    gathered_data = []

    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        gathered_data.append(pickle.loads(buffer))

    return gathered_data
