"""
Collection of profiling utilities.
"""
from torch.nn.utils import clip_grad_norm_
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler
from torch.profiler import schedule as profiler_schedule


def profile_model(model, dataloader, optimizer=None, max_grad_norm=-1, num_samples=10, profile_train=True,
                  profile_inf=True, output_dir=None):
    """
    Function profiling the given model during both training and inference.

    Args:
        model (nn.Module): Module containing the model to be profiled.
        dataloader (torch.utils.data.Dataloader): Dataloader used to obtain the images and target dictionaries.
        optimizer (torch.optim.Optimizer): Optimizer used during training to update model parameters (default=None).
        max_grad_norm (float): Maximum gradient norm of parameters throughout model (default=-1).
        num_samples (int): Integer containing the number of batches sampled from dataloader (default=10).
        profile_train (bool): Boolean indicating whether to profile model in training mode (default=True).
        profile_inf (bool): Boolean indicating whether to profile model in inference mode (default=True).
        output_dir (Path): Path to directory to save the profiling results (default=None).

    Raises:
        ValueError: Error when no optimizer is provided when profiling model in training mode.
    """

    # Get model device
    device = next(model.parameters()).device

    # Create or clear profiling file if needed
    if output_dir is not None:
        open(output_dir / 'model_profiling.txt', 'w').close()

    # Get profiling activities
    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA] if 'cuda' in device.type else [ProfilerActivity.CPU]

    # Get profiling schedule
    warmup = active = 1
    schedule = profiler_schedule(wait=1, warmup=warmup, active=active, repeat=num_samples)

    # Get profiling trace handler
    trace_handler = tensorboard_trace_handler(output_dir)

    # Get profiling keyword arguments
    profile_kwargs = {'activities': activities, 'schedule': schedule, 'on_trace_ready': trace_handler}
    profile_kwargs = {**profile_kwargs, 'record_shapes': True, 'with_stack': True}

    # Profile model in training mode if requested
    if profile_train:

        # Print string indicating start of model profiling in training mode
        print("\n================================")
        print("|  Model profiling (training)  |")
        print("================================\n")

        # Check whether optimizer is provided
        if optimizer is None:
            error_msg = "An optimizer must be provided when profiling model in training mode, but is missing."
            raise ValueError(error_msg)

        # Set model in training mode
        model.train()

        # Set optimizer learning rates equal to zero
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.0

        # Start profiling
        with profile(**profile_kwargs) as prof:

            # Iterate over dataloader
            for i, (images, tgt_dict) in enumerate(dataloader, 1):

                # Place images and target dictionary on correct device
                images = images.to(device)
                tgt_dict = {k: v.to(device) for k, v in tgt_dict.items()}

                # Get tuple of model inputs
                inputs = (images, tgt_dict, optimizer, max_grad_norm)
                prof.step()

                # Profile model
                for _ in range(warmup + active):
                    loss_dict = model(images, tgt_dict)[0]

                    optimizer.zero_grad(set_to_none=True)
                    loss = sum(loss_dict.values())
                    loss.backward()

                    if max_grad_norm > 0:
                        clip_grad_norm_(model.parameters(), max_grad_norm)

                    optimizer.step()
                    prof.step()

                # Break when 'num_samples' batches are processed
                if i == num_samples:
                    break

    # Profile model in inference mode if requested
    if profile_inf:

        # Print string indicating start of model profiling in inference mode
        print("\n=================================")
        print("|  Model profiling (inference)  |")
        print("=================================\n")

        # Set model in evaluation mode
        model.eval()

        # Start profiling
        with profile(**profile_kwargs) as prof:

            # Iterate over dataloader
            for i, (images, _) in enumerate(dataloader, 1):

                # Place images on correct device
                images = images.to(device)

                # Get tuple of model inputs
                inputs = (images,)
                prof.step()

                # Profile model
                for _ in range(warmup + active):
                    model(*inputs)
                    prof.step()

                # Break when 'num_samples' batches are processed
                if i == num_samples:
                    break
