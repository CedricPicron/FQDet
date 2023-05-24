"""
Collection of loss functions.
"""


def sigmoid_dice_loss(pred_logits, tgt_labels, reduction='mean'):
    """
    Computes the DICE loss between the prediction logits and the corresponding binary target labels.

    The sigmoid function is used to compute the prediction probabilities from the prediction logits.

    Args:
        pred_logits (FloatTensor): Prediction logits of shape [num_groups, group_size].
        tgt_labels (FloatTensor): Binary target labels of shape [num_groups, group_size].
        reduction (str): String specifying the reduction operation applied on group-wise losses (default='mean').

    Returns:
        * If reduction is 'none':
            losses (FloatTensor): Tensor with group-wise DICE losses of shape [num_groups].

        * If reduction is 'mean':
            loss (FloatTensor): Mean of tensor with group-wise DICE losses of shape [].

        * If reduction is 'sum':
            loss (FloatTensor): Sum of tensor with group-wise DICE losses of shape [].

    Raises:
        ValueError: Error when an invalid reduction string is provided.
    """

    # Get prediction probabilities
    pred_probs = pred_logits.sigmoid()

    # Get group-wise DICE losses
    numerator = 2 * (pred_probs * tgt_labels).sum(dim=1)
    denominator = pred_probs.sum(dim=1) + tgt_labels.sum(dim=1)
    losses = 1 - (numerator + 1) / (denominator + 1)

    # Apply reduction operation on group-wise losses and return
    if reduction == 'none':
        return losses
    elif reduction == 'mean':
        loss = losses.mean()
        return loss
    elif reduction == 'sum':
        loss = losses.sum()
        return loss
    else:
        error_msg = f"Invalid reduction string (got '{reduction}')."
        raise ValueError(error_msg)
