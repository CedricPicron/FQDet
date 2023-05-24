"""
Collection of modules implementing losses.
"""

from fvcore.nn import sigmoid_focal_loss
from mmdet.models.losses.utils import weight_reduce_loss
import torch
from torch import nn
import torch.nn.functional as F

from models.build import build_model, MODELS
from models.functional.loss import sigmoid_dice_loss


@MODELS.register_module()
class BoxLoss(nn.Module):
    """
    Class implementing the BoxLoss module.

    Attributes:
        box_loss_type (str): String containing the bounding box loss type.
        box_loss (nn.Module): Module containing the underlying bounding box loss function.
    """

    def __init__(self, box_loss_type, box_loss_cfg):
        """
        Initializes the BoxLoss module.

        Args:
            box_loss_type (str): String containing the bounding box loss type.
            box_loss_cfg (Dict): Configuration dictionary specifying the underlying bounding box loss module.
        """

        # Initialization of default nn.Module
        super().__init__()

        # Set attribute containing the bounding box loss type
        self.box_loss_type = box_loss_type

        # Build underlying bounding box loss module
        self.box_loss = build_model(box_loss_cfg)

    def forward(self, box_preds, box_targets, pred_boxes, tgt_boxes, **kwargs):
        """
        Forward method of the BoxLoss module.

        Args:
            box_logits (FloatTensor): 2D bounding box regression predictions of shape [num_preds, 4].
            box_targets (FloatTensor): 2D bounding box regression targets of shape [num_preds, 4].
            pred_boxes (Boxes): Boxes structure containing the predicted 2D bounding boxes of size [num_preds].
            tgt_boxes (Boxes): Boxes structure containing the target 2D bounding boxes of size [num_preds].
            kwargs (Dict): Dictionary containing keyword arguments passed to the underlying loss module.

        Returns:
            box_loss (FloatTensor): Tensor containing the box loss or losses of shape [] or [*] respectively.

        Raises:
            ValueError: Error when an invalid bounding box loss type is provided.
        """

        # Get box loss
        if self.box_loss_type == 'boxes':
            box_loss = self.box_loss(pred_boxes, tgt_boxes, **kwargs)

        elif self.box_loss_type == 'mmdet_boxes':
            pred_boxes = pred_boxes.to_format('xyxy').boxes
            tgt_boxes = tgt_boxes.to_format('xyxy').boxes
            box_loss = self.box_loss(pred_boxes, tgt_boxes, **kwargs)

        elif self.box_loss_type == 'regression':
            box_loss = self.box_loss(box_preds, box_targets, **kwargs)

        else:
            error_msg = f"Invalid bounding box loss type (got '{self.box_loss_type}') in BoxLoss module."
            raise ValueError(error_msg)

        return box_loss


@MODELS.register_module()
class MaskLoss(nn.Module):
    """
    Class implementing the MaskLoss module.

    The MaskLoss module first computes the unreduced mask loss, then averages over the mask dimensions and finally
    applies the reduction operation on the averaged loss tensor.

    Attributes:
        mask_loss (nn.Module): Module computing the unreduced mask loss.
        reduction (str): String containing the reduction operation.
    """

    def __init__(self, mask_loss_cfg):
        """
        Initializes the MaskLoss module.

        Args:
            mask_loss_cfg (Dict): Configuration dictionary specifying the mask loss module.
        """

        # Initialization of default nn.Module
        super().__init__()

        # Build mask loss module
        reduction = mask_loss_cfg.pop('reduction', 'mean')
        mask_loss_cfg['reduction'] = 'none'
        self.mask_loss = build_model(mask_loss_cfg)

        # Set reduction attribute
        self.reduction = reduction

    def forward(self, *args, reduction_override=None, **kwargs):
        """
        Forward method of the MaskLoss module.

        Args:
            args (Tuple): Tuple of positional arguments passed to the underlying mask loss module.
            reduction_override (str): String overriding the module's reduction operation (default=None).
            kwargs (Dict): Dictionary of keyword arguments passed to the underlying mask loss module.

        Returns:
            mask_loss (FloatTensor): Tensor with reduced mask loss or losses of shape [] or [num_masks] respectively.
        """

        # Get unreduced mask loss
        mask_loss = self.mask_loss(*args, **kwargs)

        # Get reduced mask loss
        mask_loss = mask_loss.flatten(1).mean(dim=1)
        reduction = reduction_override if reduction_override is not None else self.reduction
        mask_loss = weight_reduce_loss(mask_loss, reduction=reduction)

        return mask_loss


@MODELS.register_module()
class SigmoidDiceLoss(nn.Module):
    """
    Class implementing the SigmoidDiceLoss module.

    Attributes:
        reduction (str): String specifying the reduction operation applied on group-wise losses.
        weight (float): Factor weighting the sigmoid DICE loss.
    """

    def __init__(self, reduction='mean', weight=1.0):
        """
        Initializes the SigmoidDiceLoss module.

        Args:
            reduction (str): String specifying the reduction operation applied on group-wise losses (default='mean').
            weight (float): Factor weighting the sigmoid DICE loss (default=1.0).
        """

        # Initialization of default nn.Module
        super().__init__()

        # Set attributes
        self.reduction = reduction
        self.weight = weight

    def forward(self, pred_logits, tgt_labels):
        """
        Forward method of the SigmoidDiceLoss module.

        Args:
            pred_logits (FloatTensor): Prediction logits of shape [num_groups, group_size].
            tgt_labels (Tensor): Binary target labels of shape [num_groups, group_size].

        Returns:
            * If self.reduction is 'none':
                loss (FloatTensor): Tensor with group-wise sigmoid DICE losses of shape [*]

            * If self.reduction is 'mean':
                loss (FloatTensor): Mean of tensor with group-wise sigmoid DICE losses of shape [].

            * If self.reduction is 'sum':
                loss (FloatTensor): Sum of tensor with group-wise sigmoid DICE losses of shape [].
        """

        # Get weighted sigmoid DICE loss
        tgt_labels = tgt_labels.to(pred_logits.dtype)
        loss = self.weight * sigmoid_dice_loss(pred_logits, tgt_labels, reduction=self.reduction)

        return loss


@MODELS.register_module()
class SigmoidFocalLoss(nn.Module):
    """
    Class implementing the SigmoidFocalLoss module.

    Attributes:
        alpha (float): Alpha value of the sigmoid focal loss function.
        gamma (float): Gamma value of the sigmoid focal loss function.
        reduction (str): String specifying the reduction operation applied on element-wise losses.
        weight (float): Factor weighting the sigmoid focal loss.
    """

    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean', weight=1.0):
        """
        Initializes the SigmoidFocalLoss module.

        Args:
            alpha (float): Alpha value of the sigmoid focal loss function (default=0.25).
            gamma (float): Gamma valud of the sigmoid focal loss function (default=2.0).
            reduction (str): String specifying the reduction operation applied on element-wise losses (default='mean').
            weight (float): Factor weighting the sigmoid focal loss (default=1.0).
        """

        # Initialization of default nn.Module
        super().__init__()

        # Set attributes
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.weight = weight

    def forward(self, pred_logits, tgt_labels):
        """
        Forward method of the SigmoidFocalLoss module.

        Args:
            pred_logits (FloatTensor): Tensor with prediction logits of shape [*].
            tgt_labels (Tensor): Tensor with binary classification labels of shape [*].

        Returns:
            * If self.reduction is 'none':
                loss (FloatTensor): Tensor with element-wise sigmoid focal losses of shape [*]

            * If self.reduction is 'mean':
                loss (FloatTensor): Mean of tensor with element-wise sigmoid focal losses of shape [].

            * If self.reduction is 'sum':
                loss (FloatTensor): Sum of tensor with element-wise sigmoid focal losses of shape [].
        """

        # Get weighted sigmoid focal loss
        tgt_labels = tgt_labels.to(pred_logits.dtype)
        loss = self.weight * sigmoid_focal_loss(pred_logits, tgt_labels, self.alpha, self.gamma, self.reduction)

        return loss


@MODELS.register_module()
class SigmoidGroupBCELoss(nn.Module):
    """
    Class implementing the SigmoidGroupBCELoss module.

    Attributes:
        reduction (str): String specifying the reduction operation applied on group-wise losses.
        weight (float): Factor weighting the sigmoid group BCE loss.
    """

    def __init__(self, reduction='mean', weight=1.0):
        """
        Initializes the SigmoidGroupBCELoss module.

        Args:
            reduction (str): String specifying the reduction operation applied on group-wise losses (default='mean').
            weight (float): Factor weighting the sigmoid group BCE loss (default=1.0).
        """

        # Initialization of default nn.Module
        super().__init__()

        # Set attributes
        self.reduction = reduction
        self.weight = weight

    def forward(self, pred_logits, tgt_labels):
        """
        Forward method of the SigmoidGroupBCELoss module.

        Args:
            pred_logits (FloatTensor): Prediction logits of shape [num_groups, group_size].
            tgt_labels (Tensor): Binary target labels of shape [num_groups, group_size].

        Returns:
            * If self.reduction is 'none':
                loss (FloatTensor): Tensor with group-wise sigmoid BCE losses of shape [*]

            * If self.reduction is 'mean':
                loss (FloatTensor): Mean of tensor with group-wise sigmoid BCE losses of shape [].

            * If self.reduction is 'sum':
                loss (FloatTensor): Sum of tensor with group-wise sigmoid BCE losses of shape [].

        Raises:
            ValueError: Error when an invalid reduction string is provided.
        """

        # Get group-wise sigmoid BCE losses
        tgt_labels = tgt_labels.to(pred_logits.dtype)
        losses = F.binary_cross_entropy_with_logits(pred_logits, tgt_labels, reduction='none').mean(dim=1)

        # Apply reduction operation on group-wise losses and return
        if self.reduction == 'none':
            return losses
        elif self.reduction == 'mean':
            loss = losses.mean()
            return loss
        elif self.reduction == 'sum':
            loss = losses.sum()
            return loss
        else:
            error_msg = f"Invalid reduction string (got '{self.reduction}')."
            raise ValueError(error_msg)


@MODELS.register_module()
class SigmoidHillLoss(nn.Module):
    """
    Class implementing the SigmoidHillLoss module.

    Attributes:
        low_left (float): Coordinate of low left constant to quadratic transition.
        mid_left (float): Coordinate of middle left quadratic to linear transition.
        up_left (float): Coordinate of up left linear to quadratic transition.
        top (float): Coordinate corresponding to the top of the hill.
        up_right (float): Coordinate of up right quadratic to linear transition.
        mid_right (float): Coordinae of middle right linear to quadratic transition.
        low_right (float): Coordinate of low right quadratic to constant transition.
        coeffs (List): List of size [8] with coefficients of the constant, linear and quadratic sub-functions.
    """

    def __init__(self, low_left=0.1, mid_left=0.2, up_left=0.4, top=0.5, up_right=0.6, mid_right=0.8, low_right=0.9):
        """
        Initializes the SigmoidHillLoss module.

        Args:
            low_left (float): Coordinate of low left constant to quadratic transition (default=0.1).
            mid_left (float): Coordinate of middle left quadratic to linear transition (default=0.2).
            up_left (float): Coordinate of up left linear to quadratic transition (default=0.4).
            top (float): Coordinate corresponding to the top of the hill (default=0.5).
            up_right (float): Coordinate of up right quadratic to linear transition (default=0.6).
            mid_right (float): Coordinae of middle right linear to quadratic transition (default=0.8).
            low_right (float): Coordinate of low right quadratic to constant transition (default=0.9).

        Raises:
            ValueError: Error when low left value is smaller than 0.
            ValueError: Error when low left value is greater than middle left value.
            ValueError: Error when middle left value is greater than up left value.
            ValueError: Error when up left value is greater than top value.
            ValueError: Error when top value is greater than up right value.
            ValueError: Error when up right value is greater than middle right value.
            ValueError: Error when middle right value is greater than low right value.
            ValueError: Error when low right value is greater than 1.
        """

        # Initialization of default nn.Module
        super().__init__()

        # Check inputs
        if low_left <= 0.0:
            error_msg = f"Low left value should be greater than 0 (got {low_left})."
            raise ValueError(error_msg)

        if low_left >= mid_left:
            error_msg = f"Middle left value ({mid_left}) should be greater than low left value ({low_left})."
            raise ValueError(error_msg)

        if mid_left >= up_left:
            error_msg = f"Up left value ({up_left}) should be greater than middle left value ({mid_left})."
            raise ValueError(error_msg)

        if up_left >= top:
            error_msg = f"Top value ({top}) should be greater than up left value ({up_left})."
            raise ValueError(error_msg)

        if top >= up_right:
            error_msg = f"Up right value ({up_right}) should be greater than top value ({top})."
            raise ValueError(error_msg)

        if up_right >= mid_right:
            error_msg = f"Middle right value ({mid_right}) should be greater than up right value ({up_right})."
            raise ValueError(error_msg)

        if mid_right >= low_right:
            error_msg = f"Low right value ({low_right}) should be greater than middle right value ({mid_right})."
            raise ValueError(error_msg)

        if low_right >= 1.0:
            error_msg = f"Low right value should be smaller than 1 (got {low_right})."
            raise ValueError(error_msg)

        # Set hill coordinate attributes
        self.low_left = low_left
        self.mid_left = mid_left
        self.up_left = up_left
        self.top = top
        self.up_right = up_right
        self.mid_right = mid_right
        self.low_right = low_right

        # Get coefficients of constant, linear and quadratic sub-functions
        self.coeffs = [0] * 8
        self.coeffs[0] = (0,)
        self.coeffs[1], f1 = self.get_quadratic(low_left, 0, 0, mid_left, 1)
        self.coeffs[2], f1 = self.get_linear(mid_left, f1, 1, up_left)
        self.coeffs[3], f1 = self.get_quadratic(up_left, f1, 1, top, 0)
        self.coeffs[4], f1 = self.get_quadratic(top, f1, 0, up_right, -1)
        self.coeffs[5], f1 = self.get_linear(up_right, f1, -1, mid_right)
        self.coeffs[6], f1 = self.get_quadratic(mid_right, f1, -1, low_right, 0)
        self.coeffs[7] = (f1,)

    @staticmethod
    def get_linear(x0, f0, df0, x1):
        """
        Get linear function with given properties, and apply it at coordinate x1.

        Args:
            x0 (float): Coordinate x0.
            f0 (float): Function value at x0.
            df0 (float): Derivative value at x0.
            x1 (float): Coordinate x1.

        Returns:
            coeffs (Tuple): Tuple containing the coefficients of the linear function.
            f1 (float): Function value at x1.
        """

        # Get coefficients of linear function
        a = df0
        b = f0 - a*x0
        coeffs = (a, b)

        # Get function value at x1
        f1 = a*x1 + b

        return coeffs, f1

    @staticmethod
    def get_quadratic(x0, f0, df0, x1, df1):
        """
        Get quadratic function with given properties, and apply it at coordinate x1.

        Args:
            x0 (float): Coordinate x0.
            f0 (float): Function value at x0.
            df0 (float): Derivative value at x0.
            x1 (float): Coordinate x1.
            df1 (float): Derivative value at x1.

        Returns:
            coeffs (Tuple): Tuple containing the coefficients of the quadratic function.
            f1 (float): Function value at x1.
        """

        # Get coefficients of quadratic function
        a = 0.5 * (df1-df0) / (x1-x0)
        b = df0 - 2*a*x0
        c = f0 - a*x0**2 - b*x0
        coeffs = (a, b, c)

        # Get function value at x1
        f1 = a*x1**2 + b*x1 + c

        return coeffs, f1

    @staticmethod
    def apply_linear(input, coeffs):
        """
        Apply linear function with given coefficients on given input.

        Args:
            input (FloatTensor): Input tensor of arbitrary shape.
            coeffs (Tuple): Tuple of size [2] containing the coefficients of the linear function.

        Returns:
            output (FloatTensor): Output tensor of same shape as input.

        Raises:
            ValueError: Error when the number of given coefficients is different from two.
        """

        # Check whether exactly two coefficients are provided
        if len(coeffs) != 2:
            error_msg = f"Two coefficients should be provided for linear function, but got {len(coeffs)}."
            raise ValueError(error_msg)

        # Apply linear function on input to get output
        a, b = coeffs
        output = a*input + b

        return output

    @staticmethod
    def apply_quadratic(input, coeffs):
        """
        Apply quadratic function with given coefficients on given input.

        Args:
            input (FloatTensor): Input tensor of arbitrary shape.
            coeffs (Tuple): Tuple of size [3] containing the coefficients of the quadratic function.

        Returns:
            output (FloatTensor): Output tensor of same shape as input.

        Raises:
            ValueError: Error when the number of given coefficients is different from three.
        """

        # Check whether exactly three coefficients are provided
        if len(coeffs) != 3:
            error_msg = f"Three coefficients should be provided for quadratic function, but got {len(coeffs)}."
            raise ValueError(error_msg)

        # Apply quadratic function on input to get output
        a, b, c = coeffs
        output = a*input**2 + b*input + c

        return output

    def forward(self, logits, reduction='mean'):
        """
        Forward method of the SigmoidHillLoss module.

        Args:
            logits (FloatTensor): Tensor with input logits of arbitrary shape.
            reduction (str): String specifying the reduction operation applied on losses tensor (default='mean').

        Returns:
            * If reduction is 'none':
                losses (FloatTensor): Tensor with element-wise sigmoid hill losses of same shape as input logits.

            * If reduction is 'mean':
                loss (FloatTensor): Mean of the tensor with element-wise sigmoid hill losses of shape [1].

            * If reduction is 'sum':
                loss (FloatTensor): Sum of the tensor with element-wise sigmoid hill losses of shape [1].

        Raises:
            ValueError: Error when invalid reduction string is provided.
        """

        # Compute sigmoid probabilities from input logits
        probs = torch.sigmoid(logits)

        # Get masks
        mask0 = probs < self.low_left
        mask1 = (probs >= self.low_left) & (probs < self.mid_left)
        mask2 = (probs >= self.mid_left) & (probs < self.up_left)
        mask3 = (probs >= self.up_left) & (probs < self.top)
        mask4 = (probs >= self.top) & (probs < self.up_right)
        mask5 = (probs >= self.up_right) & (probs < self.mid_right)
        mask6 = (probs >= self.mid_right) & (probs < self.low_right)
        mask7 = probs >= self.low_right

        # Compute sigmoid hill losses
        losses = torch.zeros_like(probs)
        losses[mask0] = self.coeffs[0][0]
        losses[mask1] = self.apply_quadratic(probs[mask1], self.coeffs[1])
        losses[mask2] = self.apply_linear(probs[mask2], self.coeffs[2])
        losses[mask3] = self.apply_quadratic(probs[mask3], self.coeffs[3])
        losses[mask4] = self.apply_quadratic(probs[mask4], self.coeffs[4])
        losses[mask5] = self.apply_linear(probs[mask5], self.coeffs[5])
        losses[mask6] = self.apply_quadratic(probs[mask6], self.coeffs[6])
        losses[mask7] = self.coeffs[7][0]

        # Apply reduction operation to tensor with sigmoid hill losses and return
        if reduction == 'none':
            return losses
        elif reduction == 'mean':
            loss = torch.mean(losses)[None]
            return loss
        elif reduction == 'sum':
            loss = torch.sum(losses)[None]
            return loss
        else:
            error_msg = f"Invalid reduction string '{reduction}'."
            raise ValueError(error_msg)


@MODELS.register_module()
class SmoothL1Loss(nn.Module):
    """
    Class implementing the SmoothL1Loss module.

    Attributes:
        beta (float): Beta value of the smooth L1 loss function.
        reduction (str): String specifying the reduction operation applied on element-wise losses.
        weight (float): Factor weighting the smooth L1 loss.
    """

    def __init__(self, beta=1.0, reduction='mean', weight=1.0):
        """
        Initializes the SmoothL1Loss module.

        Args:
            beta (float): Beta value of the smooth L1 loss function (default=1.0).
            reduction (str): String specifying the reduction operation applied on element-wise losses (default='mean').
            weight (float): Factor weighting the smooth L1 loss (default=1.0).
        """

        # Initialization of default nn.Module
        super().__init__()

        # Set attributes
        self.beta = beta
        self.reduction = reduction
        self.weight = weight

    def forward(self, predictions, targets):
        """
        Forward method of the SmoothL1Loss module.

        Args:
            predictions (FloatTensor): Tensor containing the predictions of shape [*].
            targets (FloatTensor): Tensor containing the targets of shape [*].

        Returns:
            * If self.reduction is 'none':
                loss (FloatTensor): Tensor with element-wise smooth L1 losses of shape [*]

            * If self.reduction is 'mean':
                loss (FloatTensor): Mean of tensor with element-wise smooth L1 losses of shape [].

            * If self.reduction is 'sum':
                loss (FloatTensor): Sum of tensor with element-wise smooth L1 losses of shape [].
        """

        # Get weighted smooth L1 loss
        if self.beta == 0:
            loss = self.weight * F.l1_loss(predictions, targets, reduction=self.reduction)
        else:
            loss = self.weight * F.smooth_l1_loss(predictions, targets, beta=self.beta, reduction=self.reduction)

        return loss
