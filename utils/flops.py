"""
Collection of flop counting utilities.
"""

from collections import Counter
from math import prod
from numbers import Number
import warnings

from detectron2.export import TracingAdapter
from detectron2.utils.analysis import _IGNORED_OPS
from fvcore.common.checkpoint import _named_modules_with_dup
from fvcore.nn.flop_count import _DEFAULT_SUPPORTED_OPS
from fvcore.nn.jit_analysis import JitModelAnalysis as FvcoreJitModelAnalysis
from fvcore.nn.jit_analysis import _get_scoped_trace_graph, Statistics
from fvcore.nn.jit_handles import get_shape
from torch.jit import TracerWarning


def id_conv2d_flop_jit(inputs, outputs):
    """
    Function counting the number of 2D index-based convolution FLOPs.

    Args:
        inputs (List): List with inputs of the 2D index-based convolution operation.
        outputs (List): List with outputs of the 2D index-based convolution operation.

    Returns:
        flops (Counter): Counter dictionary containing the number of 2D index-based convolution FLOPs.
    """

    # Get weight from list with inputs
    weight = inputs[3]

    # Get number of 2D index-based convolution FLOPs
    flops = get_shape(weight)[1] * prod(get_shape(outputs[0]))
    flops = Counter({'id_conv2d': flops})

    return flops


def msda_flop_jit(inputs, outputs):
    """
    Function counting the number of MSDA (MultiScaleDeformableAttention) FLOPs.

    Args:
        inputs (List): List with inputs of the MSDA operation.
        outputs (List): List with outputs of the MSDA operation.

    Returns:
        flops (Counter): Counter dictionary containing the number of MSDA FLOPs.
    """

    # Get number of levels and number of points
    attn_weights = inputs[4]
    num_levels, num_points = get_shape(attn_weights)[-2:]

    # Get number of MSDA FLOPs
    flops = 5 * num_levels * num_points * prod(get_shape(outputs[0]))
    flops = Counter({'msda': flops})

    return flops


def roi_align_mmcv_flop_jit(inputs, outputs):
    """
    Function counting the number of MMCV RoIAlign FLOPs.

    Args:
        inputs (List): List with inputs of the MMCV RoIAlign operation.
        outputs (List): List with outputs of the MMCV RoIAlign operation.

    Returns:
        flops (Counter): Counter dictionary containing the number of MMCV RoIAlign FLOPs.
    """

    # Get sampling ratio
    sampling_ratio = inputs[4]
    sampling_ratio = 1 if sampling_ratio == 0 else sampling_ratio

    # Get number of MMCV RoIAlign FLOPs
    flops = 4 * sampling_ratio**2 * prod(get_shape(outputs[0]))
    flops = Counter({'roi_align': flops})

    return flops


class JitModelAnalysis(FvcoreJitModelAnalysis):
    """
    Modified version of the JitModelAnalysis class from fvcore.

    It adds non-tensor inputs to list with inputs of custom autograd operations.
    """

    def _analyze(self) -> "Statistics":
        """
        Computes the statistics of the JitModelAnalysis object.

        It modifies the implementation from fvcore by adding non-tensor inputs to the list with inputs of custom
        autograd operations.
        """

        # Don't calculate if results are already stored.
        stats = self._stats
        if stats is not None:
            return stats

        with warnings.catch_warnings():
            if self._warn_trace == "none":
                warnings.simplefilter("ignore")
            elif self._warn_trace == "no_tracer_warning":
                warnings.filterwarnings("ignore", category=TracerWarning)
            graph = _get_scoped_trace_graph(self._model, self._inputs, self._aliases)

        # Assures even modules not in the trace graph are initialized to zero count
        counts = {}
        unsupported_ops = {}
        # We don't need the duplication here, but self._model.named_modules()
        # gives slightly different results for some wrapped models.
        for _, mod in _named_modules_with_dup(self._model):
            name = self._aliases[mod]
            counts[name] = Counter()
            unsupported_ops[name] = Counter()

        all_seen = set()
        for node in graph.nodes():
            kind = node.kind()
            if kind == "prim::PythonOp":
                # for PythonOp, pyname contains the actual name in Python
                kind = kind + "." + node.pyname()
            scope_names = node.scopeName().split("/")
            all_seen.update(scope_names)
            if self._ancestor_mode == "caller":
                ancestors = set(scope_names)
            else:
                ancestors = self._get_all_ancestors(scope_names[-1])
                all_seen.update(ancestors)
            if kind not in self._op_handles:
                # Ignore all prim:: operators. However, prim::PythonOp can be
                # a user-implemented `torch.autograd.Function` so we shouldn't
                # ignore it.
                if kind in self._ignored_ops or (
                    kind.startswith("prim::") and not kind.startswith("prim::PythonOp")
                ):
                    continue

                for name in ancestors:
                    unsupported_ops[name][kind] += 1
            else:
                inputs, outputs = list(node.inputs()), list(node.outputs())
                inputs = inputs + node.scalar_args() if kind.startswith("prim::PythonOp") else inputs
                op_counts = self._op_handles[kind](inputs, outputs)
                if isinstance(op_counts, Number):
                    op_counts = Counter({self._simplify_op_name(kind): op_counts})

                # Assures an op contributes at most once to a module
                for name in ancestors:
                    counts[name] += op_counts

        uncalled_mods = set(self._aliases.values()) - all_seen
        stats = Statistics(
            counts=counts, unsupported_ops=unsupported_ops, uncalled_mods=uncalled_mods
        )
        self._stats = stats
        self._warn_unsupported_ops(unsupported_ops[""])
        self._warn_uncalled_mods(uncalled_mods)
        return stats


class FlopCountAnalysis(JitModelAnalysis):
    """
    Modified version of the FlopCountAnalysis class from Detectron2 using our modified JitModelAnalysis class.
    """

    def __init__(self, model, inputs):
        """
        Args:
            model (nn.Module): Module containing the desired model.
            inputs (Any): Inputs of the given model which can be of any type.
        """

        # Initialize FlopCountAnalysis object
        wrapper = TracingAdapter(model, inputs, allow_non_tensor=True)
        super().__init__(wrapper, wrapper.flattened_inputs)
        self.set_op_handle(**_DEFAULT_SUPPORTED_OPS)
        self.set_op_handle(**{k: None for k in _IGNORED_OPS})
