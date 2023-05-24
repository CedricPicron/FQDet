"""
Import backbone modules to guarantee their registration.
"""

from .mmdet import MMDetBackbone  # noqa: F401
from .resnet import ResNet  # noqa: F401
from .timm import TimmBackbone  # noqa: F401
