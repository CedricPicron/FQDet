"""
Import from different sub-directories to guarantee module registration.
"""

from .build import MODELS  # noqa: F401
from .archs import *  # noqa: F401, F403
from .backbones import *  # noqa: F401, F403
from .cores import *  # noqa: F401, F403
from .extensions import *  # noqa: F401, F403
from .heads import *  # noqa: F401, F403
from .modules import *  # noqa: F401, F403
