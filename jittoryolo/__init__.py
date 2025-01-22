# jittoryolo YOLO 🚀, AGPL-3.0 license

__version__ = "8.3.50"

import os

# Set ENV variables (place before imports)
if not os.environ.get("OMP_NUM_THREADS"):
    os.environ["OMP_NUM_THREADS"] = "1"  # default for reduced CPU utilization during training

from jittoryolo.models import YOLO
from jittoryolo.utils import ASSETS, SETTINGS
from jittoryolo.utils.checks import check_yolo as checks
from jittoryolo.utils.downloads import download

# settings = SETTINGS
__all__ = (
    "__version__",
    # "ASSETS",
    # "YOLO",
    # "YOLOWorld",
    # "NAS",
    # "SAM",
    # "FastSAM",
    # "RTDETR",
    # "checks",
    # "download",
    # "settings",
)