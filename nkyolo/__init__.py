# NK-YOLO 🚀, AGPL-3.0 license
# Refer to https://github.com/ultralytics/ultralytics/blob/main/ultralytics/__init__.py

__version__ = "0.0.2"

import os

# Set ENV variables (place before imports)
if not os.environ.get("OMP_NUM_THREADS"):
    os.environ["OMP_NUM_THREADS"] = "1"  # default for reduced CPU utilization during training

from nkyolo.models import YOLO

# settings = SETTINGS
__all__ = (
    "__version__",
     "YOLO"
)