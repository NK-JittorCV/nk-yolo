import inspect
from pathlib import Path
from typing import Any,Dict,List,Union

import numpy as np
import jittor as jt
from PIL import Image

from jittoryolo.cfg import TASK2DATA,get_cfg,get_save_dir
from jittoryolo.engine.results import Result


