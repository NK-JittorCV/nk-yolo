from copy import deepcopy
from functools import lru_cache
from pathlib import Path

import numpy as np
import jittor as jt

from jittoryolo.data.augment import LetterBox
from jittoryolo.utils import LOGGER,SimpleClass,ops
# from jittoryolo.utils.checks import check_requirements
# from jittoryolo.utils.plotting import Annotator, colors, save_one_box

