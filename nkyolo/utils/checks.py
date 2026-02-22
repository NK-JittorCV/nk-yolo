# NK-YOLO 🚀 AGPL-3.0 License
# Refer to https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/checks.py

import glob
import inspect
import math
import os
import platform
import re
import shutil
import subprocess
import time
from importlib import metadata
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import requests
import jittor as jt

from nkyolo.utils import (
    ASSETS,
    AUTOINSTALL,
    IS_COLAB,
    IS_GIT_DIR,
    IS_JUPYTER,
    IS_KAGGLE,
    IS_PIP_PACKAGE,
    LINUX,
    LOGGER,
    MACOS,
    ONLINE,
    PYTHON_VERSION,
    ROOT,
    USER_CONFIG_DIR,
    WINDOWS,
    Retry,
    SimpleNamespace,
    ThreadingLocked,
    TryExcept,
    clean_url,
    colorstr,
    downloads,
    emojis,
    is_github_action_running,
    url2file,
)


def parse_requirements(file_path=ROOT.parent / "requirements.txt", package=""):
    """
    Parse a requirements.txt file, ignoring lines that start with '#' and any text after '#'.

    Args:
        file_path (Path): Path to the requirements.txt file.
        package (str, optional): Python package to use instead of requirements.txt file, i.e. package='nkyolo'.

    Returns:
        (List[Dict[str, str]]): List of parsed requirements as dictionaries with `name` and `specifier` keys.

    Example:
        ```python
        from nkyolo.utils.checks import parse_requirements

        parse_requirements(package="nkyolo")
        ```
    """
    if package:
        requires = [x for x in metadata.distribution(package).requires if "extra == " not in x]
    else:
        requires = Path(file_path).read_text().splitlines()

    requirements = []
    for line in requires:
        line = line.strip()
        if line and not line.startswith("#"):
            line = line.split("#")[0].strip()  # ignore inline comments
            match = re.match(r"([a-zA-Z0-9-_]+)\s*([<>!=~]+.*)?", line)
            if match:
                requirements.append(SimpleNamespace(name=match[1], specifier=match[2].strip() if match[2] else ""))

    return requirements


def parse_version(version="0.0.0") -> tuple:
    """
    Convert a version string to a tuple of integers, ignoring any extra non-numeric string attached to the version. This
    function replaces deprecated 'pkg_resources.parse_version(v)'.

    Args:
        version (str): Version string, i.e. '2.0.1+cpu'

    Returns:
        (tuple): Tuple of integers representing the numeric part of the version and the extra string, i.e. (2, 0, 1)
    """
    return tuple(map(int, re.findall(r"\d+", version)[:3]))  # '2.0.1+cpu' -> (2, 0, 1)
def is_ascii(s) -> bool:
    """
    Check if a string is composed of only ASCII characters.

    Args:
        s (str): String to be checked.

    Returns:
        (bool): True if the string is composed only of ASCII characters, False otherwise.
    """
    # Convert list, tuple, None, etc. to string
    s = str(s)

    # Check if the string is composed of only ASCII characters
    return all(ord(c) < 128 for c in s)


def check_imgsz(imgsz, stride=32, min_dim=1, max_dim=2, floor=0):
    """
    Verify image size is a multiple of the given stride in each dimension. If the image size is not a multiple of the
    stride, update it to the nearest multiple of the stride that is greater than or equal to the given floor value.

    Args:
        imgsz (int | cList[int]): Image size.
        stride (int): Stride value.
        min_dim (int): Minimum number of dimensions.
        max_dim (int): Maximum number of dimensions.
        floor (int): Minimum allowed value for image size.

    Returns:
        (List[int]): Updated image size.
    """
    # Convert stride to integer if it is a tensor
    stride = int(stride.max() if isinstance(stride, jt.Var) else stride)

    # Convert image size to list if it is an integer
    if isinstance(imgsz, int):
        imgsz = [imgsz]
    elif isinstance(imgsz, (list, tuple)):
        imgsz = list(imgsz)
    elif isinstance(imgsz, str):  # i.e. '640' or '[640,640]'
        imgsz = [int(imgsz)] if imgsz.isnumeric() else eval(imgsz)
    else:
        raise TypeError(
            f"'imgsz={imgsz}' is of invalid type {type(imgsz).__name__}. "
            f"Valid imgsz types are int i.e. 'imgsz=640' or list i.e. 'imgsz=[640,640]'"
        )

    # Apply max_dim
    if len(imgsz) > max_dim:
        msg = (
            "'train' and 'val' imgsz must be an integer, while 'predict' and 'export' imgsz may be a [h, w] list "
            "or an integer, i.e. 'yolo export imgsz=640,480' or 'yolo export imgsz=640'"
        )
        if max_dim != 1:
            raise ValueError(f"imgsz={imgsz} is not a valid image size. {msg}")
        LOGGER.warning(f"WARNING ⚠️ updating to 'imgsz={max(imgsz)}'. {msg}")
        imgsz = [max(imgsz)]
    # Make image size a multiple of the stride
    sz = [max(math.ceil(x / stride) * stride, floor) for x in imgsz]

    # Print warning message if image size was updated
    if sz != imgsz:
        LOGGER.warning(f"WARNING ⚠️ imgsz={imgsz} must be multiple of max stride {stride}, updating to {sz}")

    # Add missing dimensions if necessary
    sz = [sz[0], sz[0]] if min_dim == 2 and len(sz) == 1 else sz[0] if min_dim == 1 and len(sz) == 1 else sz

    return sz


def check_version(
    current: str = "0.0.0",
    required: str = "0.0.0",
    name: str = "version",
    hard: bool = False,
    verbose: bool = False,
    msg: str = "",
) -> bool:
    """
    Check current version against the required version or range.

    Args:
        current (str): Current version or package name to get version from.
        required (str): Required version or range (in pip-style format).
        name (str, optional): Name to be used in warning message.
        hard (bool, optional): If True, raise an AssertionError if the requirement is not met.
        verbose (bool, optional): If True, print warning message if requirement is not met.
        msg (str, optional): Extra message to display if verbose.

    Returns:
        (bool): True if requirement is met, False otherwise.

    Example:
        ```python
        # Check if current version is exactly 22.04
        check_version(current="22.04", required="==22.04")

        # Check if current version is greater than or equal to 22.04
        check_version(current="22.10", required="22.04")  # assumes '>=' inequality if none passed

        # Check if current version is less than or equal to 22.04
        check_version(current="22.04", required="<=22.04")

        # Check if current version is between 20.04 (inclusive) and 22.04 (exclusive)
        check_version(current="21.10", required=">20.04,<22.04")
        ```
    """
    if not current:  # if current is '' or None
        LOGGER.warning(f"WARNING ⚠️ invalid check_version({current}, {required}) requested, please check values.")
        return True
    elif not current[0].isdigit():  # current is package name rather than version string, i.e. current='nkyolo'
        name = current  # assigned package name to 'name' arg
        current = metadata.version(current)  # get version string from package name
    if not required:  # if required is '' or None
        return True

    if "sys_platform" in required and (  # i.e. required='<2.4.0,>=1.8.0; sys_platform == "win32"'
        (WINDOWS and "win32" not in required)
        or (LINUX and "linux" not in required)
        or (MACOS and "macos" not in required and "darwin" not in required)
    ):
        return True

    op = ""
    version = ""
    result = True
    c = parse_version(current)  # '1.2.3' -> (1, 2, 3)
    for r in required.strip(",").split(","):
        m = re.match(r"([^0-9]*)([\d.]+)", r)  # split '>=22.04' -> ('>=', '22.04')
        if not m:
            LOGGER.warning(f"WARNING ⚠️ invalid requirement '{r}' in check_version({current}, {required})")
            continue
        op, version = m.groups()
        if not op:
            op = ">="  # assume >= if no op passed
        v = parse_version(version)  # '1.2.3' -> (1, 2, 3)
        if op == "==" and c != v:
            result = False
        elif op == "!=" and c == v:
            result = False
        elif op == ">=" and not (c >= v):
            result = False
        elif op == "<=" and not (c <= v):
            result = False
        elif op == ">" and not (c > v):
            result = False
        elif op == "<" and not (c < v):
            result = False
    if not result:
        warning = f"WARNING ⚠️ {name}{op}{version} is required, but {name}=={current} is currently installed {msg}"
        if hard:
            raise ModuleNotFoundError(emojis(warning))  # assert version requirements met
        if verbose:
            LOGGER.warning(warning)
    return result


def check_latest_pypi_version(package_name="nkyolo"):
    """
    Returns the latest version of a PyPI package without downloading or installing it.

    Args:
        package_name (str): The name of the package to find the latest version for.

    Returns:
        (str): The latest version of the package.
    """
    requests.packages.urllib3.disable_warnings()  # Disable the InsecureRequestWarning
    response = requests.get(f"https://pypi.org/pypi/{package_name}/json", timeout=3)
    if response.status_code == 200:
        info = response.json().get("info", {})
        version = info.get("version", "")
        return version if isinstance(version, str) else ""
    return ""
def check_pip_update_available():
    """
    Checks if a new version of the NK-YOLO package is available on PyPI.

    Returns:
        (bool): True if an update is available, False otherwise.
    """
    if ONLINE and IS_PIP_PACKAGE:
        from nkyolo import __version__

        latest = check_latest_pypi_version()
        if latest and latest[0].isdigit() and check_version(__version__, f"<{latest}"):
            LOGGER.info(
                f"Update with 'pip install -U nkyolo'"
            )
            return True
    return False


@ThreadingLocked()
def check_font(font="Arial.ttf"):
    """
    Find font locally or download to user's configuration directory if it does not already exist.

    Args:
        font (str): Path or name of font.

    Returns:
        file (Path): Resolved font file path.
    """
    from matplotlib import font_manager

    # Check USER_CONFIG_DIR
    name = Path(font).name
    file = USER_CONFIG_DIR / name
    if file.exists():
        return file

    # Check system fonts
    matches = [s for s in font_manager.findSystemFonts() if font in s]
    if any(matches):
        return matches[0]

    # Download to USER_CONFIG_DIR if missing
    url = f"https://github.com/ultralytics/assets/releases/download/v0.0.0/{name}"
    if downloads.is_url(url, check=True):
        downloads.safe_download(url=url, file=file)
        return file


def check_python(minimum: str = "3.8.0", hard: bool = True, verbose: bool = True) -> bool:
    """
    Check current python version against the required minimum version.

    Args:
        minimum (str): Required minimum version of python.
        hard (bool, optional): If True, raise an AssertionError if the requirement is not met.
        verbose (bool, optional): If True, print warning message if requirement is not met.

    Returns:
        (bool): Whether the installed Python version meets the minimum constraints.
    """
    return check_version(PYTHON_VERSION, minimum, name="Python", hard=hard, verbose=verbose)


@TryExcept()
def check_requirements(requirements=ROOT.parent / "requirements.txt", exclude=(), install=True, cmds=""):
    """
    Check if installed dependencies meet YOLOv8 requirements and attempt to auto-update if needed.

    Args:
        requirements (Union[Path, str, List[str]]): Path to a requirements.txt file, a single package requirement as a
            string, or a list of package requirements as strings.
        exclude (Tuple[str]): Tuple of package names to exclude from checking.
        install (bool): If True, attempt to auto-update packages that don't meet requirements.
        cmds (str): Additional commands to pass to the pip install command when auto-updating.

    Example:
        ```python
        from nkyolo.utils.checks import check_requirements

        # Check a requirements.txt file
        check_requirements("path/to/requirements.txt")

        # Check a single package
        check_requirements("nkyolo>=0.0.1")

        # Check multiple packages
        check_requirements(["numpy", "nkyolo>=0.0.1"])
        ```
    """
    prefix = colorstr("red", "bold", "requirements:")
    if isinstance(requirements, Path):  # requirements.txt file
        file = requirements.resolve()
        assert file.exists(), f"{prefix} {file} not found, check failed."
        requirements = [f"{x.name}{x.specifier}" for x in parse_requirements(file) if x.name not in exclude]
    elif isinstance(requirements, str):
        requirements = [requirements]

    pkgs = []
    for r in requirements:
        r_stripped = r.split("/")[-1].replace(".git", "")  # replace git+https://org/repo.git -> 'repo'
        match = re.match(r"([a-zA-Z0-9-_]+)([<>!=~]+.*)?", r_stripped)
        name, required = match[1], match[2].strip() if match[2] else ""
        assert check_version(metadata.version(name), required)  # exception if requirements not met
    @Retry(times=2, delay=1)
    def attempt_install(packages, commands):
        """Attempt pip install command with retries on failure."""
        return subprocess.check_output(f"pip install --no-cache-dir {packages} {commands}", shell=True).decode()

    s = " ".join(f'"{x}"' for x in pkgs)  # console string
    if s:
        if install and AUTOINSTALL:  # check environment variable
            n = len(pkgs)  # number of packages updates
            LOGGER.info(f"{prefix} nkyolo requirement{'s' * (n > 1)} {pkgs} not found, attempting AutoUpdate...")
            t = time.time()
            assert ONLINE, "AutoUpdate skipped (offline)"
            LOGGER.info(attempt_install(s, cmds))
            dt = time.time() - t
            LOGGER.info(
                f"{prefix} AutoUpdate success ✅ {dt:.1f}s, installed {n} package{'s' * (n > 1)}: {pkgs}\n"
                f"{prefix} ⚠️ {colorstr('bold', 'Restart runtime or rerun command for updates to take effect')}\n"
            )
        else:
            return False

    return True



def check_suffix(file="yolo11n.pt", suffix=".pt", msg=""):
    """Check file(s) for acceptable suffix."""
    if file and suffix:
        if isinstance(suffix, str):
            suffix = (suffix,)
        for f in file if isinstance(file, (list, tuple)) else [file]:
            s = Path(f).suffix.lower().strip()  # file suffix
            if len(s):
                assert s in suffix, f"{msg}{f} acceptable suffix is {suffix}, not {s}"


def check_yolov5u_filename(file: str, verbose: bool = True):
    """Replace legacy YOLOv5 filenames with updated YOLOv5u filenames."""
    if "yolov3" in file or "yolov5" in file:
        if "u.yaml" in file:
            file = file.replace("u.yaml", ".yaml")  # i.e. yolov5nu.yaml -> yolov5n.yaml
        elif ".pt" in file and "u" not in file:
            original_file = file
            file = re.sub(r"(.*yolov5([nsmlx]))\.pt", "\\1u.pt", file)  # i.e. yolov5n.pt -> yolov5nu.pt
            file = re.sub(r"(.*yolov5([nsmlx])6)\.pt", "\\1u.pt", file)  # i.e. yolov5n6.pt -> yolov5n6u.pt
            file = re.sub(r"(.*yolov3(|-tiny|-spp))\.pt", "\\1u.pt", file)  # i.e. yolov3-spp.pt -> yolov3-sppu.pt
            if file != original_file and verbose:
                LOGGER.info(
                    f"PRO TIP 💡 Replace 'model={original_file}' with new 'model={file}'.\nYOLOv5 'u' models are "
                    f"trained with https://github.com/ultralytics/ultralytics and feature improved performance vs "
                    f"standard YOLOv5 models trained with https://github.com/ultralytics/yolov5.\n"
                )
    return file


def check_model_file_from_stem(model="yolov8n"):
    """Return a model filename from a valid model stem."""
    if model and not Path(model).suffix and Path(model).stem in downloads.GITHUB_ASSETS_STEMS:
        return Path(model).with_suffix(".pt")  # add suffix, i.e. yolov8n -> yolov8n.pt
    else:
        return model


def check_file(file, suffix="", download=True, download_dir=".", hard=True):
    """Search/download file (if necessary) and return path."""
    check_suffix(file, suffix)  # optional
    file = str(file).strip()  # convert to string and strip spaces
    file = check_yolov5u_filename(file)  # yolov5n -> yolov5nu
    if (
        not file
        or ("://" not in file and Path(file).exists())  # '://' check required in Windows Python<3.10
        or file.lower().startswith("grpc://")
    ):  # file exists or gRPC Triton images
        return file
    elif download and file.lower().startswith(("https://", "http://", "rtsp://", "rtmp://", "tcp://")):  # download
        url = file  # warning: Pathlib turns :// -> :/
        file = Path(download_dir) / url2file(file)  # '%2F' to '/', split https://url.com/file.txt?auth
        if file.exists():
            LOGGER.info(f"Found {clean_url(url)} locally at {file}")  # file already exists
        else:
            downloads.safe_download(url=url, file=file, unzip=False)
        return str(file)
    else:  # search
        files = glob.glob(str(ROOT / "**" / file), recursive=True) or glob.glob(str(ROOT.parent / file))  # find file
        if not files and hard:
            raise FileNotFoundError(f"'{file}' does not exist")
        elif len(files) > 1 and hard:
            raise FileNotFoundError(f"Multiple files match '{file}', specify exact path: {files}")
        return files[0] if len(files) else []  # return file


def check_yaml(file, suffix=(".yaml", ".yml"), hard=True):
    """Search/download YAML file (if necessary) and return path, checking suffix."""
    return check_file(file, suffix, hard=hard)


def check_is_path_safe(basedir, path):
    """
    Check if the resolved path is under the intended directory to prevent path traversal.

    Args:
        basedir (Path | str): The intended directory.
        path (Path | str): The path to check.

    Returns:
        (bool): True if the path is safe, False otherwise.
    """
    base_dir_resolved = Path(basedir).resolve()
    path_resolved = Path(path).resolve()

    return path_resolved.exists() and path_resolved.parts[: len(base_dir_resolved.parts)] == base_dir_resolved.parts


def check_imshow(warn=False):
    """Check if environment supports image displays."""
    if LINUX:
        assert not IS_COLAB and not IS_KAGGLE
        assert "DISPLAY" in os.environ, "The DISPLAY environment variable isn't set."
    cv2.imshow("test", np.zeros((8, 8, 3), dtype=np.uint8))  # show a small 8-pixel image
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    return True
def check_yolo(verbose=True, device=""):
    """Return a human-readable YOLO software and hardware summary."""
    import psutil

    from nkyolo.utils.jittor_utils import select_device

    if IS_JUPYTER:
        if check_requirements("wandb", install=False):
            os.system("pip uninstall -y wandb")  # uninstall wandb: unwanted account creation prompt with infinite hang
        if IS_COLAB:
            shutil.rmtree("sample_data", ignore_errors=True)  # remove colab /sample_data directory

    if verbose:
        # System info
        gib = 1 << 30  # bytes per GiB
        ram = psutil.virtual_memory().total
        total, used, free = shutil.disk_usage("/")
        s = f"({os.cpu_count()} CPUs, {ram / gib:.1f} GB RAM, {(total - free) / gib:.1f}/{total / gib:.1f} GB disk)"
        from IPython import display

        display.clear_output()  # clear display if notebook
    else:
        s = ""

    select_device(device=device, newline=False)
    LOGGER.info(f"Setup complete ✅ {s}")


def collect_system_info():
    """Collect and print relevant system information including OS, Python, RAM, CPU, and CUDA."""
    import psutil

    from nkyolo.utils import ENVIRONMENT  # scope to avoid circular import
    from nkyolo.utils.jittor_utils import get_cpu_info, get_gpu_info

    gib = 1 << 30  # bytes per GiB
    cuda = jt and jt.cuda.is_available()
    check_yolo()
    total, used, free = shutil.disk_usage("/")

    info_dict = {
        "OS": platform.platform(),
        "Environment": ENVIRONMENT,
        "Python": PYTHON_VERSION,
        "Install": "git" if IS_GIT_DIR else "pip" if IS_PIP_PACKAGE else "other",
        "RAM": f"{psutil.virtual_memory().total / gib:.2f} GB",
        "Disk": f"{(total - free) / gib:.1f}/{total / gib:.1f} GB",
        "CPU": get_cpu_info(),
        "CPU count": os.cpu_count(),
        "GPU": get_gpu_info(index=0) if cuda else None,
        "GPU count": jt.cuda.device_count() if cuda else None,
        "CUDA": jt.version.cuda if cuda else None,
    }
    LOGGER.info("\n" + "\n".join(f"{k:<20}{v}" for k, v in info_dict.items()) + "\n")

    package_info = {}
    for r in parse_requirements(package="nkyolo"):
        current = metadata.version(r.name)
        is_met = "✅ " if check_version(current, str(r.specifier), name=r.name, hard=True) else "❌ "
        package_info[r.name] = f"{is_met}{current}{r.specifier}"
        LOGGER.info(f"{r.name:<20}{package_info[r.name]}")

    info_dict["Package Info"] = package_info

    if is_github_action_running():
        github_info = {
            "RUNNER_OS": os.getenv("RUNNER_OS"),
            "GITHUB_EVENT_NAME": os.getenv("GITHUB_EVENT_NAME"),
            "GITHUB_WORKFLOW": os.getenv("GITHUB_WORKFLOW"),
            "GITHUB_ACTOR": os.getenv("GITHUB_ACTOR"),
            "GITHUB_REPOSITORY": os.getenv("GITHUB_REPOSITORY"),
            "GITHUB_REPOSITORY_OWNER": os.getenv("GITHUB_REPOSITORY_OWNER"),
        }
        LOGGER.info("\n" + "\n".join(f"{k}: {v}" for k, v in github_info.items()))
        info_dict["GitHub Info"] = github_info

    return info_dict


def check_amp(model):
    """
    Checks Automatic Mixed Precision (AMP) functionality for a YOLO11 model. If the checks fail, it means
    there are anomalies with AMP on the system that may cause NaN losses or zero-mAP results, so AMP will be disabled
    during training.

    Args:
        model (nn.Module): A YOLO11 model instance.

    Example:
        ```python
        from nkyolo import YOLO
        from nkyolo.utils.checks import check_amp

        model = YOLO("yolo11n.pt").model.cuda()
        check_amp(model)
        ```

    Returns:
        (bool): Returns True if the AMP functionality works correctly with YOLO11 model, else False.
    """
    # AMP is temporarily disabled.
    raise NotImplementedError("AMP is temporarily disabled.")


def git_describe(path=ROOT):  # path must be a directory
    """Return human-readable git description, i.e. v5.0-5-g3e25f1e https://git-scm.com/docs/git-describe."""
    return subprocess.check_output(f"git -C {path} describe --tags --long --always", shell=True).decode()[:-1]
def print_args(args: Optional[dict] = None, show_file=True, show_func=False):
    """Print function arguments (optional args dict)."""

    def strip_auth(v):
        """Clean longer NK-YOLO HUB URLs by stripping potential authentication information."""
        return clean_url(v) if (isinstance(v, str) and v.startswith("http") and len(v) > 100) else v

    x = inspect.currentframe().f_back  # previous frame
    file, _, func, _, _ = inspect.getframeinfo(x)
    if args is None:  # get args automatically
        args, _, _, frm = inspect.getargvalues(x)
        args = {k: v for k, v in frm.items() if k in args}
    file = Path(file).resolve().relative_to(ROOT).with_suffix("")
    s = (f"{file}: " if show_file else "") + (f"{func}: " if show_func else "")
    LOGGER.info(colorstr(s) + ", ".join(f"{k}={strip_auth(v)}" for k, v in args.items()))


def cuda_device_count() -> int:
    """
    Get the number of NVIDIA GPUs available in the environment.

    Returns:
        (int): The number of NVIDIA GPUs available.
    """
    # Run the nvidia-smi command and capture its output
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=count", "--format=csv,noheader,nounits"], encoding="utf-8"
    )

    # Take the first line and strip any leading/trailing white space
    first_line = output.strip().split("\n")[0]

    return int(first_line)
def cuda_is_available() -> bool:
    """
    Check if CUDA is available in the environment.

    Returns:
        (bool): True if one or more NVIDIA GPUs are available, False otherwise.
    """
    return cuda_device_count() > 0


# Run checks and define constants
check_python("3.8", hard=False, verbose=True)  # check python version
IS_PYTHON_MINIMUM_3_10 = check_python("3.10", hard=False)
IS_PYTHON_3_12 = PYTHON_VERSION.startswith("3.12")
