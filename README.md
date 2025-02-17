# JittorYOLO
## Introduction
JittorYOLO is an object detection benchmark based on [Jittor](https://github.com/Jittor/jittor). 

## Supported Models
Currently supported YOLOs are as below:
- YOLOv5: [https://github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5)
- YOLOv6: [https://github.com/meituan/YOLOv6](https://github.com/meituan/YOLOv6)
- YOLOv7: [https://github.com/WongKinYiu/yolov7](https://github.com/WongKinYiu/yolov7)
- YOLOv8: [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)
- YOLOv10: [https://github.com/THU-MIG/yolov10](https://github.com/THU-MIG/yolov10)
- ✨ YOLO-MS: [https://github.com/FishAndWasabi/YOLO-MS](https://github.com/FishAndWasabi/YOLO-MS)

<!-- **Features**
- Automatic compilation. Our framwork is based on Jittor, which means we don't need to Manual compilation for these code with CUDA and C++.
-  -->

<!-- Framework details are avaliable in the [framework.md](docs/framework.md) -->
## Install
JittorYOLO environment requirements:

* System: **Linux**(e.g. Ubuntu/CentOS/Arch), **macOS**, or **Windows Subsystem of Linux (WSL)**
* Python version >= 3.7
* CPU compiler (require at least one of the following)
    * g++ (>=5.4.0)
    * clang (>=8.0)
* GPU compiler (optional)
    * nvcc (>=10.0 for g++ or >=10.2 for clang)
* GPU library: cudnn-dev (recommend tar file installation, [reference link](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#installlinux-tar))

**Step 1: Install the requirements**
```shell
git clone https://github.com/NK-JittorCV/nk-yolo.git
cd nk-yolo
python -m pip install -r requirements.txt
```
If you have any installation problems for Jittor, please refer to [Jittor](https://github.com/Jittor/jittor)

**Step 2: Install JittorYOLO**
 
```shell
cd nk-yolo
pip install -v -e .
```
If you don't have permission for install,please add ```--user```.

Or use ```PYTHONPATH```: 
You can add ```export PYTHONPATH=$PYTHONPATH:{you_own_path}/JRS/python``` into ```.bashrc```, and run
```shell
source .bashrc
```

## Getting Started

### Demo
```shell
python demo.py
```

<!-- ## Models
| Model | Test Size | #Params | FLOPs | AP<sup>val</sup> | Latency |
|:---------------|:----:|:---:|:--:|:--:|:--:|
| [YOLOv10-N](https://huggingface.co/jameslahm/yolov10n) |   640  |     2.3M    |   6.7G   |     38.5%     | 1.84ms |
| [YOLOv10-S](https://huggingface.co/jameslahm/yolov10s) |   640  |     7.2M    |   21.6G  |     46.3%     | 2.49ms |
| [YOLOv10-M](https://huggingface.co/jameslahm/yolov10m) |   640  |     15.4M   |   59.1G  |     51.1%     | 4.74ms |
| [YOLOv10-B](https://huggingface.co/jameslahm/yolov10b) |   640  |     19.1M   |  92.0G |     52.5%     | 5.74ms |
| [YOLOv10-L](https://huggingface.co/jameslahm/yolov10l) |   640  |     24.4M   |  120.3G   |     53.2%     | 7.28ms |
| [YOLOv10-X](https://huggingface.co/jameslahm/yolov10x) |   640  |     29.5M    |   160.4G   |     54.4%     | 10.70ms | -->


## Citation

If this work is helpful for your research, please consider citing the following entry.

```bibtex
@article{hu2020jittor,
  title={Jittor: a novel deep learning framework with meta-operators and unified graph execution},
  author={Hu, Shi-Min and Liang, Dun and Yang, Guo-Ye and Yang, Guo-Wei and Zhou, Wen-Yang},
  journal={Science China Information Sciences},
  volume={63},
  number={222103},
  pages={1--21},
  year={2020}
}

@article{Chen2025,
  title = {YOLO-MS: Rethinking Multi-Scale Representation Learning for Real-time Object Detection},
  ISSN = {1939-3539},
  url = {http://dx.doi.org/10.1109/TPAMI.2025.3538473},
  DOI = {10.1109/tpami.2025.3538473},
  journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
  publisher = {Institute of Electrical and Electronics Engineers (IEEE)},
  author = {Chen, Yuming and Yuan, Xinbin and Wang, Jiabao and Wu, Ruiqi and Li, Xiang and Hou, Qibin and Cheng, Ming-Ming},
  year = {2025},
  pages = {1–14}
}

@article{wang2024yolov10,
  title={YOLOv10: Real-Time End-to-End Object Detection},
  author={Wang, Ao and Chen, Hui and Liu, Lihao and Chen, Kai and Lin, Zijia and Han, Jungong and Ding, Guiguang},
  journal={arXiv preprint arXiv:2405.14458},
  year={2024}
}
```

## Reference
Our code is developed on top of the following open source codebase:

1. [Jittor](https://github.com/Jittor/jittor)
2. [ultralytics](https://github.com/ultralytics/ultralytics)
3. [yolov10](https://github.com/THU-MIG/yolov10/tree/main)

We sincerely appreciate their incredible work.
