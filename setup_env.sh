#!/bin/bash
# One-shot environment setup for NK-YOLO.
#
# Installs the PINNED Jittor from the third_party/jittor submodule (branch
# nk-stable of github.com/FishAndWasabi/jittor — carries fp16-training, NMS
# and pooling fixes absent from stock Jittor wheels), then NK-YOLO itself.
set -e
cd "$(dirname "$0")"

git submodule update --init third_party/jittor

python -m pip install -r requirements.txt
python -m pip install -e third_party/jittor/python
python -m pip install -e .

python - <<'EOF'
import jittor, nkyolo
print(f"OK: jittor {jittor.__version__} @ {jittor.__file__}")
EOF
