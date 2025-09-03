# convert_pt_to_jittor_pkl.py

import torch
import pickle
import argparse
import os
from collections import OrderedDict

def convert_pt_to_jittor_pkl(pt_file, pkl_file, strict=False):
    """
    Convert PyTorch .pt model weights to Jittor-compatible .pkl file.
    
    Args:
        pt_file (str): Path to input .pt file (PyTorch checkpoint)
        pkl_file (str): Path to output .pkl file (for Jittor)
        strict (bool): Whether to enforce key matching
    """
    print(f"🚀 Loading PyTorch checkpoint from {pt_file}...")
    
    # Load checkpoint
    ckpt = torch.load(pt_file, map_location='cpu', weights_only=False)
    
    # Extract state_dict
    if isinstance(ckpt, dict):
        if 'model' in ckpt:
            state_dict = ckpt['model'].state_dict()  # YOLO/DETR style
        elif 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt.state_dict()
    
    print(f"✅ Loaded {len(state_dict)} tensors.")

    # Convert to OrderedDict and replace 'module.' prefix (if any)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        # Remove 'module.' prefix (from DataParallel)
        name = k[7:] if k.startswith('module.') else k
        
        # Convert torch.Tensor to numpy
        if isinstance(v, torch.Tensor):
            new_state_dict[name] = v.detach().cpu().numpy()
        else:
            new_state_dict[name] = v  # e.g., scalar

    # Save as .pkl for Jittor
    print(f"💾 Saving Jittor-compatible weights to {pkl_file}...")
    with open(pkl_file, 'wb') as f:
        pickle.dump(new_state_dict, f)
    
    print(f"🎉 Conversion completed! Saved {len(new_state_dict)} tensors.")
    print(f"📌 You can now load this .pkl in Jittor using `pickle.load()`.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert PyTorch .pt to Jittor-compatible .pkl")
    parser.add_argument("pt_file", type=str, help="Input .pt file path")
    parser.add_argument("pkl_file", type=str, help="Output .pkl file path")
    parser.add_argument("--strict", action="store_true", help="Enforce strict key matching")

    args = parser.parse_args()

    if not os.path.exists(args.pt_file):
        raise FileNotFoundError(f"❌ {args.pt_file} not found!")

    convert_pt_to_jittor_pkl(args.pt_file, args.pkl_file, args.strict)