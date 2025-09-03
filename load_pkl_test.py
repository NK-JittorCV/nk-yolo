# infer_with_pkl.py

from nkyolo import YOLO
import jittor as jt
import argparse
import pickle
import os

def load_pkl_weights(model, pkl_file):
    """lad
    Load Jittor-compatible .pkl weights into nkyolo YOLO model.
    
    Args:
        model: YOLO instance (Jittor model)
        pkl_file: Path to .pkl file (dict: {param_name: numpy_array})
    """
    if not os.path.exists(pkl_file):
        raise FileNotFoundError(f"❌ {pkl_file} not found!")

    print(f"📦 Loading weights from {pkl_file}...")
    with open(pkl_file, 'rb') as f:
        state_dict = pickle.load(f)

    # 获取模型当前所有参数
    model_state = model.model.state_dict()  # 假设 YOLO.model 是主网络

    loaded_count = 0
    mismatched = []
    not_found = []

    for name, param in model_state.items():
        if name in state_dict:
            data = state_dict[name]
            if param.shape == data.shape:
                param.update(jt.array(data))  # 更新 Jittor 参数
                loaded_count += 1
            else:
                mismatched.append(f"{name}: model {param.shape} != pkl {data.shape}")
        else:
            not_found.append(name)

    print(f"✅ Successfully loaded {loaded_count}/{len(model_state)} parameters.")

    if mismatched:
        print(f"❌ Shape mismatch ({len(mismatched)}):")
        for m in mismatched[:10]:
            print(f"   {m}")
        if len(mismatched) > 10:
            print(f"   ... and {len(mismatched)-10} more.")

    if not_found:
        print(f"🟡 Missing in .pkl ({len(not_found)}):")
        for n in not_found[:10]:
            print(f"   {n}")
        if len(not_found) > 10:
            print(f"   ... and {len(not_found)-10} more.")

    return model


def main(args):
    # 1. 创建模型（仅结构，不加载权重）
    print("🏗️  Creating model from YAML...")
    model = YOLO(args.model)  # 如 yolov8n.yaml，只构建结构
    model.eval()  # 切换到评估模式

    # 2. 加载 .pkl 权重
    model = load_pkl_weights(model, args.pkl)

    # 3. 推理
    print("🚀 Running inference...")
    results = model("assets/bus.jpg")

    # 4. 打印结果
    for r in results:
        print(r.summary())  # 或 r.plot() 显示图像
        # r.save("result.jpg")  # 保存结果

    print("🎉 Inference completed.")


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO inference with .pkl weights")
    parser.add_argument("--model", type=str, default="yolov8n.yaml", help="Model config YAML")
    parser.add_argument("--pkl", type=str, required=True, help="Path to .pkl weight file")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)