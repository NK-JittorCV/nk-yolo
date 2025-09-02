from nkyolo import YOLO
import argparse

def main(args):
    model = YOLO(args.model)
    model.eval()
    results = model("assets/bus.jpg")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="yolov8n.yaml")
    return parser.parse_args()
if __name__ == "__main__":
    args = parse_args()
    main(args)