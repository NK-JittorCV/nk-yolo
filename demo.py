from nkyolo import YOLO
import argparse
import debugpy

def main(args):
    model = YOLO(args.model)

    # Train the model
    train_results = model.train(
        data="coco8.yaml",  # path to dataset YAML
        epochs=2,  # number of training epochs
        batch=1,
        imgsz=640,  # training image size
        device='cpu',  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
    )

    # Evaluate model performance on the validation set
    metrics = model.val()

    # Perform object detection on an image
    results = model("jittoryolo/assets/bus.jpg")
    results[0].show()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="yolov8n.yaml")
    return parser.parse_args()
if __name__ == "__main__":
    args = parse_args()
    main(args)