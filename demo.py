from jittoryolo import YOLO
import argparse

def main(args):
    model = YOLO(args.model)
    model.eval()
    # Run batched inference on a list of images
    results = model(["jittoryolo/assets/bus.jpg", "jittoryolo/assets/zidane.jpg"])  # return a list of Results objects

    # Process results list
    for result in results:
        boxes = result.boxes  # Boxes object for bounding box outputs
        masks = result.masks  # Masks object for segmentation masks outputs
        result.show()  # display to screen
        result.save(filename="result.jpg")  # save to disk

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="yolov10n-msv2.yaml")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)