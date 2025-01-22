from jittoryolo import YOLO


def main():
    model = YOLO("yolov10s.yaml")
    model.eval()
    # Run batched inference on a list of images
    results = model(["../jittoryolo/assets/bus.jpg", "../jittoryolo/assets/zidane.jpg"])  # return a list of Results objects

    # Process results list
    for result in results:
        boxes = result.boxes  # Boxes object for bounding box outputs
        masks = result.masks  # Masks object for segmentation masks outputs
        result.show()  # display to screen
        result.save(filename="result.jpg")  # save to disk

if __name__ == "__main__":
    main()