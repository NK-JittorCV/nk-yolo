from jittoryolo import YOLO
import argparse
import debugpy
# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 9506))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass
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
# from jittoryolo import YOLO
# import argparse
# # Load a model
# def main(args):
#     model = YOLO(args.model)

#     # Train the model
#     train_results = model.train(
#         data="coco8.yaml",  # path to dataset YAML
#         epochs=100,  # number of training epochs
#         imgsz=640,  # training image size
#         device="cpu",  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
#     )

#     # Evaluate model performance on the validation set
#     metrics = model.val()

#     # Perform object detection on an image
#     results = model("jittoryolo/assets/bus.jpg")
#     results[0].show()

# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--model", type=str, default="yolov10n-msv2.yaml")
#     return parser.parse_args()
# if __name__ == "__main__":
#     args = parse_args()
#     main(args)