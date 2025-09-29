from nkyolo import YOLO
from ultralytics import YOLO as YOLO_ULT
import argparse
import cv2

def main(args):
    model = YOLO(args.model)
    model.load(args.weights)    
    model.eval()
    ultralytics_model = YOLO_ULT(args.ultralytics_model)
    ultralytics_model.load(args.ultralytics_weights)
    ultralytics_model.eval()
    results = model(args.image)
    ultralytics_results = ultralytics_model(args.image)
    for i, (result, ultralytics_result) in enumerate(zip(results, ultralytics_results)):
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()  # bounding box coordinates
            conf = box.conf[0].item()  # confidence
            cls = int(box.cls[0].item())  # class index
            if 0 <= cls < len(model.names):
                cls_name = model.names[cls]
            else:
                continue
            print(f"[Jittor Version] Class: {cls_name}, Confidence: {conf:.2f}, Coordinates: ({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})")
        
        
        for box in ultralytics_result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()  # bounding box coordinates
            conf = box.conf[0].item()  # confidence
            cls = int(box.cls[0].item())  # class index
            if 0 <= cls < len(model.names):
                cls_name = model.names[cls]
            else:
                continue
            print(f"[Ultralytics Version] Class: {cls_name}, Confidence: {conf:.2f}, Coordinates: ({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})")

    visualized_img = result.plot()  # draw bounding boxes on image
    save_path = f"jittor_inference_result_{i+1}.jpg"
    cv2.imwrite(save_path, visualized_img)
    print(f"[Jittor Version] Inference result saved to: {save_path}")
    
    visualized_img = ultralytics_result.plot()  # draw bounding boxes on image
    save_path = f"ultralytics_inference_result_{i+1}.jpg"
    cv2.imwrite(save_path, visualized_img)
    print(f"[Ultralytics Version] Inference result saved to: {save_path}")
        
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="yolo12.yaml")
    parser.add_argument("--weights", type=str, default="yolo12n.pkl")
    parser.add_argument("--ultralytics_model", type=str, default="yolo12n.yaml")
    parser.add_argument("--ultralytics_weights", type=str, default="yolo12n.pt")
    parser.add_argument("--image", type=str, default="assets/bus.jpg")
    return parser.parse_args()
if __name__ == "__main__":
    args = parse_args()
    main(args)