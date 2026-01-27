from nkyolo import YOLO

model = YOLO("yolov8n.yaml")
model.train(data="coco_test.yaml", epochs=10, batch=128, device="0,1")