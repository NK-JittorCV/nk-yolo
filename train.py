from nkyolo import YOLO

model = YOLO("yolov8n.yaml")
model.train(data="coco128.yaml", epochs=10, batch=128)
