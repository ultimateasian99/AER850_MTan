from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt
import os


## YOLO V8 Evaluation

# load pre-trained model 
model = YOLO("/home/chenjitong/Millie_ws/AER850_MTan/Project3/runs/detect/3_120-1200-5/train/weights/best.pt")


# Load image and resize
results = model(["/home/chenjitong/Millie_ws/AER850_MTan/Project3/data/evaluation/ardmega.jpg", 
                 "/home/chenjitong/Millie_ws/AER850_MTan/Project3/data/evaluation/arduno.jpg", 
                 "/home/chenjitong/Millie_ws/AER850_MTan/Project3/data/evaluation/rasppi.jpg"])

# Process results list
for i, result in enumerate(results, start=1):
    boxes = result.boxes
    masks = result.masks
    keypoints = result.keypoints
    probs = result.probs
    obb = result.obb
    result.show()
    result.save(filename=f"result_{i}.jpg")