#libraries
import matplotlib.pyplot as plt
from ultralytics import YOLO
from PIL import Image
import pandas as pd
import numpy as np
import cv2 #4.10


## OBJECT MASKING 

# Load the motherboard image
image_path = "/home/chenjitong/Millie_ws/AER850_MTan/Project3/motherboard_image.JPEG"
image = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
imageRGB = cv2.imread(image_path,cv2.IMREAD_COLOR)

# Checking if the image is loaded correctly
if image is None:
    print(f"Error: Image not found at {image_path}")
else:
    print("Image loaded successfully.")

# threshholding the image
threshold = cv2.threshold(image, 0,255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

# contour detection and noise reduction 
result = np.zeros_like(image)
contours,_ = cv2.findContours(threshold,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

for c in contours:
    area = cv2.contourArea(c)
    if 10000000 > area > 1900000:
        cv2.drawContours(result, [c],0,255,thickness=cv2.FILLED)

# Binary mask
mask = result.copy()

# feature extraction of motherboard
extract = cv2.bitwise_and(imageRGB,imageRGB,mask=mask)

plt.imshow(extract)
plt.show()


# writing images
cv2.imwrite('extracted_motherboard.png', extract)
cv2.imwrite('threshold.png',threshold)
cv2.imwrite('contours.png',result)

# Figures
extract = cv2.resize(extract,(0,0), fx =0.25,fy=0.25)
cv2.imshow("Extracted Motherboard", extract)
cv2.waitKey(0)

threshold = cv2.resize(threshold,(0,0), fx =0.25,fy=0.25)
cv2.imshow("threshold",threshold)
cv2.waitKey(0)

result = cv2.resize(result,(0,0), fx =0.25,fy=0.25)
cv2.imshow("result",result)
cv2.waitKey(0)



## YOLOv8 TRAINING
#load model 
model = YOLO('yolov8n.pt')

# Model Training
data_path = "/home/chenjitong/Millie_ws/AER850_MTan/Project3/data/data.yaml"

train_results = model.train(
    data = data_path,
    epochs = 135,
    imgsz = 1200,
    batch = 5
    )
# evaluate model performance on validation set
metrics = model.val()








