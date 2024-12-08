#libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2 #4.10

## OBJECT MASKING 
image =cv2.imread("C:\\Users\\Millie Tan\\Documents\\AER850_MTan\\Project 3\\motherboard_image.JPEG")
# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# Apply thresholding to the image
_, thresholded = cv2.threshold(gray_image, 110, 255, cv2.THRESH_BINARY)

# Invert mask
inverted_mask=cv2.bitwise_not(thresholded)

# Perform edge detection using contour detection
contours, _ = cv2.findContours(inverted_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Find the largest contour (assuming it's the motherboard)
largest_contour = max(contours, key=cv2.contourArea)
image_with_contours = image.copy()

# Create an empty mask to extract the board
mask = np.zeros_like(gray_image)
cv2.drawContours(mask, [largest_contour], -1, (255, 255, 255), thickness=cv2.FILLED)
cv2.drawContours(image_with_contours, [largest_contour], -1, (0, 255, 0), 2)

# Extract the motherboard from the background
board_extracted = cv2.bitwise_and(image, image, mask=mask)

#Image resizing
target_width, target_height = int(2172/2.5),int(2896/2.5)
resized_image = cv2.resize(image, (target_width, target_height))

# Display the results
cv2.imshow('',resized_image)
cv2.imshow('',inverted_mask)
cv2.imshow('',board_extracted)
cv2.imshow('',image_with_contours)
cv2.waitKey(0)
cv2.destroyAllWindows()