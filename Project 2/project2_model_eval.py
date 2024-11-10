import tensorflow as tf
from keras._tf_keras.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from keras._tf_keras.keras.models import load_model
import os
import matplotlib.pyplot as plt

#STEP 5

#Load the saved Keras model
loaded_model = load_model('mymodel.keras')

img_height = 500
img_width = 500

# Define the relative path to the training directory 
train_data_dir = 'Data/test' 

# Automatically retrieve class labels from subfolder names in the training directory
class_labels = {i: class_name for i, class_name in enumerate(sorted(os.listdir(train_data_dir)))}
print("Class labels:", class_labels)


specific_images = {
    "crack": "test_crack.jpg",  
    "missing-head": "test_missinghead.jpg",  
    "paint-off": "test_paintoff.jpg",
}

# Building image_paths to find the specified image in each class subfolder
image_paths = {}
for class_name, image_name in specific_images.items():
    class_folder = os.path.join(train_data_dir, class_name)
    image_path = os.path.join(class_folder, image_name)
    if os.path.exists(image_path):
        image_paths[class_name] = image_path
    else:
        print(f"Warning: Image {image_name} not found in folder {class_folder}")

print("Image paths for evaluation:", image_paths)

# Function to preprocess and predict the class of an input image
def predict_and_display_image(img_path, model, class_labels, true_label):
    # Load and preprocess the image
    img = load_img(img_path, target_size=(img_height, img_width))
    img_array = img_to_array(img) / 255.0  # Normalize 
    img_array = np.expand_dims(img_array, axis=0)  

    # Predict the class probabilities
    predictions = model.predict(img_array)[0]  # Get prediction probabilities for each class

    # Get the predicted class with the highest probability, since softmax was used
    predicted_index = np.argmax(predictions)
    predicted_label = class_labels[predicted_index]

    # Display the image and the prediction details
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Predicted Label: {predicted_label}\nTrue Label: {true_label}", fontsize=12)
    
    # Format the probabilities for display
    for idx, prob in enumerate(predictions):
        label = class_labels[idx]
        plt.text(10, (idx + 1) * 20, f"{label}: {prob * 100:.1f}%", color="green", fontsize=12, ha='left')
    
    plt.show()

# Loop through each specified image and display predictions
for true_label, img_path in image_paths.items():
    print(f"Evaluating image for class: {true_label}")
    predict_and_display_image(img_path, loaded_model, class_labels, true_label)

