from tensorflow.keras.models import load_model
import cv2
import numpy as np
import os

model = load_model('model.h5') #add model path

def preprocess_image(image_path, target_size=(150, 150)): 
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, target_size)  
    img = img / 255.0  
    img = np.expand_dims(img, axis=-1)  
    img = np.expand_dims(img, axis=0)  
    return img

image_path = "image_path" #add image path
processed_img = preprocess_image(image_path)
prediction = model.predict(processed_img)

if prediction[0] > 0.5:
    print(f"Predicted Class: Normal with confidence {prediction[0][0] * 100:.2f}%")
else:
    print(f"Predicted Class: Pneumoniac with confidence {(1 - prediction[0][0]) * 100:.2f}%")

