import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import json

# dimensions of our images
img_width, img_height = 224, 224

# load the model we saved
model = load_model('coffeeModel.h5')
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Load the class indices
with open('class_indices_coffee.json', 'r') as f:
    class_indices = json.load(f)
class_indices = {int(k): v for k, v in class_indices.items()}

# Function to preprocess the image
def preprocess_image(img_path, target_size):
    img = image.load_img(img_path, target_size=target_size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0  # Scale the pixel values to [0, 1]
    return x

# Function to predict the class name of a single image
def predict_single_image_class(model, img_path, class_indices):
    x = preprocess_image(img_path, (img_width, img_height))
    predictions = model.predict(x)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[predicted_class_index]
    return predicted_class_name



# Predicting a single image
img_path = '/Users/surajmeharwade/PlantDiseasePredictor/test/rust/1122.jpg'
predicted_class_name = predict_single_image_class(model, img_path, class_indices)
print(f"Predicted class for {img_path}: {predicted_class_name}")

# Predicting multiple images
# img_paths = [
#     '/Users/surajmeharwade/PlantDiseasePredictor/test_potato_early_blight.jpg',
#     '/Users/surajmeharwade/PlantDiseasePredictor/test_potato.jpg'
# ]
# predicted_class_names = predict_multiple_image_classes(model, img_paths, class_indices)
# for img_path, class_name in zip(img_paths, predicted_class_names):
#     print(f"Predicted class for {img_path}: {class_name}")


# Function to predict the class names of multiple images
# def predict_multiple_image_classes(model, img_paths, class_indices):
#     images = [preprocess_image(img_path, (img_width, img_height)) for img_path in img_paths]
#     images = np.vstack(images)
#     predictions = model.predict(images)
#     predicted_class_indices = np.argmax(predictions, axis=1)
#     predicted_class_names = [class_indices[index] for index in predicted_class_indices]
#     return predicted_class_names