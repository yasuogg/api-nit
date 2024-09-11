import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
import matplotlib.pyplot as plt

# Use Agg backend for matplotlib to work well on Mac
plt.switch_backend('TkAgg')  # Add this line for VSCode on Mac

# Load the model
def load_model(model_path):
    model = tf.saved_model.load(model_path)
    return model

def preprocess_image(image: Image.Image) -> np.ndarray:
    # Resize the image to the required input size
    image = image.resize((224, 224))  # Example size, adjust based on your model
    
    # Convert the image to a numpy array and normalize
    input_array = np.array(image) / 255.0  # Normalize to [0, 1]
    
    # Expand dimensions to match the model's input shape (batch size, height, width, channels)
    input_array = np.expand_dims(input_array, axis=0)
    
    return input_array

# Load and prepare image
def load_image(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    input_tensor = tf.convert_to_tensor(image_rgb)
    input_tensor = input_tensor[tf.newaxis, ...]
    return image_rgb, input_tensor

# Run inference on the image
def detect_objects(model, input_tensor):
    detections = model(input_tensor)
    return detections

# Visualize detection results
def visualize_detections(image, detections, detection_threshold=0.5, save=False, output_path="output_image.png"):
    height, width, _ = image.shape
    boxes = detections['detection_boxes'][0].numpy()
    scores = detections['detection_scores'][0].numpy()
    classes = detections['detection_classes'][0].numpy().astype(np.int32)

    for i in range(boxes.shape[0]):
        if scores[i] > detection_threshold:
            box = boxes[i]
            y_min, x_min, y_max, x_max = box

            (startX, startY, endX, endY) = (int(x_min * width), int(y_min * height),
                                            int(x_max * width), int(y_max * height))

            # Draw bounding box and label
            cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
            label = f"Class: {classes[i]}, Score: {scores[i]:.2f}"
            cv2.putText(image, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    if save:
        # Save the image to the output path
        cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        print(f"Image saved to {output_path}")

if __name__ == "__main__":
    # Model and image paths
    model_path = "/Users/cex/Downloads/model/saved_model"
    image_path = '/Users/cex/Downloads/IMG_3080.jpg'

    # Load model and image
    model = load_model(model_path)
    image, input_tensor = load_image(image_path)

    # Run detection
    detections = detect_objects(model, input_tensor)

    # Visualize results and save the image if needed
    visualize_detections(image, detections, save=True, output_path="output_image_with_detections.png")
