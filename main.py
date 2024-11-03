from http.client import HTTPResponse
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from io import BytesIO
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
import os

app = FastAPI()

@app.get("/", response_class=HTMLResponse)
async def main_page():
    return """
    <html>
        <head>
            <title>Image Upload</title>
        </head>
        <body>
            <h1>Upload a bottle :)</h1>
            <form action="/upload-image/" enctype="multipart/form-data" method="post">
                <input name="file" type="file">
                <input type="submit">
            </form>
        </body>
    </html>
    """

# Global variable for the model
model = None
model_path = "./saved_model"

# Load the model at startup
@app.on_event("startup")
async def load_model_on_startup():
    global model
    model = tf.saved_model.load(model_path)



# Function to detect objects and process the image
def detect_and_process_image(model, image: Image.Image, detection_threshold=0.5):
    # Convert PIL image to NumPy array (RGB format)
    image_rgb = np.array(image)

    # Convert to tensor for TensorFlow model
    input_tensor = tf.convert_to_tensor(image_rgb)
    input_tensor = input_tensor[tf.newaxis, ...]  # Add batch dimension

    # Run object detection
    detections = model(input_tensor)

    # Extract detection results
    height, width, _ = image_rgb.shape
    boxes = detections['detection_boxes'][0].numpy()
    scores = detections['detection_scores'][0].numpy()
    classes = detections['detection_classes'][0].numpy().astype(np.int32)

    # Draw bounding boxes on the image if scores exceed threshold
    for i in range(boxes.shape[0]):
        if scores[i] > detection_threshold:
            box = boxes[i]
            y_min, x_min, y_max, x_max = box

            (startX, startY, endX, endY) = (int(x_min * width), int(y_min * height),
                                            int(x_max * width), int(y_max * height))

            # Draw bounding box and label
            cv2.rectangle(image_rgb, (startX, startY), (endX, endY), (0, 255, 0), 2)
            label = f"Class: {classes[i]}, Score: {scores[i]:.2f}"
            cv2.putText(image_rgb, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Convert processed NumPy array back to PIL image
    processed_image = Image.fromarray(image_rgb)

    return processed_image

# Endpoint to handle image uploads and return processed images
@app.post("/upload-image/")
async def upload_image(file: UploadFile = File(...)):
    try:
        # Read the uploaded image
        contents = await file.read()

        # Convert byte contents to PIL Image
        image = Image.open(BytesIO(contents)).convert("RGB")

        # Run object detection and get the processed image
        processed_image = detect_and_process_image(model, image)

        # Convert the processed image to a BytesIO object to send as response
        buffer = BytesIO()
        processed_image.save(buffer, format="PNG")
        buffer.seek(0)

        # Return the processed image as a PNG
        return StreamingResponse(buffer, media_type="image/png")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Example FastAPI app run
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
