from fastapi import FastAPI, File, UploadFile
from starlette.responses import JSONResponse
import io
import tensorflow as tf
from PIL import Image
import numpy as np

# Initialize FastAPI
app = FastAPI()

# Load the pre-trained TensorFlow model
model = tf.keras.models.load_model("your_model_path")

# Define labels for classification
class_labels = ['Healthy', 'Mosaic', 'RedRot', 'Rust', 'Yellow']  

# Define function to preprocess image
def preprocess_image(image):
    img = image.resize((224, 224))  # Resize image to match model input size
    img = np.array(img) / 255.0  # Normalize pixel values to [0, 1]
    img = img.reshape((-1, 224, 224, 3))  # Reshape image to match model input shape
    return img

# Define endpoint for image classification
@app.post("/classify/")
async def classify_image(file: UploadFile = File(...)):
    try:
        # Read image file from request
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Preprocess image
        preprocessed_image = preprocess_image(image)
        
        # Perform inference
        predictions = model.predict(preprocessed_image)
        predicted_class_index = np.argmax(predictions[0])
        predicted_class = class_labels[predicted_class_index]
        
        return JSONResponse(content={"class": predicted_class, "confidence": float(predictions[0][predicted_class_index])})
    
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)