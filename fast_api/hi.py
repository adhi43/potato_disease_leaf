import tensorflow as tf
import numpy as np
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from io import BytesIO
import uvicorn

app = FastAPI()

CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

# Load the saved model
MODEL = tf.saved_model.load(r"C:\Users\Admin\potato_disease\saved_models\1")

# Access the model's 'serving_default' signature
infer = MODEL.signatures['serving_default']

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.get('/ping')
async def ping():
    return "I am alive"

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read the image
    image = read_file_as_image(await file.read())
    # Preprocess the image (expand dimensions to match input shape)
    img_batch = np.expand_dims(image, axis=0)

    # Run inference using the signature
    predictions = infer(tf.convert_to_tensor(img_batch, dtype=tf.float32))

    # Assuming the model outputs a tensor with the predicted class probabilities
    predicted_class = CLASS_NAMES[np.argmax(predictions['output_0'])]  # Adjust 'output_0' to the correct key
    confidence = np.max(predictions['output_0'])  # Adjust 'output_0' if needed

    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)

