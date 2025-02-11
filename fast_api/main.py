import streamlit as st
import requests
from PIL import Image
import numpy as np
import io

# Custom CSS for styling
st.markdown(
    """
   <style>
        /* Change the background to an image */
        body {
            background-image: url(https://www.epicgardening.com/wp-content/uploads/2023/05/potato-growth-stages-1200x667.jpeg);
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
            }

        /* Title styling */
        .title {
            text-align: center;
            color: #4CAF50;
            font-size: 3em;
            font-weight: bold;
            margin-bottom: 15px;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.1);
        }

        /* Subtitle styling */
        .subtitle {
            text-align: center;
            color: #555555;
            font-size: 1.5em;
            margin-bottom: 30px;
            font-style: italic;
        }

        /* Container for the uploaded image */
        .image-container {
            display: flex;
            justify-content: center;
            margin-bottom: 25px;
        }

        /* File uploader styling */
        .stFileUploader {
            background-color: #fff;
            border-radius: 8px;
            padding: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            border: 2px solid #ddd;
        }

        /* Prediction result box */
        .result {
            font-size: 1.7em;
            text-align: center;
            color: #4CAF50;
            font-weight: bold;
            background-color: #e0ffe0;
            border-radius: 10px;
            padding: 20px;
            margin-top: 20px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        }

        /* Confidence level styling */
        .confidence {
            font-size: 1.3em;
            text-align: center;
            color: #555555;
            margin-top: 10px;
            font-style: italic;
        }

        /* Spinner styling */
        .stSpinner {
            display: flex;
            justify-content: center;
            align-items: center;
        }

        /* Button styles */
        .stButton {
            background-color: #4CAF50;
            color: white;
            font-weight: bold;
            padding: 12px 25px;
            font-size: 1.1em;
            border-radius: 5px;
            border: none;
            cursor: pointer;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }

        .stButton:hover {
            background-color: #45a049;
        }

        /* Error message styling */
        .stError {
            color: #FF6F61;
            font-size: 1.2em;
            text-align: center;
            font-weight: bold;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Title Section
with st.container():
    st.markdown('<div class="title">Potato Disease Classifier</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Upload an image of a potato leaf to predict the disease</div>', unsafe_allow_html=True)

# Upload file
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)

    # Convert image to bytes for sending to FastAPI
    image_bytes = io.BytesIO()
    image.save(image_bytes, format=image.format)
    image_bytes = image_bytes.getvalue()

    # Send the image to FastAPI server
    with st.spinner('Processing...'):
        response = requests.post("http://localhost:8000/predict", files={"file": image_bytes})

    if response.status_code == 200:
        result = response.json()
        st.markdown(f'<div class="result">Predicted Class: {result["class"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="confidence">Confidence: {result["confidence"]:.2%}</div>', unsafe_allow_html=True)
    else:
        st.error("Error in prediction. sPlease try again later.")
