import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO
import cv2

# Load YOLOv8 model
model = YOLO("D:\\ComputerVision\\YOLO_DET\\TumorDetection\\runs\\detect\\train\\weights\\best.pt")

# Define function for image prediction
def predict(image):
    # Convert image to numpy array
    img_array = np.array(image)
    
    # Perform inference
    results = model(img_array)
    
    # Draw bounding boxes on the image
    img_array = np.array(image)
    if results and results[0].boxes:
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0].item()
            cls = int(box.cls[0].item())
            label = model.names[cls]
            
            # Draw bounding box and label
            cv2.rectangle(img_array, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue bounding box
            cv2.putText(img_array, f'{label} {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)  # Blue text
    
    return Image.fromarray(img_array)

# Streamlit UI Styling
st.markdown("""
    <style>
    .main {
        background-color: #00bcd4; /* Cyan background */
    }
    .sidebar .sidebar-content {
        background-color: #ff5722; /* Red sidebar */
        color: #ffffff;
    }
    .stButton > button {
        background-color: #03a9f4; /* Blue button */
        color: white;
        border: none;
        border-radius: 5px;
        padding: 10px;
    }
    .stButton > button:hover {
        background-color: #0288d1; /* Darker blue on hover */
    }
    .stImage {
        border: 3px solid #ffeb3b; /* Yellow border */
        border-radius: 10px;
        padding: 5px;
        background-color: #ffffff; /* White background for images */
    }
    .stText {
        color: #ff5722; /* Red text */
        font-size: 24px;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# Title
st.title("Tumor MRI Detection")
st.markdown("<h2 class='stText'>Upload an MRI image to detect tumors.</h2>", unsafe_allow_html=True)

# File uploader
uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Load and display the image
    image = Image.open(uploaded_file)
    result_image = predict(image)
    
    # Resize images for better display
    image = image.resize((int(image.width * 0.8), int(image.height * 0.8)))
    result_image = result_image.resize((int(result_image.width * 0.8), int(result_image.height * 0.8)))
    
    # Create two columns for side-by-side display
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption="Uploaded Image", use_column_width=True)
    
    with col2:
        st.image(result_image, caption="Detected Tumors", use_column_width=True)
