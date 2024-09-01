import streamlit as st
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

# Load the trained model
model = load_model('your_trained_model.h5')

# List of class names including 'healthy'
class_names = ['bacterial_leaf_blight', 'blast', 'brown_spot', 'healthy']

def load_and_preprocess_image(img):
    img = img.resize((150, 150))
    img_array = image.img_to_array(img)  # Ensure the image is converted to a NumPy array
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

# Set up the title and description
st.set_page_config(page_title="Rice Leaf Disease Detection", layout="centered")
st.title('🌾 Rice Leaf Disease Detection')
st.subheader('Upload an image of a rice leaf to identify if it is affected by bacterial leaf blight, blast, brown spot, or if it is healthy.')

# Background image using inline CSS
st.markdown("""
    <style>
    .stApp {
        background-image: url("background.jpg");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        min-height: 100vh;
    }
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #f1f1f1;
        color: black;
        text-align: center;
        padding: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar for file upload
st.sidebar.title("Upload Image")
uploaded_file = st.sidebar.file_uploader("Choose a file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_column_width=True)
    
    # Add a submit button
    if st.sidebar.button('Submit'):
        with st.spinner("Processing..."):
            img_array = load_and_preprocess_image(img)
            prediction = model.predict(img_array)
            predicted_class_index = np.argmax(prediction)
            predicted_class = class_names[predicted_class_index]
        
        st.success(f"The uploaded rice leaf is classified as: **{predicted_class}**")
else:
    st.info("Please upload an image file to get started.")

# Footer
st.markdown("""
    <div class="footer">
        Developed by Tejas Jain 
    </div>
""", unsafe_allow_html=True)
