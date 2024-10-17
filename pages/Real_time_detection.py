import os
import numpy as np
import cv2
import streamlit as st
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import base64
import io
from PIL import Image
import tempfile
import gdown
import time

# Define the model links with target sizes
model_links = {
    'CNN': {
        'url': 'https://drive.google.com/uc?id=1mtDtPtM-E7y20LlFlEPn1UI20fykFL2P',
        'target_size': (128, 128)
    },
    'ResNet50': {
        'url': 'https://drive.google.com/uc?id=1F47j2nr5JSa09mBWtUsWOkQYl5pLixqG',
        'target_size': (260, 260)
    },
    'EfficientNet': {
        'url': 'https://drive.google.com/uc?id=10mXZQWQ1RyGx6BqEsaJdcv8NcSKv-v8A',
        'target_size': (260, 260)
    },
    'DenseNet': {
        'url': 'https://drive.google.com/uc?id=14-7XGitYTJTYAksSI-LPS-b_lW7Dn8aC',
        'target_size': (224, 224)
    },
    'VGG19': {
        'url': 'https://drive.google.com/uc?id=19o-JaeGBDXpITObAVkII2sNFYp3qmGoH',
        'target_size': (224, 224)
    },
}

# Initialize session state to hold temp file
if 'model_temp_file' not in st.session_state:
    st.session_state.model_temp_file = None

# Function to download and load model from Google Drive link
def download_and_load_model(model_url):
    if st.session_state.model_temp_file is None:
        # Create a temporary file for the model
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
            st.session_state.model_temp_file = tmp.name
            
            # Download the model and show progress
            with st.spinner("Downloading model..."):
                # Use gdown to download the model file
                gdown.download(model_url, st.session_state.model_temp_file, quiet=False)

                # Here we can simulate a progress bar
                total_size = os.path.getsize(st.session_state.model_temp_file)
                downloaded_size = 0
                chunk_size = 1024 * 1024  # 1 MB chunks

                while downloaded_size < total_size:
                    # Update the progress bar
                    time.sleep(0.1)  # Simulate download time
                    downloaded_size = min(downloaded_size + chunk_size, total_size)
                    update_progress(downloaded_size, total_size)
    
    # Load the model from the temporary file
    model = load_model(st.session_state.model_temp_file)
    return model

# Function to update the progress bar
def update_progress(current, total):
    if total > 0:
        percentage = (current / total) * 100
        st.progress(percentage)
        st.write(f"Downloaded {current} of {total} bytes ({percentage:.2f}%)")

# Function to predict a single image
def predict_image(model, image, target_size):
    image = cv2.resize(image, target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    prediction = model.predict(image)
    return (prediction > 0.5).astype(int)

# Function to show real-time detection
def show_real_time_detection():
    st.title('Real-Time Oral Cancer Detection')

    # Model selection
    model_selection = st.selectbox("Select a model", list(model_links.keys()))

    # Create layout for buttons
    col1, col2 = st.columns(2)

    # Initialize capturing state in session state
    if 'capturing' not in st.session_state:
        st.session_state.capturing = False

    with col1:
        if st.button('Start Video'):
            st.session_state.capturing = True
            cap = cv2.VideoCapture(0)

            # Create a placeholder for the video feed
            video_placeholder = st.empty()

            # Get the model URL and target size for the selected model
            model_url = model_links[model_selection]['url']
            target_size = model_links[model_selection]['target_size']

            # Download and load the selected model
            model = download_and_load_model(model_url)

            while st.session_state.capturing:
                ret, frame = cap.read()
                if not ret:
                    break

                # Make predictions
                prediction = predict_image(model, frame, target_size)
                result = 'Cancer' if prediction[0][0] == 0 else 'Non Cancer'

                # Draw the prediction on the frame
                cv2.putText(frame, f'Prediction: {result}', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

                # Convert the frame to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Display the current frame in the video placeholder
                video_placeholder.image(
                    frame_rgb, channels='RGB', use_column_width=True)

            cap.release()

    with col2:
        if st.button('Stop Video'):
            st.session_state.capturing = False

    # Utility function to convert an image to base64 for display
    def image_to_base64(image: Image.Image) -> str:
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()

    # Display logo
    logo_path = "./assets/logo.png"  # Update with your logo file path
    logo_image = Image.open(logo_path)

    # Convert the logo image to base64
    logo_base64 = image_to_base64(logo_image)

    # Display the logo with custom CSS styles
    st.sidebar.markdown(
        f"""
        <img src="data:image/jpeg;base64,{logo_base64}"
            style="border-radius: 30px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2); width: 90%; height: auto;" />
        """, unsafe_allow_html=True
    )

# Main function to run the app
if __name__ == "__main__":
    show_real_time_detection()
