import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
import io
import zipfile
from PIL import Image
import matplotlib.pyplot as plt
import json
import random
import os
import base64
import gdown
import tempfile

# Ensure all session state variables are initialized at the beginning
if 'saved_predictions' not in st.session_state:
    st.session_state.saved_predictions = []
if 'predictions' not in st.session_state:
    st.session_state.predictions = []
if 'uploaded_images' not in st.session_state:
    st.session_state.uploaded_images = []
if 'model_temp_file' not in st.session_state:
    st.session_state.model_temp_file = None
if 'Predict' not in st.session_state:
    st.session_state.Predict = False

# Define the model links and their target sizes
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

# Function to load existing predictions
def load_existing_predictions():
    if os.path.exists('prediction_history.json'):
        with open('prediction_history.json', 'r') as f:
            return json.load(f)
    return []

# Load predictions into session state at the start
st.session_state.saved_predictions = load_existing_predictions()

# Function to save predictions to history
def save_predictions_to_history(uploaded_files, predictions, model_name):
    prediction_data = []
    for i, uploaded_file in enumerate(uploaded_files):
        actual = 'Cancer' if predictions[i][0] == 0 else 'Non Cancer'
        prediction_data.append({
            'file_name': uploaded_file.name,
            'model_used': model_name,
            'prediction': actual
        })

    st.session_state.saved_predictions.extend(prediction_data)

    with open('prediction_history.json', 'w') as f:
        json.dump(st.session_state.saved_predictions, f, indent=4)
    st.success("Predictions saved to history successfully.")

# Cancer warning messages
cancer_warning_messages = [
    "Please consult a doctor immediately.",
    "We recommend scheduling a medical check-up soon.",
    "It's crucial to seek medical advice right away.",
    "Contact your healthcare provider for further examination.",
    "This result may be concerning. Please consult a specialist."
]

# Function to download and load the model from a Google Drive link
def download_and_load_model(model_url):
    """Downloads and loads the model from the provided Google Drive URL."""
    if st.session_state.model_temp_file is None:
        with tempfile.NamedTemporaryFile(suffix='.keras', delete=False) as tmp:
            st.session_state.model_temp_file = tmp.name
            st.toast("📥 Downloading model... Please wait.")
            with st.spinner("Downloading the model..."):
                gdown.download(model_url, st.session_state.model_temp_file, quiet=False)
            st.toast("✅ Model download completed!")

    return load_model(st.session_state.model_temp_file)

# Function to predict images
def show_image_prediction():
    st.title('Oral Cancer Detection Model Evaluation')

    # Model selection
    model_selection = st.selectbox("Select a model", list(model_links.keys()))

    # Upload images
    uploaded_files = st.file_uploader(
        "Upload images", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)

    if uploaded_files:
        target_size = model_links[model_selection]['target_size']

        def load_uploaded_images(uploaded_files, target_size):
            images = []
            for uploaded_file in uploaded_files:
                image = load_img(uploaded_file, target_size=target_size)
                image_array = img_to_array(image)
                images.append(image_array)
            return np.array(images)

        X_test = load_uploaded_images(uploaded_files, target_size)

        # Function to evaluate the model
        def evaluate_model(model, images):
            predictions = model.predict(images)
            return (predictions > 0.5).astype(int)

        # Add a button to trigger predictions
        if st.button('Predict'):
            st.session_state.Predict = True

        if st.session_state.Predict:
            st.info("Downloading and loading the model. This may take a few moments...")

            model_url = model_links[model_selection]['url']
            with st.spinner("Loading model..."):
                model_to_use = download_and_load_model(model_url)

            with st.spinner("Evaluating images..."):
                st.session_state.predictions = evaluate_model(model_to_use, X_test)
                st.session_state.uploaded_images = uploaded_files

            st.toast("✨ Images predicted successfully!")

            # Display predictions
            st.subheader('Predictions:')
            for i, uploaded_file in enumerate(uploaded_files):
                actual = 'Cancer' if st.session_state.predictions[i][0] == 0 else 'Non Cancer'
                caption = f'Predicted: {actual}'
                st.image(uploaded_file, caption=caption, use_column_width=True)

                if actual == 'Cancer':
                    warning_message = random.choice(cancer_warning_messages)
                    st.warning(warning_message)

    # Column buttons for clearing or saving predictions
    col1, col2 = st.columns(2)

    with col1:
        if st.button('Clear'):
            st.session_state.predictions = []
            st.session_state.uploaded_images = []
            st.session_state.model_temp_file = None  # Reset the temp file
            st.success("🗑️ Cleared all predictions and uploaded images.")

    with col2:
        if st.session_state.get('predictions') and st.session_state.get('uploaded_images'):
            if st.button('Save Predictions'):
                save_predictions_to_history(
                    st.session_state.uploaded_images, st.session_state.predictions, model_selection)

    # Download predictions functionality
    if st.session_state.get('predictions') and st.session_state.get('uploaded_images'):
        prediction_images = []
        for i, uploaded_file in enumerate(st.session_state.uploaded_images):
            actual = 'Cancer' if st.session_state.predictions[i][0] == 0 else 'Non Cancer'
            image = Image.open(uploaded_file)
            pred_image = image.copy()
            plt.imshow(pred_image)
            plt.axis('off')
            plt.title(f'Predicted: {actual}')
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
            plt.close()
            buf.seek(0)
            prediction_images.append((buf, f'prediction_{i + 1}.png'))

        # Create zip file for download
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w') as zf:
            for image_buf, filename in prediction_images:
                zf.writestr(filename, image_buf.getvalue())
        zip_buffer.seek(0)

        st.download_button(
            label="📥 Download Predictions",
            data=zip_buffer,
            file_name='predictions.zip',
            mime='application/zip'
        )

# Call the function to show image prediction
if __name__ == "__main__":
    show_image_prediction()
