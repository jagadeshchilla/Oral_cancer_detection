import numpy as np
import streamlit as st
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
import io
import zipfile
from PIL import Image
import matplotlib.pyplot as plt
import json
import random
import os
import tempfile
import gdown

# Model Links
MODEL_LINKS = {
    'CNN': {'url': 'https://drive.google.com/uc?id=1mtDtPtM-E7y20LlFlEPn1UI20fykFL2P', 'target_size': (128, 128)},
    'ResNet50': {'url': 'https://drive.google.com/uc?id=1F47j2nr5JSa09mBWtUsWOkQYl5pLixqG', 'target_size': (260, 260)},
    'EfficientNet': {'url': 'https://drive.google.com/uc?id=10mXZQWQ1RyGx6BqEsaJdcv8NcSKv-v8A', 'target_size': (260, 260)},
    'DenseNet': {'url': 'https://drive.google.com/uc?id=14-7XGitYTJTYAksSI-LPS-b_lW7Dn8aC', 'target_size': (224, 224)},
    'VGG19': {'url': 'https://drive.google.com/uc?id=19o-JaeGBDXpITObAVkII2sNFYp3qmGoH', 'target_size': (224, 224)},
}

# Initialize session state only once
if 'saved_predictions' not in st.session_state:
    st.session_state.update({
        'saved_predictions': [],
        'predictions': [],
        'uploaded_images': [],
        'model_temp_file': None
    })

# Load existing predictions
def load_existing_predictions():
    if os.path.exists('prediction_history.json'):
        with open('prediction_history.json', 'r') as f:
            return json.load(f)
    return []

st.session_state.saved_predictions = load_existing_predictions()

# Save predictions to history
def save_predictions_to_history(uploaded_files, predictions, model_name):
    prediction_data = [
        {'file_name': file.name, 'model_used': model_name, 'prediction': 'Cancer' if pred[0] == 0 else 'Non Cancer'}
        for file, pred in zip(uploaded_files, predictions)
    ]
    st.session_state.saved_predictions.extend(prediction_data)

    with open('prediction_history.json', 'w') as f:
        json.dump(st.session_state.saved_predictions, f, indent=4)
    st.success("Predictions saved to history successfully.")

# Download and load model
def download_and_load_model(model_url):
    if st.session_state.model_temp_file is None:
        with tempfile.NamedTemporaryFile(suffix='.keras', delete=False) as tmp:
            st.session_state.model_temp_file = tmp.name
            st.toast("📥 Downloading model... Please wait.")
            with st.spinner("Downloading the model..."):
                gdown.download(model_url, st.session_state.model_temp_file, quiet=False)
            st.toast("✅ Model download completed!")
    return load_model(st.session_state.model_temp_file)

# Load uploaded images
def load_uploaded_images(uploaded_files, target_size):
    return np.array([img_to_array(load_img(file, target_size=target_size)) for file in uploaded_files])

# Evaluate model
def evaluate_model(model, images):
    predictions = model.predict(images)
    return (predictions > 0.5).astype(int)

# Main function
def show_image_prediction():
    st.title('Oral Cancer Detection Model Evaluation')

    model_selection = st.selectbox("Select a model", list(MODEL_LINKS.keys()))
    uploaded_files = st.file_uploader("Upload images", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)

    if uploaded_files:
        target_size = MODEL_LINKS[model_selection]['target_size']
        X_test = load_uploaded_images(uploaded_files, target_size)

        if st.button('Predict'):
            st.info("Downloading and loading the model. This may take a few moments...")
            model_url = MODEL_LINKS[model_selection]['url']
            model_to_use = download_and_load_model(model_url)

            st.session_state.predictions = evaluate_model(model_to_use, X_test)
            st.session_state.uploaded_images = uploaded_files
            st.toast("✨ Images predicted successfully!")

            st.subheader('Predictions:')
            for i, file in enumerate(uploaded_files):
                result = 'Cancer' if st.session_state.predictions[i][0] == 0 else 'Non Cancer'
                st.image(file, caption=f'Predicted: {result}', use_column_width=True)
                if result == 'Cancer':
                    st.warning(random.choice([
                        "Please consult a doctor immediately.",
                        "We recommend scheduling a medical check-up soon.",
                        "It's crucial to seek medical advice right away.",
                        "Contact your healthcare provider for further examination."
                    ]))

    col1, col2 = st.columns(2)
    with col1:
        if st.button('Clear'):
            st.session_state.update({'predictions': [], 'uploaded_images': [], 'model_temp_file': None})
            st.success("🗑️ Cleared all predictions and uploaded images.")

    with col2:
        if st.session_state.predictions and st.session_state.uploaded_images:
            if st.button('Save Predictions'):
                save_predictions_to_history(st.session_state.uploaded_images, st.session_state.predictions, model_selection)

    if st.session_state.predictions and st.session_state.uploaded_images:
        with io.BytesIO() as zip_buffer:
            with zipfile.ZipFile(zip_buffer, 'w') as zf:
                for i, file in enumerate(st.session_state.uploaded_images):
                    img = Image.open(file)
                    buf = io.BytesIO()
                    img.save(buf, format="PNG")
                    zf.writestr(f'prediction_{i+1}.png', buf.getvalue())
            zip_buffer.seek(0)
            st.download_button("📥 Download Predictions", data=zip_buffer, file_name='predictions.zip', mime='application/zip')

if __name__ == "__main__":
    show_image_prediction()
