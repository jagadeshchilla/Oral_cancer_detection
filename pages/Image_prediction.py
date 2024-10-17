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
import base64
import gdown
import tempfile

# Define the model links
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

# Initialize session state
if 'saved_predictions' not in st.session_state:
    st.session_state.saved_predictions = []
if 'predictions' not in st.session_state:
    st.session_state.predictions = []
if 'uploaded_images' not in st.session_state:
    st.session_state.uploaded_images = []
if 'model_temp_file' not in st.session_state:
    st.session_state.model_temp_file = None

def load_existing_predictions():
    """Loads existing predictions from a file."""
    if os.path.exists('prediction_history.json'):
        with open('prediction_history.json', 'r') as f:
            return json.load(f)
    return []

# Load existing predictions into session state
st.session_state.saved_predictions = load_existing_predictions()

def save_predictions_to_history(uploaded_files, predictions, model_name):
    """Save predictions to a JSON file for history tracking."""
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

cancer_warning_messages = [
    "Please consult a doctor immediately.",
    "We recommend scheduling a medical check-up soon.",
    "It's crucial to seek medical advice right away.",
    "Contact your healthcare provider for further examination.",
    "This result may be concerning. Please consult a specialist."
]

def download_and_load_model(model_url):
    """Downloads and loads the model from the provided Google Drive URL."""
    if st.session_state.model_temp_file is None:
        with tempfile.NamedTemporaryFile(suffix='.keras', delete=False) as tmp:
            st.session_state.model_temp_file = tmp.name
            st.toast("📥 Downloading model... Please wait.")
            with st.spinner("Downloading the model..."):
                gdown.download(model_url, st.session_state.model_temp_file, quiet=False)
            st.toast("✅ Model download completed!")

    model = load_model(st.session_state.model_temp_file)
    return model

def load_uploaded_images(uploaded_files, target_size):
    """Loads and processes uploaded images."""
    images = []
    for uploaded_file in uploaded_files:
        image = load_img(uploaded_file, target_size=target_size)
        image_array = img_to_array(image)
        images.append(image_array)
    return np.array(images)

def evaluate_model(model, images):
    """Evaluate the model on the uploaded images and return predictions."""
    predictions = model.predict(images)
    predicted_classes = (predictions > 0.5).astype(int)
    return predicted_classes

def display_predictions(uploaded_files, predictions):
    """Display the predictions along with the images."""
    for i, uploaded_file in enumerate(uploaded_files):
        actual = 'Cancer' if predictions[i][0] == 0 else 'Non Cancer'
        caption = f'Predicted: {actual}'
        st.image(uploaded_file, caption=caption, use_column_width=True)

        if actual == 'Cancer':
            warning_message = random.choice(cancer_warning_messages)
            st.warning(warning_message)

def handle_prediction_and_display(model_url, uploaded_files, target_size):
    """Handle prediction and display results for uploaded images."""
    st.info("Downloading and loading the model. This may take a few moments...")
    with st.spinner("Loading model..."):
        model = download_and_load_model(model_url)

    images = load_uploaded_images(uploaded_files, target_size)
    predictions = evaluate_model(model, images)
    st.session_state.predictions = predictions
    st.session_state.uploaded_images = uploaded_files

    st.toast("✨ Images predicted successfully!")
    st.subheader('Predictions:')
    display_predictions(uploaded_files, predictions)

def clear_predictions():
    """Clear all session state related to predictions and uploaded images."""
    st.session_state.predictions = []
    st.session_state.uploaded_images = []
    st.session_state.model_temp_file = None
    st.success("🗑️ Cleared all predictions and uploaded images.")

def image_to_base64(image: Image.Image) -> str:
    """Convert an image to base64 for displaying on Streamlit."""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def show_image_prediction():
    st.title('Oral Cancer Detection Model Evaluation')

    # Model selection
    model_selection = st.selectbox("Select a model", list(model_links.keys()))
    uploaded_files = st.file_uploader("Upload images", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)

    if uploaded_files:
        target_size = model_links[model_selection]['target_size']
        if st.button('Predict'):
            handle_prediction_and_display(model_links[model_selection]['url'], uploaded_files, target_size)

    col1, col2 = st.columns(2)
    with col1:
        if st.button('Clear'):
            clear_predictions()

    with col2:
        if st.session_state.predictions and st.session_state.uploaded_images:
            if st.button('Save Predictions'):
                save_predictions_to_history(st.session_state.uploaded_images, st.session_state.predictions, model_selection)

    # Download predictions functionality
    if st.session_state.predictions and st.session_state.uploaded_images:
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

    # Display logo
    logo_path = "./assets/logo.png"  # Update with your logo file path
    logo_image = Image.open(logo_path)
    logo_base64 = image_to_base64(logo_image)
    st.sidebar.markdown(
        f"""
        <img src="data:image/jpeg;base64,{logo_base64}"
            style="border-radius: 30px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2); width: 90%; height: auto;" />
        """, unsafe_allow_html=True
    )

if __name__ == "__main__":
    show_image_prediction()
