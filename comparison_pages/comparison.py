import os
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
import streamlit as st
import gdown
import tempfile

# Define model URLs and target sizes
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

def show_comparison():
    # Function to load images from a directory with adjustable target size
    def load_data(folder_path, target_size):
        images = []
        labels = []
        for label, category in enumerate(['CANCER', 'NON CANCER']):
            category_folder = os.path.join(folder_path, category)
            for file_name in os.listdir(category_folder):
                image_path = os.path.join(category_folder, file_name)
                image = load_img(image_path, target_size=target_size)
                image_array = img_to_array(image)
                images.append(image_array)
                labels.append(label)
        return np.array(images), np.array(labels)

    # Function to evaluate the model on new data
    def evaluate_model(model, new_data_path, target_size):
        new_images, new_labels = load_data(new_data_path, target_size)
        predictions = model.predict(new_images)

        # Predicted classes based on a threshold of 0.5
        predicted_classes = (predictions > 0.5).astype(int)

        # Evaluate the model
        accuracy = np.mean(predicted_classes.flatten() == new_labels.flatten())

        # Initialize counters for probabilities
        total_cancer_probability = 0
        total_non_cancer_probability = 0
        total_images = len(new_images)

        # Accumulate probabilities for each class
        for i in range(total_images):
            probability = predictions[i][0]  # Probability for cancer
            total_cancer_probability += probability
            total_non_cancer_probability += (1 - probability)

        # Calculate total percentage
        average_cancer_probability = (total_cancer_probability / total_images) * 100
        average_non_cancer_probability = (total_non_cancer_probability / total_images) * 100

        return accuracy * 100, average_cancer_probability, average_non_cancer_probability

    # Function to download and load models from Google Drive
    def load_selected_model(model_name):
        model_info = model_links.get(model_name)
        if model_info:
            # Notify the user that the download is starting
            st.toast(f"Downloading {model_name} model...")

            # Create a temporary file to save the model
            with tempfile.NamedTemporaryFile(suffix='.keras', delete=False) as tmp:
                model_file_path = tmp.name

            # Download the model and show progress
            with st.spinner(f"Downloading {model_name} model..."):
                gdown.download(model_info['url'], model_file_path, quiet=False)

            # Notify the user that the download is complete
            st.toast(f"{model_name} model downloaded successfully.")

            # Load the model from the downloaded file
            model = load_model(model_file_path)
            target_size = model_info['target_size']
            return model, target_size
        else:
            st.error("Unknown model selected.")
            return None, None

    # Streamlit app UI
    st.title("Model Comparison for Cancer Detection")
    st.write("As we know, when we are training the model, a probability less than 0.5 indicates cancer, and greater than 0.5 indicates non-cancer.")

    # User selects models
    model_options = list(model_links.keys())
    selected_models = st.multiselect("Select Models to Compare", model_options)

    # Button to start evaluation
    if st.button("Evaluate Selected Models"):
        results = []

        # Loop through selected models and evaluate each
        for model_name in selected_models:
            model, target_size = load_selected_model(model_name)
            if model:
                with st.spinner(f"Evaluating {model_name} model..."):
                    accuracy, avg_cancer_prob, avg_non_cancer_prob = evaluate_model(model, "./test", target_size)
                    results.append({
                        'Model': model_name,
                        'Accuracy (%)': accuracy,
                        'Avg Cancer Probability (%)': avg_cancer_prob,
                        'Avg Non Cancer Probability (%)': avg_non_cancer_prob
                    })

        # Convert results to DataFrame and display
        if results:
            df = pd.DataFrame(results)
            st.toast("DataFrame created successfully.")
            st.write(df)

# Main function to run the app
if __name__ == "__main__":
    show_comparison()
