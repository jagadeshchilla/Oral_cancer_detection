import os
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
import gdown
import tempfile

# Define a dictionary of models with their respective Google Drive links and input sizes
model_info = {
    'CNN': {
        'url': 'https://drive.google.com/uc?id=1mtDtPtM-E7y20LlFlEPn1UI20fykFL2P', 
        'input_size': (128, 128)
    },
    'ResNet50': {
        'url': 'https://drive.google.com/uc?id=1F47j2nr5JSa09mBWtUsWOkQYl5pLixqG', 
        'input_size': (260, 260)
    },
    'EfficientNet': {
        'url': 'https://drive.google.com/uc?id=10mXZQWQ1RyGx6BqEsaJdcv8NcSKv-v8A', 
        'input_size': (260, 260)
    },
    'DenseNet': {
        'url': 'https://drive.google.com/uc?id=14-7XGitYTJTYAksSI-LPS-b_lW7Dn8aC', 
        'input_size': (224, 224)
    },
    'VGG19': {
        'url': 'https://drive.google.com/uc?id=19o-JaeGBDXpITObAVkII2sNFYp3qmGoH', 
        'input_size': (224, 224)
    },
}

# Function to load images and labels
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

# Function to download and load the model from Google Drive
def download_and_load_model(model_url):
    # Create a temporary file for the model
    with tempfile.NamedTemporaryFile(suffix='.keras', delete=False) as tmp:
        model_temp_file = tmp.name
        
        # Download the model and show progress
        with st.spinner("Downloading model..."):
            gdown.download(model_url, model_temp_file, quiet=False)
            st.toast("Model downloaded successfully!")

    # Load the model from the temporary file
    model = load_model(model_temp_file)
    return model

# Function to evaluate the selected model
def evaluate_model(model, test_data_path, target_size):
    # Load new unseen data
    X_test, y_test = load_data(test_data_path, target_size)

    # Make predictions
    predictions = model.predict(X_test)
    predicted_classes = (predictions > 0.5).astype(int)

    # Calculate accuracy
    accuracy = np.mean(predicted_classes.flatten() == y_test.flatten())
    st.write(f'Accuracy: {accuracy:.4f}')

    # Generate confusion matrix and classification report
    cm = confusion_matrix(y_test, predicted_classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Cancer", "Non Cancer"])

    # Plot confusion matrix
    st.subheader('Confusion Matrix')
    with st.spinner("Creating confusion matrix..."):
        fig, ax = plt.subplots()
        disp.plot(cmap=plt.cm.Blues, ax=ax)
        st.pyplot(fig)
        st.toast("Confusion matrix created!")

    # Display classification report
    report = classification_report(y_test, predicted_classes, target_names=["Cancer", "Non Cancer"])
    st.subheader("Classification Report")
    st.text(report)
    st.toast("Classification report created!")

# Streamlit app structure
def show_classify():
    st.title("Cancer Detection Model Evaluation")
    st.write("Select a model and upload your test dataset folder containing subfolders for 'CANCER' and 'NON CANCER' images.")

    # User selects the model
    selected_model_name = st.selectbox("Choose a model", list(model_info.keys()))

    # Upload folder using Streamlit's text input
    test_data_path = st.text_input("Enter the path to the test data folder:", "./test")  # Default path

    # Button to evaluate the model
    if st.button("Evaluate Model"):
        if os.path.exists(test_data_path):
            model_details = model_info[selected_model_name]
            # Download and load the selected model
            model = download_and_load_model(model_details['url'])
            # Get the input size for the selected model
            target_size = model_details['input_size']
            evaluate_model(model, test_data_path, target_size)
        else:
            st.error("The specified folder does not exist. Please check the path.")

# Main function to run the app
if __name__ == "__main__":
    show_classify()
