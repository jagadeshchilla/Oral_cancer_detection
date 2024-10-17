import os
import streamlit as st
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model
import tempfile
import gdown

def show_plots():
    st.title("Neural Network Model Architecture Visualization")

    # Define a dictionary of models with their respective Google Drive links
    model_info = {
        'CNN': {
            'url': 'https://drive.google.com/uc?id=1mtDtPtM-E7y20LlFlEPn1UI20fykFL2P',  # Example Google Drive link
        },
        'ResNet50': {
            'url': 'https://drive.google.com/uc?id=1F47j2nr5JSa09mBWtUsWOkQYl5pLixqG',
        },
        'EfficientNet': {
            'url': 'https://drive.google.com/uc?id=10mXZQWQ1RyGx6BqEsaJdcv8NcSKv-v8A',
        },
        'VGG19': {
            'url': 'https://drive.google.com/uc?id=19o-JaeGBDXpITObAVkII2sNFYp3qmGoH',
        },
        'DenseNet': {
            'url': 'https://drive.google.com/uc?id=14-7XGitYTJTYAksSI-LPS-b_lW7Dn8aC',
        }
    }

    # User selects the model
    selected_model_name = st.selectbox("Choose a model", list(model_info.keys()))

    # Button to plot the model architecture
    if st.button("Plot Model Architecture"):
        model_url = model_info[selected_model_name]['url']

        # Create a temporary file for the model
        with tempfile.NamedTemporaryFile(suffix='.keras', delete=False) as tmp:
            model_temp_file = tmp.name
        
        # Download the model
        with st.spinner("Downloading model..."):
            st.toast("Downloading model...")  # Toast for downloading
            gdown.download(model_url, model_temp_file, quiet=False)
            st.success("Model downloaded successfully!")  # Toast for successful download

        # Load the selected model
        model = load_model(model_temp_file)

        # Create a temporary file to store the plot
        with st.spinner("Plot is being created..."):
            st.toast("Creating plot...")  # Toast for plot creation
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmpfile:
                plot_model(model, to_file=tmpfile.name, show_shapes=True,
                           show_layer_names=True, dpi=300)
                tmpfile.seek(0)
                img_data = tmpfile.read()

        # Display the plot in the Streamlit interface
        caption = f'{selected_model_name} Architecture'
        st.image(img_data, caption=caption, use_column_width=True)

        # Create a download button for the plot image
        st.download_button(
            label="Download Model Architecture Plot",
            data=img_data,
            file_name=f"{selected_model_name}_architecture.png",
            mime="image/png"
        )


if __name__ == "__main__":
    show_plots()
