import os
import streamlit as st
from tensorflow.keras.models import load_model
import gdown
import tempfile

# Define a function to show model summaries
def show_summary():
    # Define available models and their Google Drive links
    model_info = {
        'CNN': 'https://drive.google.com/uc?id=1mtDtPtM-E7y20LlFlEPn1UI20fykFL2P',
        'ResNet50': 'https://drive.google.com/uc?id=1F47j2nr5JSa09mBWtUsWOkQYl5pLixqG',
        'EfficientNet': 'https://drive.google.com/uc?id=10mXZQWQ1RyGx6BqEsaJdcv8NcSKv-v8A',
        'VGG19': 'https://drive.google.com/uc?id=19o-JaeGBDXpITObAVkII2sNFYp3qmGoH',
        'DenseNet': 'https://drive.google.com/uc?id=14-7XGitYTJTYAksSI-LPS-b_lW7Dn8aC'
    }

    # Streamlit page title
    st.title("Model Summary Page")

    # User selects a model from the dropdown
    selected_model_name = st.selectbox(
        "Choose a model to summarize", list(model_info.keys()))

    # Load the selected model when the button is clicked
    if st.button("Summarize Model"):
        model_url = model_info[selected_model_name]

        # Notify the user that the download is starting
        st.toast("Downloading model...")

        with st.spinner("Downloading model..."):
            # Create a temporary file for the model
            with tempfile.NamedTemporaryFile(suffix='.keras', delete=False) as tmp:
                model_temp_file = tmp.name
                
                # Download the model and show progress
                gdown.download(model_url, model_temp_file, quiet=False)
        
        st.toast("Downloaded successfully!")  # Toast for successful download

        # Load the model
        model = load_model(model_temp_file)
        st.write(f"Showing summary for the {selected_model_name} model:")

        # Display the model's summary
        summary_str = []
        # Capture summary in list
        model.summary(print_fn=lambda x: summary_str.append(x))
        st.text("\n".join(summary_str))  # Display summary in Streamlit

        st.toast("Summary is created!")  # Toast for summary creation

# Main function to display the page
def main():
    st.sidebar.title("Navigation")
    options = st.sidebar.radio(
        "Go to", ["Model Summary", "Model Classification"])

    if options == "Model Summary":
        show_summary()

if __name__ == "__main__":
    main()
