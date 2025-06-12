import streamlit as st
import torch
from typing import Literal
import requests


class StreamlitApp:
    """
    Class to render a Streamlit app for training or evaluating a model.
    """
    def __init__(self) -> None:
        """
        Constructor for StreamlitApp class.
        """
        self.model_type: Literal["Simple", "Base"] = "Base"
        self.lr: float = 0.001
        self.epochs: int = 20
        self.early_stopping: bool = True
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def render(self) -> None:
        """Render the Streamlit app. """
        st.set_page_config(page_title="Recaptcha Classifier",
                           page_icon=":guardsman:",
                           layout="centered")
        tab1, tab2 = st.tabs(["Training", "Inference"])

        with tab1:
            self.render_training_tab()
        with tab2:
            self.render_inference_tab()

    def render_training_tab(self) -> None:
        """ Render the training tab. """
        st.title("Training")
        st.write("Please select the model type and training parameters.")

        self.model_type = st.selectbox("Select Model", ["Simple", "Base"],
                                       index=1)
        self.lr = st.number_input("Learning Rate", min_value=0.0001,
                                  max_value=1.0, value=0.001,
                                  step=0.0001, format="%.4f")
        self.epochs = st.number_input("Number of Epochs", min_value=1,
                                      max_value=100, value=20, step=1)
        self.early_stopping = st.checkbox("Early Stopping", value=True)

        if torch.cuda.is_available():
            self.device = st.radio("Device", ["cuda", "cpu"], index=0)
        else:
            st.radio("Device", ["cpu"], index=0, disabled=True,
                     help="CUDA is not available on this machine.")

        if st.button("Start Training"):
            if self.model_type == "Simple":
                from recaptcha_classifier.pipeline.simple_cnn_pipeline import (
                    SimpleClassifierPipeline)
                pipeline = SimpleClassifierPipeline(
                    lr=self.lr,
                    epochs=self.epochs,
                    early_stopping=self.early_stopping,
                    device=self.device)
                pipeline.run()
            else:
                from recaptcha_classifier.pipeline.main_model_pipeline import (
                    MainClassifierPipeline)
                pipeline = MainClassifierPipeline(
                    lr=self.lr,
                    epochs=self.epochs,
                    early_stopping=self.early_stopping,
                    device=self.device)
                pipeline.run()
            st.success(f"Started training {self.model_type} model with "
                       f"learning rate {self.lr}, epochs {self.epochs}, "
                       f"early stopping: {self.early_stopping}, on {
                        self.device}.")

    def render_inference_tab(self) -> None:
        st.title("Inference")
        st.write("Please upload an image for inference.")

        file = st.file_uploader("Choose an image...", type=[
            "jpg", "jpeg", "png"])

        if file is not None:
            st.image(file, caption='Uploaded Image.', use_container_width=True)
            if st.button("Run Inference"):
                files = {"file": (file.name, file.getvalue(), file.type)}

                try:
                    resp = requests.post(
                        "http://127.0.0.1:8000/predict", files=files)
                    resp.raise_for_status()
                    result = resp.json()

                    st.success("Prediction successful!")
                    st.write(f"Label: {result['label']}")
                    st.write(f"Confidence: {result['confidence']}")
                    st.write(f"Class ID: {result['class_id']}")
                except Exception as e:
                    st.error(f"Error during inference: {str(e)}")


if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "..")))
    app = StreamlitApp()
    app.render()
