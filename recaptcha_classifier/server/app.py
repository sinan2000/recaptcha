import streamlit as st
import torch
from typing import Literal


class StreamlitApp:
    """
    Class to render a Streamlit app for training or evaluating a model.
    """
    def __init__(self) -> None:
        self.model_type: Literal["Simple", "Base"] = "Base"
        self.lr: float = 0.001
        self.epochs: int = 20
        self.early_stopping: bool = True
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def render(self) -> None:
        st.set_page_config(page_title="Recaptcha Classifier",
                           page_icon=":guardsman:",
                           layout="centered")
        tab1, tab2 = st.tabs(["Training", "Inference"])
        
        with tab1:
            self.render_training_tab()
        with tab2:
            self.render_inference_tab()

    def render_training_tab(self) -> None:
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
            # logic here
            st.success(f"Started training {self.model_type} model with "
                       f"learning rate {self.lr}, epochs {self.epochs}, "
                       f"early stopping: {self.early_stopping}, on {self.device}.")
    
    def render_inference_tab(self) -> None:
        st.title("Inference")
        st.write("Please upload an image for inference.")

        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            # process and call api
            st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
            if st.button("Run Inference"):
                pass