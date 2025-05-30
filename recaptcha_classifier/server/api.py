import numpy as np
import io
import torch
from fastapi import FastAPI, File, UploadFile, Response
from fastapi.responses import JSONResponse
import torch.nn.functional as F
from PIL import Image
from .load_model import load_main_model, get_model_path
from recaptcha_classifier.detection_labels import DetectionLabels
from recaptcha_classifier.constants import IMAGE_SIZE
from pydantic import BaseModel
import os


class PredictionResponse(BaseModel):
    label: str
    confidence: str
    class_id: int


app = FastAPI()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None

@app.on_event("startup")
def load_models():
    """Load the models into memory at startup."""
    global model
    model_path = get_model_path("main")
    if os.path.exists(model_path):
        model = load_main_model(device)
    else:
        print("Model file not found. API will return an error for predictions"
            "until model is available.")

@app.post("/predict", response_model=PredictionResponse)
async def predict(response: Response,
                  file: UploadFile = File(...)
                  ) -> PredictionResponse:
    if model is None:
        return JSONResponse(
            status_code=503,
            content={"error": "Model not loaded. Please train "
                     "or download first."}
        )
    try:
        data = await file.read()
        img = Image.open(io.BytesIO(data)).convert("RGB")
        result = inference(model, device, img)
        
        response.headers["Cache-Control"] = "max-age=3600"
        
        return result
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )
    
def inference(model: torch.nn.Module, device: torch.device, image: Image.Image) -> dict:
    """
    Handles the prediction logic for the uploaded image.
    """
    resized = image.resize(IMAGE_SIZE, Image.LANCZOS)
    to_array = np.array(resized).astype(np.float32) / 255.0
    to_array = np.transpose(to_array, (2, 0, 1)) 
    tensor = torch.from_numpy(to_array).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(tensor)
        prob = F.softmax(output, dim=1)
        conf = prob.max().item()
        id = output.argmax(dim=1).item()
        label = DetectionLabels.from_id(id)
        
    return {
        "label": label,
        "confidence": f"{conf * 100:.2f}%",
        "class_id": id
    }