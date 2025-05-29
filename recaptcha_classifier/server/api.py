import numpy as np
import io
import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from .load_model import load_main_model
from recaptcha_classifier.detection_labels import DetectionLabels
from recaptcha_classifier.constants import IMAGE_SIZE

app = FastAPI()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@app.on_event("startup")
def load_models():
    """Load the models into memory at startup."""
    global model
    model = load_main_model(device)
    model.to(device)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        data = await file.read()
        img = Image.open(io.BytesIO(data)).convert("RGB")
        result = predict(model, device, img)
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )
        
    # !! check /docs, pydantic and restful API design
    
def predict(model: torch.nn.Module, device: torch.device, image: Image.Image) -> dict:
    """
    Handles the prediction logic for the uploaded image.
    """
    resized = image.resize(IMAGE_SIZE, Image.LANCZOS)
    to_array = np.array(resized).astype(np.float32) / 255.0
    to_array = np.transpose(to_array, (2, 0, 1)) 
    tensor = torch.from_numpy(to_array).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(tensor)
        id = output.argmax(dim=1).item()
        label = DetectionLabels.from_id(id)
        
    return {
        "label": label.name,
        "confidence": float(output.max().item()),
        "class_id": id
    }