from fastapi import FastAPI, File, UploadFile
import numpy as np
from fastapi.responses import JSONResponse
from PIL import Image
import io
import torch
from .models.load_model import load_main_model, load_simple_model
from recaptcha_classifier.detection_labels import DetectionLabels

app = FastAPI()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        """Endpoint to predict the class of an image using our model."""
        data = await file.read()
        image = Image.open(io.BytesIO(data)).convert("RGB")
        
        resized = image.resize((224, 224), Image.LANCZOS)
        # Apply transforms
        arr = np.array(resized, dtype=np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))
        tensor = torch.from_numpy(arr).unsqueeze(0).to(device)
        
        model = load_simple_model()
        model.to(device)
        with torch.no_grad():
            output = model(tensor)
            c_id = output.argmax(dim=1).item()
        
        class_name = DetectionLabels.from_id(c_id)
        
        return JSONResponse({
            "class_id": c_id,
            "class_name": class_name
        })
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )