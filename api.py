from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
import io
from torchvision import datasets

app = FastAPI()

# -------------------------
# LOAD DATASET JUST FOR CLASSES
# -------------------------
train_dataset = datasets.ImageFolder(
    root="../datasets/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train"
)

class_names = train_dataset.classes
num_classes = len(class_names)

print("✔ Classes loaded:", num_classes)

# -------------------------
# LOAD MODEL
# -------------------------
model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

model.load_state_dict(torch.load("../model/plant_disease_model.pth", map_location="cpu"))
model.eval()

print("✔ Model loaded")

# -------------------------
# TRANSFORM
# -------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# -------------------------
# PREDICT
# -------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    try:
        image_bytes = await file.read()

        if len(image_bytes) == 0:
            return JSONResponse(status_code=400, content={"error": "Empty file"})

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        input_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1)
            confidence, pred = torch.max(probs, 1)

        return {
            "prediction": class_names[pred.item()],
            "confidence": float(confidence.item())
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})