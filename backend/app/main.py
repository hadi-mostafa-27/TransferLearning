from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import os

from app.schema import PredictResponse
from app.inference import InferenceService



APP_DISCLAIMER = "Educational demo only. Not a medical device. Do not use for clinical decisions."

app = FastAPI(title="MedXfer API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

WEIGHTS_PATH = os.getenv("MODEL_WEIGHTS", "model/model.pt")
META_PATH = os.getenv("MODEL_META", "model/model_meta.json")
svc = InferenceService(weights_path=WEIGHTS_PATH, meta_path=META_PATH)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model-info")
def model_info():
    return {
        "disclaimer": APP_DISCLAIMER,
        "meta": svc.meta,
        "device": svc.device,
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    content = await file.read()
    img = Image.open(io.BytesIO(content))

    label, confidence, probs = svc.predict(img)

    return PredictResponse(
        label=label,
        confidence=confidence,
        probs=probs,
        disclaimer=APP_DISCLAIMER,
        model=svc.meta.get("backbone"),
    )
