from pydantic import BaseModel
from typing import Dict, Optional


class PredictResponse(BaseModel):
    label: str
    confidence: float
    probs: Dict[str, float]
    disclaimer: str
    model: Optional[str] = None

    # NEW: base64-encoded PNG overlay (Grad-CAM)
    gradcam_overlay_png_b64: Optional[str] = None
