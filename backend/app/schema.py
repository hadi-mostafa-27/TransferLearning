from pydantic import BaseModel
from typing import Dict, Optional


class PredictResponse(BaseModel):
    label: str
    confidence: float
    probs: Dict[str, float]
    disclaimer: str
    model: Optional[str] = None
