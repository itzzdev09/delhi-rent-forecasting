from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import pandas as pd
from .services import predict_rent

router = APIRouter()

class RentInput(BaseModel):
    location: str
    bhk: int
    bathroom: int
    size: float
    furnishing: str
    tenant_preferred: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None

@router.post("/predict_rent")
def rent_prediction(input_data: RentInput):
    # Convert to DataFrame
    input_df = pd.DataFrame([input_data.dict()])

    try:
        pred = predict_rent(input_df)
        return {"predicted_rent": round(float(pred.iloc[0]), 2)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
