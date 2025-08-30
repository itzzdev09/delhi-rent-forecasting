from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from app.utils import predict_rent, get_dropdown_options

app = FastAPI(title="Delhi Rent Prediction API")

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to Delhi Rent Prediction API"}

# Predict endpoint
class RentInput(BaseModel):
    house_type: str
    house_size: str
    location: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    numBathrooms: Optional[int] = None
    numBalconies: Optional[int] = None
    furnishing: Optional[str] = None
    tenant_preferred: Optional[str] = None

@app.post("/predict")
def predict_rent_endpoint(data: RentInput):
    input_data = data.dict()
    try:
        rent = predict_rent(input_data)
        return {"predicted_rent": rent}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Dropdown options endpoint
@app.get("/dropdowns")
def get_dropdowns():
    options = get_dropdown_options()
    return options
