from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from app.utils import predict_rent, get_dropdown_options

app = FastAPI(title="Delhi Rent Prediction API")

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

@app.get("/")
def read_root():
    return {"message": "Welcome to Delhi Rent Prediction API"}

@app.post("/predict")
def predict(input_data: RentInput):
    input_dict = input_data.dict()
    rent = predict_rent(input_dict)
    return {"predicted_rent": rent, "input": input_dict}

@app.get("/dropdowns")
def dropdowns():
    return get_dropdown_options()
