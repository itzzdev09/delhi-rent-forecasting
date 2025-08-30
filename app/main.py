from fastapi import FastAPI
from app.schemas import RentRequest, RentResponse
from app.utils import predict_rent

app = FastAPI(title="Delhi Rent Prediction API")

@app.get("/")
def root():
    return {"message": "Welcome to Delhi Rent Prediction API"}

@app.post("/predict", response_model=RentResponse)
def predict_rent_endpoint(request: RentRequest):
    input_data = request.dict()
    rent = predict_rent(input_data)
    return RentResponse(predicted_rent=rent)
