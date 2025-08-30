from fastapi import FastAPI
from .routes import router

app = FastAPI(title="Delhi Rent Prediction Microservice")

# Include router
app.include_router(router)
