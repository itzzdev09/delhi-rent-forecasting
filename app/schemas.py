from pydantic import BaseModel
from typing import Optional

class RentRequest(BaseModel):
    house_type: str
    house_size: str
    location: str
    latitude: Optional[float] = 0.0
    longitude: Optional[float] = 0.0
    numBathrooms: int
    numBalconies: Optional[int] = 0
    furnishing: Optional[str] = "Unfurnished"
    tenant_preferred: Optional[str] = "Family"
    # Add more fields as needed

class RentResponse(BaseModel):
    predicted_rent: float
