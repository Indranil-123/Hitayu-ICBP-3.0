from fastapi import FastAPI,APIRouter
from pydantic import BaseModel
import joblib
import numpy as np

#deine the router
pcos_router = APIRouter()


#load model
model = joblib.load("models/pcos_model.pkl")



#deinfing input schema
class PCOSInput(BaseModel):
    age: float
    bmi: float
    menstrual_irregularity: int  
    testosterone_level: float    
    antral_follicle_count: int

#defing output schema

class PCOSOutput(BaseModel):
    prediction: str
    confidence: float


@app.post("/predict", response_model=PCOSOutput)
def predict_pcos(data: PCOSInput):

    input_data = np.array([[
        data.age,
        data.bmi,
        data.menstrual_irregularity,
        data.testosterone_level,
        data.antral_follicle_count
    ]])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]

    result = "PCOS Detected" if prediction == 1 else "No PCOS"
    confidence = probability[prediction]

    return {
        "prediction": result,
        "confidence": round(confidence, 2)
    }

