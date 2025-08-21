# main.py

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from pycaret.classification import load_model, predict_model

# Create a FastAPI instance
app = FastAPI()

# Load the saved model and preprocessing pipeline
model = load_model('nb_churn_model')

# Define the data model for the API request body
# This ensures incoming data has the correct structure and types
class CustomerData(BaseModel):
    SeniorCitizen: int
    tenure: int
    MonthlyCharges: float
    TotalCharges: float
    gender: str
    Partner: str
    Dependents: str
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    customerID: str
    Tenure_Group: str

# Define the prediction endpoint
@app.post("/predict")
def predict_churn(data: CustomerData):
    # Convert the incoming data to a Pandas DataFrame
    # Note: FastAPI automatically handles the Pydantic model conversion
    data_df = pd.DataFrame([data.dict()])

    # Make predictions using the loaded model
    prediction = predict_model(model, data=data_df)

    # Extract the prediction label and score
    prediction_label = prediction['prediction_label'][0]
    prediction_score = prediction['prediction_score'][0]

    # Return the prediction result as a JSON response
    return {
        "prediction_label": str(prediction_label),
        "prediction_score": float(prediction_score)
    }

# This block is for running the server directly
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)