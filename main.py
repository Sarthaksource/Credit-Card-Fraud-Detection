from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import pandas as pd
import numpy as np
import pickle
import os
import shutil
from werkzeug.utils import secure_filename
from fastapi.staticfiles import StaticFiles

class ManualPredictionRequest(BaseModel):
    time: float
    v1: float
    v2: float
    v3: float
    v4: float
    v5: float
    v6: float
    v7: float
    v8: float
    v9: float
    v10: float
    v11: float
    v12: float
    v13: float
    v14: float
    v15: float
    v16: float
    v17: float
    v18: float
    v19: float
    v20: float
    v21: float
    v22: float
    v23: float
    v24: float
    v25: float
    v26: float
    v27: float
    v28: float
    amount: float

# Initialize FastAPI app
app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

# Configuration for file uploads
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the trained model and scaler on startup
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    print("Model and scaler loaded successfully!")
except FileNotFoundError:
    print("ERROR: Model files not found. Ensure 'model.pkl' and 'scaler.pkl' are in the same directory.")
    model = None
    scaler = None

EXPECTED_COLUMNS = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']

@app.get("/", include_in_schema=False)
def home():
    return FileResponse('templates/index.html')

@app.post("/predict_manual")
def predict_manual(data: ManualPredictionRequest):
    """
    Predicts fraud for a single transaction entered manually.
    Receives data that conforms to the ManualPredictionRequest model.
    """
    if not model or not scaler:
        raise HTTPException(status_code=503, detail="Model is not available.")

    try:
        # Pydantic model is automatically converted to a dict
        feature_dict = data.dict()
        features = [feature_dict[col.lower()] for col in EXPECTED_COLUMNS] # Ensure order is correct

        # Convert to numpy array, scale, and predict
        features_array = np.array(features).reshape(1, -1)
        features_scaled = scaler.transform(features_array)
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0]

        # Return the result
        return {
            'prediction': 'Fraud' if int(prediction) == 1 else 'Normal',
            'confidence': float(max(probability)) * 100,
            'fraud_probability': float(probability[1]) * 100,
            'normal_probability': float(probability[0]) * 100
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


@app.post("/predict_csv")
def predict_csv(file: UploadFile = File(...)):
    """
    Predicts fraud for all transactions in an uploaded CSV file.
    """
    if not model or not scaler:
        raise HTTPException(status_code=503, detail="Model is not available.")

    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a CSV file.")

    # Save the uploaded file temporarily
    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    
    try:
        with open(filepath, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Read and validate the CSV
        df = pd.read_csv(filepath)
        missing_cols = [col for col in EXPECTED_COLUMNS if col not in df.columns]
        if missing_cols:
            raise HTTPException(status_code=400, detail=f"CSV is missing required columns: {missing_cols}")
            
        df_features = df[EXPECTED_COLUMNS]
        if df_features.isnull().any().any():
            raise HTTPException(status_code=400, detail="CSV contains missing values. Please check the file.")
        
        # Scale features and make predictions
        features_scaled = scaler.transform(df_features)
        predictions = model.predict(features_scaled)
        probabilities = model.predict_proba(features_scaled)
        
        # Format results
        results = [
            {
                'row': i + 1,
                'prediction': 'Fraud' if int(pred) == 1 else 'Normal',
                'fraud_probability': float(prob[1]) * 100
            }
            for i, (pred, prob) in enumerate(zip(predictions, probabilities))
        ]

        total_transactions = len(predictions)
        fraud_count = int(np.sum(predictions))
        
        summary = {
            'total_transactions': total_transactions,
            'fraud_detected': fraud_count,
            'normal_transactions': total_transactions - fraud_count,
            'fraud_percentage': (fraud_count / total_transactions) * 100 if total_transactions > 0 else 0
        }

        return {
            'results': results,
            'summary': summary
        }
    
    except HTTPException as e:
        # Re-raise HTTP exceptions to send proper error responses
        raise e
    except Exception as e:
        # Catch any other unexpected errors
        raise HTTPException(status_code=500, detail=f"An error occurred processing the file: {str(e)}")
    
    finally:
        # Ensure the uploaded file is always removed
        if os.path.exists(filepath):
            os.remove(filepath)


@app.get("/health")
def health_check():
    """Health check endpoint to verify the application and models are running."""
    return {
        'status': 'running',
        'model_status': 'loaded' if model else 'not loaded',
        'scaler_status': 'loaded' if scaler else 'not loaded'
    }

# --- 3. Running the App ---
if __name__ == '__main__':
    import uvicorn
    # Use reload=True for development to auto-reload server on code changes
    uvicorn.run("main:app", host='0.0.0.0', port=5000, reload=True)