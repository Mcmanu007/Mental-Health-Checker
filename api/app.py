from fastapi import FastAPI
from base import Model
import joblib
import yaml
import uvicorn
from pathlib import Path
import os
import joblib

base_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(base_dir)
LOG_PATH = os.path.join(root_dir, 'pickle', 'log_model.pkl')
VECTORIZER_PATH = os.path.join(root_dir, 'pickle', 'vectorizer.pkl')

try:
    log_model = joblib.load(LOG_PATH)
    vectorizer_model = joblib.load(VECTORIZER_PATH)
except Exception as e:
    raise RuntimeError(f"Error loading log_model or vectorizer: {str(e)}")


app = FastAPI()

@app.get('/greet')
def greet_user(user: str):
    return {"message": f"Hello {user}"}

@app.post('/predict_mental_state')
def predictions(data: Model):
    text = data.text_prepocess
    vectorized = vectorizer_model.transform([text])
    
    probabilities = log_model.predict_proba(vectorized)[0].tolist()
    prediction_idx = int(log_model.predict(vectorized)[0])
    mapping = {
        0: 'Anxiety',
        1: 'Depression',
        2: 'Normal',
        3: 'Suicidal'
    }
    
    label = mapping.get(prediction_idx, "Unknown")
    
    return {
        'prediction': label,
        'probabilities': probabilities,
        'confidence_score': max(probabilities)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
