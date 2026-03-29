from fastapi import FastAPI
from base import Model
import joblib
import yaml
import uvicorn
from pathlib import Path

def resolve_path(relative_path: str) -> Path:
    BASE_DIR = Path(__file__).resolve().parents[2]
    return BASE_DIR / relative_path
# 1. Load configuration
BASE_DIR = Path(__file__).parent.resolve()
config_path = BASE_DIR / "config.yaml"

with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

log_path = resolve_path(config['saved_model']['model'])
vectorizer_path = resolve_path(config['vectorizer_model']['vec'])


log_model = joblib.load(log_path)
vectorizer_model = joblib.load(vectorizer_path)

app = FastAPI()

@app.get('/greet')
def greet_user(user: str):
    return {"message": f"Hello {user}"}

@app.post('/predict_mental_state')
def predictions(data: Model):
    text = data.text_prepocess
    vectorized = vectorizer_model.transform([text])
    prediction_idx = int(log_model.predict(vectorized)[0])
    
    probabilities = log_model.predict_proba(vectorized)[0].tolist()
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
