import os
import joblib
from src.train import train_model

def test_model_training():
    model = train_model()
    assert model is not None
    assert os.path.exists("model.pkl")
    loaded_model = joblib.load("model.pkl")
    assert hasattr(loaded_model, "predict")
