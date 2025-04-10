import numpy as np
from fastapi import APIRouter, HTTPException

from app.clients.service.ml_models import MLModelManager, MLModelRepository, \
    InterfaceBaseMLModel
from app.clients.service.models import PredictionFeatures, PredictionRequest

router = APIRouter(prefix="/ml_models", tags=["model"])
model_repository = MLModelRepository()
model_manager = MLModelManager(model_repository)


@router.get("/list")
def list_models():
    """List all available ML models"""
    # return {"models": model_repository.list_models()}
    return {"models": [str(model) for model in model_repository.list_models()]}


@router.post("/switch/{model_name}")
def switch_models(model_name: str):
    """Switch between ML models"""
    success = model_manager.switch_model(model_name)
    if not success:
        raise HTTPException(status_code=400, detail="Model switch failed")
    return {"message": f"Model switched to {model_name}"}


@router.get("/current")
def current_model():
    """Get the current ML model"""
    # return {"current_model": model_manager.get_current_model()}
    return {"current_model": str(model_manager.get_current_model())}


@router.post("/predict/{model_name}")
def predict_with_model_name(features: PredictionFeatures, model_name: str):
    """Predict based on a given ML model name"""
    model = model_repository.get_model_instance(model_name)
    # model.load_if_trained()
    return predict_model(model, features)


@router.post("/predict")
def predict_with_current_model(features: PredictionFeatures):
    """Predict based on current ML model"""
    model = model_manager.get_current_model()
    # model.load_if_trained()
    return predict_model(model, features)


def predict_model(model: InterfaceBaseMLModel, features: PredictionFeatures):
    """Predict based on given ML model"""
    model.load_if_trained()
    prediction_request = PredictionRequest.from_structured_features(features)
    try:
        prediction = model.predict(np.array([prediction_request.features]))
        return {
            "model": str(model),
            "input": prediction_request.features,
            "prediction": prediction.tolist(),
        }
    except Exception as e:
        raise HTTPException(status_code=500,
                            detail=f"Prediction failed: {str(e)}") from e
