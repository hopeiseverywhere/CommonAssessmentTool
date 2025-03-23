from fastapi import APIRouter, HTTPException
from app.clients.service.ml_models import MLModelRepository, MLModelManager

router = APIRouter(prefix="/ml_models")
model_repository = MLModelRepository()
model_manager = MLModelManager(model_repository)
@router.get("/list")
def list_models():
    """List all available ML models"""
    return {"models": model_repository.list_models()}


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
    return {"current_model": model_manager.get_current_model()}
