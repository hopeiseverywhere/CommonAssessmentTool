from fastapi import APIRouter, HTTPException
from app.clients.service.ml_models import MLModels

router = APIRouter(prefix="/ml_models")

@router.get("/list")
def list_models():
  models = MLModels.list_available_models()
  return {"models": models}

@router.post("/switch/{model_name}")
def switch_models(model_name: str):
  success = MLModels.switch_model(model_name)
  if not success:
    raise HTTPException(status_code=400, detail="Model switch failed")
  return {"message": f"Model switched to {model_name}"}

@router.get("/current")
def current_model():
  model = MLModels.get_current_model()
  return {"current_model": model}
