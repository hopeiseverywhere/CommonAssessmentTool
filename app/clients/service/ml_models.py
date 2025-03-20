class MLModels:
  current_model = "Random Forest"
  """List of available ml models"""
  available_models = ["Linear Regression", "Random Forest Regressor", "Support Vector Machine"]

  @staticmethod
  def get_current_model():
    """Get the current active ml model"""
    return MLModels.current_model 
  
  @staticmethod
  def list_available_models():
    return MLModels.available_models
  
  @staticmethod
  def switch_model(model_name: str):
    """Switch the current ml model"""
    if model_name in MLModels.available_models:
      MLModels.current_model = model_name
      return True
    return False