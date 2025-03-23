from abc import ABC, abstractmethod
from typing import List


class InterfaceMLModelRepository(ABC):
    """Interface for ML Models storage"""

    @abstractmethod
    def list_models(self) -> List[str]:
        """Get list of all available models"""
        pass

    @abstractmethod
    def is_model_available(self, model_name: str) -> bool:
        """Check if a model is valid"""
        pass


class InterfaceMLModelManager(ABC):
    """Interface for ML model management"""

    @abstractmethod
    def get_current_model(self) -> str:
        """Get the current active ml model"""
        pass

    @abstractmethod
    def switch_model(self, model_name: str) -> bool:
        """Switch between models"""
        pass


class MLModelRepository(InterfaceMLModelRepository):
    def __init__(self):
        self._available_models = [
            "Linear Regression",
            "Random Forest Regressor",
            "Support Vector Machine"
        ]

    def list_models(self) -> List[str]:
        return self._available_models

    def is_model_available(self, model_name: str) -> bool:
        return model_name in self._available_models

class MLModelManager(InterfaceMLModelManager):
    def __init__(self, repository: InterfaceMLModelRepository):
        self._repository = repository
        self._current_model = "Random Forest Regressor"

    def get_current_model(self) -> str:
        return self._current_model

    def switch_model(self, model_name: str) -> bool:
        if self._repository.is_model_available(model_name):
            self._current_model = model_name
            return True
        return False

