from abc import ABC, abstractmethod
from typing import List

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR


class InterfaceBaseMLModel(ABC):
    """Interface of a base ML Model"""

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray):
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        pass

    def save(self, path: str):
        import pickle

        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str):
        import pickle

        with open(path, "rb") as f:
            return pickle.load(f)

    @abstractmethod
    def __str__(self) -> str:
        """Return the name of the model"""
        pass


class LinearRegressionModel(InterfaceBaseMLModel):
    def __init__(self):
        self.model = LinearRegression()

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def __str__(self):
        return "Linear Regression"


class RandomForestModel(InterfaceBaseMLModel):
    def __init__(self, n_estimators=100, random_state=42):
        self.model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def __str__(self):
        return "Random Forest Regressor"


class SVMModel(InterfaceBaseMLModel):
    def __init__(self):
        self.model = SVR()

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def __str__(self):
        return "Support Vector Machine"


class InterfaceMLModelRepository(ABC):
    """Interface for ML Models storage"""

    @abstractmethod
    def list_models(self) -> List[InterfaceBaseMLModel]:
        """Get list of all available models instances"""
        pass

    @abstractmethod
    def is_model_available(self, model_name: str) -> bool:
        """Check if a model is valid"""
        pass

    @abstractmethod
    def get_model_instance(self, model_name: str) -> InterfaceBaseMLModel:
        """Return an instance of the requested model"""
        pass


class InterfaceMLModelManager(ABC):
    """Interface for ML model management"""

    @abstractmethod
    def get_current_model(self) -> InterfaceBaseMLModel:
        """Get the current active ml model"""
        pass

    @abstractmethod
    def switch_model(self, model_name: str) -> bool:
        """Switch between models"""
        pass


class MLModelRepository(InterfaceMLModelRepository):
    def __init__(self):
        self._model_map = {
            "Linear Regression": LinearRegressionModel,
            "Random Forest Regressor": RandomForestModel,
            "Support Vector Machine": SVMModel,
        }

    def list_models(self) -> List[InterfaceBaseMLModel]:
        return [model_class() for model_class in self._model_map.values()]

    def is_model_available(self, model_name: str) -> bool:
        return model_name in self._model_map

    def get_model_instance(self, model_name: str) -> InterfaceBaseMLModel:
        if not self.is_model_available(model_name):
            raise ValueError(f"Model '{model_name}' is not available.")
        return self._model_map[model_name]()


class MLModelManager(InterfaceMLModelManager):
    def __init__(self, repository: InterfaceMLModelRepository):
        self._repository = repository
        self._current_model = repository.get_model_instance("Random Forest Regressor")

    def get_current_model(self) -> str:
        return self._current_model

    def switch_model(self, model_name: str) -> bool:
        if self._repository.is_model_available(model_name):
            self._current_model = self._repository.get_model_instance(model_name)
            return True
        return False
