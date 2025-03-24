"""
Model training module for the Common Assessment Tool.
Handles the preparation, training, and saving of the prediction model.
Pass in model name via command line
"""

import os

# Standard library imports
import pickle
import sys

# Third-party imports
import numpy as np
import pandas as pd

# Local imports
from ml_models import (
    InterfaceBaseMLModel,
    LinearRegressionModel,
    MLModelRepository,
    RandomForestModel,
    SVMModel,
)
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

repo = MLModelRepository()

default_unformatted_model_path = "pretrained_models" + os.sep + "model_{}.pkl"


def get_model_by_name(model_type: str, n_estimators=100, random_state=42) -> InterfaceBaseMLModel:
    model_map = {
        "Linear Regression": LinearRegressionModel,
        "Random Forest Regressor": lambda: RandomForestModel(n_estimators, random_state),
        "Support Vector Machine": SVMModel,
    }

    if model_type not in model_map:
        print(f"ERROR! Invalid model type '{model_type}' passed in.")
        print(f"Available models: {repo.list_models()}")
        sys.exit(-1)

    constructor = model_map[model_type]
    return constructor() if callable(constructor) else constructor()


def prepare_model_data(test_size=0.2, random_state=42):
    """
    Prepare and train the Random Forest model using the dataset.
    Args:
        test_size: The percent of the dataset to use as test data (rest will be used as train data)
        random_state: The random state to generate train/test split with

    Returns:
        RandomForestRegressor: Trained model for predicting success rates
    """
    # Load dataset
    data = pd.read_csv("data_commontool.csv")
    # Define feature columns
    feature_columns = [
        "age",  # Client's age
        "gender",  # Client's gender (bool)
        "work_experience",  # Years of work experience
        "canada_workex",  # Years of work experience in Canada
        "dep_num",  # Number of dependents
        "canada_born",  # Born in Canada
        "citizen_status",  # Citizenship status
        "level_of_schooling",  # Highest level achieved (1-14)
        "fluent_english",  # English fluency scale (1-10)
        "reading_english_scale",  # Reading ability scale (1-10)
        "speaking_english_scale",  # Speaking ability scale (1-10)
        "writing_english_scale",  # Writing ability scale (1-10)
        "numeracy_scale",  # Numeracy ability scale (1-10)
        "computer_scale",  # Computer proficiency scale (1-10)
        "transportation_bool",  # Needs transportation support (bool)
        "caregiver_bool",  # Is primary caregiver (bool)
        "housing",  # Housing situation (1-10)
        "income_source",  # Source of income (1-10)
        "felony_bool",  # Has a felony (bool)
        "attending_school",  # Currently a student (bool)
        "currently_employed",  # Currently employed (bool)
        "substance_use",  # Substance use disorder (bool)
        "time_unemployed",  # Years unemployed
        "need_mental_health_support_bool",  # Needs mental health support (bool)
    ]
    # Define intervention columns
    intervention_columns = [
        "employment_assistance",
        "life_stabilization",
        "retention_services",
        "specialized_services",
        "employment_related_financial_supports",
        "employer_financial_supports",
        "enhanced_referrals",
    ]
    # Combine all feature columns
    all_features = feature_columns + intervention_columns
    # Prepare training data
    features = np.array(data[all_features])  # Changed from X to features
    targets = np.array(data["success_rate"])  # Changed from y to targets
    # Split the dataset
    X_train, x_test, Y_train, y_test = train_test_split(
        # Removed unused variables
        features,
        targets,
        test_size=test_size,
        random_state=random_state,
    )

    return X_train, x_test, Y_train, y_test


def train_model(
    X_train, Y_train, model_type, n_estimators=100, random_state=42
) -> InterfaceBaseMLModel:
    """
    Trains the model
    Args:
        X_train: Training features
        targets_train:  Target features
        Y_train:     Which model to create
        n_estimators:   Number estimators (for random forest)
        random_state:   Random state to train with (for random forest)

    Returns: A trained model of the type specified

    """
    model = get_model_by_name(model_type, n_estimators, random_state)
    model.fit(X_train, Y_train)
    return model


def get_true_file_name(model_type, filename):
    """
    Takes a model type and file name, formats model type, and replaces spaces with underscores
    Args:
        model_type: The model type as a String
        filename: The file name (should follow 'model_{}.pkl' format)

    Returns: The clean file name
    """
    return filename.format(model_type).replace(" ", "_")


def save_model(model, model_type, filename=default_unformatted_model_path):
    """
    Save the trained model to a file.

    Args:
        model: Trained model to save
        model_type: The type of model being saved
        filename (str): Name of the file to save the model to
    """
    true_file_name = get_true_file_name(model_type, filename)
    with open(true_file_name, "wb") as model_file:
        pickle.dump(model, model_file)


def load_model(model_type, filename=default_unformatted_model_path):
    """
    Load a trained model from a file.

    Args:
        model_type: The type of model being loaded
        filename (str): Name of the file to load the model from

    Returns:
        The loaded model
    """
    true_file_name = get_true_file_name(model_type, filename)
    with open(true_file_name, "rb") as model_file:
        return pickle.load(model_file)


def main(argv):
    """Main function to train and save the model."""
    # Get the model type from the command line arguments
    model_type = argv[1]

    # Train and save the model
    print("Starting model training for {} model...".format(model_type))
    X_train, x_test, Y_train, y_test = prepare_model_data()
    model = train_model(X_train, Y_train, model_type)
    save_model(model, model_type)
    print("Model training completed and saved successfully.")


if __name__ == "__main__":
    main(sys.argv)
