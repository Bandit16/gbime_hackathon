
from django.shortcuts import get_object_or_404
from .models import User, KeystrokeFeature
import joblib

def get_user_model(account_number):
    # This assumes models are stored as .pkl files named after the account_number
    model_path = f"models/{account_number}_model.pkl"
    try:
        return joblib.load(model_path)  # Load the user-specific model
    except FileNotFoundError:
        raise Exception(f"Model for account {account_number} not found.")

def predict_keystroke(account_number, feature_vector):
    # Extract features from keystrokes
    # Assuming feature_vector is a list of floats representing the keystroke features

    # Retrieve user and their model
    user = get_object_or_404(User, account_number=account_number)
    model = get_user_model(account_number)

    # Make prediction (assuming it's a binary classification for correct/incorrect)
    prediction = model.predict([feature_vector])

    return prediction  # Return prediction (0 = incorrect, 1 = correct)