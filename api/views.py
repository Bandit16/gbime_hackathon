from rest_framework.response import Response
from rest_framework.decorators import api_view
from .serializers import UserSerializer, KeystrokeFeatureSerializer , Account_numberSerializer
from .models import User, KeystrokeFeature
from rest_framework import status
from django.contrib.auth import authenticate, login
from rest_framework.authtoken.models import Token
from .typingModel import keyStrokeAuthenticator
import numpy as np
import os
import joblib



@api_view(['POST'])
def secure_login(request):
    acc = request.data.get('account_number')
    uname = request.data.get('username')
    pw = request.data.get('password')
    features = request.data.get('keystroke_features')
    session_id = request.data.get('session_id')

    try:
        user = User.objects.get(account_number=acc)
    except User.DoesNotExist:
        return Response({'error': 'User not found'}, status=404)

    # Save keystroke feature to DB
    serializer = KeystrokeFeatureSerializer(data={
        'user': user.id,
        'session_id': session_id,
        'feature_vector': features
    })
    if serializer.is_valid():
        serializer.save()
        # Get the IDs of the 100 most recent records
        recent_ids = KeystrokeFeature.objects.filter(user=user).order_by('-timestamp')[:100].values_list('id', flat=True)

        # Delete all records except the 100 most recent ones
        KeystrokeFeature.objects.filter(user=user).exclude(id__in=recent_ids).delete()
    else:
        return Response(serializer.errors, status=400)

    # ML: Setup and load model
    model_path = f"models/{acc}_model.joblib"
    authenticator = keyStrokeAuthenticator(account_number=acc, buffer_size=100, nu=0.5)
    buffer = list(KeystrokeFeature.objects.filter(user=user)
                  .order_by('-timestamp')[:100]
                  .values_list('feature_vector', flat=True))
    if buffer:
        authenticator.buffer.extend(buffer)

    if os.path.exists(model_path):
        print("Model exists. Loading...")
        try:
            authenticator.model = joblib.load(model_path)
            authenticator.trained = True
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
            authenticator.trained = False
    else:
        print("Model does not exist.")

    if len(authenticator.buffer) == authenticator.buffer.maxlen and not authenticator.trained:
        authenticator.train_model()
        joblib.dump(authenticator.model, model_path)

    # Authenticate typing pattern
    sample = np.array(features).reshape(1, -1)
    result = authenticator.authenticate(sample)
    print(f"Decision Score: {result['decision_score']}, Confidence: {result['confidence']}")
    if not result or not result["isAuthenticate"]:
        return Response({'error': 'Invalid typing pattern'}, status=403)
    
    # Django auth
    user_auth = authenticate(username=uname, password=pw)
    if user_auth:
        token, _ = Token.objects.get_or_create(user=user_auth)
        return Response({'success': True, 'token': token.key, 'confidence': result['confidence']})
    return Response({'error': 'Invalid credentials'}, status=403)

# api testing
'''
curl -X POST http://localhost:8000/api/login/ \
  -H "Content-Type: application/json" \
  -d '{
    "account_number": "1234567890",
    "username": "testuser",
    "password": "yourpassword",
    "session_id": "session123",
    "keystroke_features": [0.12, 0.08, 0.09, 0.11, 0.15, 0.07, 0.10, 0.13]
  }'

'''

'''
curl -X POST http://127.0.0.1:8000/secure-login/ \
  -H "Content-Type: application/json" \
  -d '{
    "account_number": "123",
    "username": "dipesh",
    "password": "dipesh",
    "session_id": "session123",
    "keystroke_features": [0.12, 0.08, 0.09, 0.11, 0.15, 0.07, 0.10, 0.13]
}'
'''




# from rest_framework.decorators import api_view
# from rest_framework.response import Response
# import pandas as pd
# from .fraud_detector import FraudDetector  # adjust import path
# import os

# MODEL_PATH = 'models/fraud_detector.joblib'

# @api_view(['POST'])
# def train_or_predict_fraud(request):
#     action = request.data.get("action", "predict")
#     transactions = request.data.get("transactions", [])

#     if not transactions:
#         return Response({"error": "No transaction data provided."}, status=400)

#     df = pd.DataFrame(transactions).fillna(0)

#     # Train and save model
#     if action == "train":
#         detector = FraudDetector(contamination=0.02)
#         detector.fit(df)
#         os.makedirs('models', exist_ok=True)
#         detector.save(MODEL_PATH)
#         return Response({"message": "Model trained and saved."})

#     # Predict using existing model
#     elif action == "predict":
#         if not os.path.exists(MODEL_PATH):
#             return Response({"error": "Model not found. Train it first."}, status=400)

#         detector = FraudDetector.load(MODEL_PATH)
#         results = detector.predict(df)
#         return Response(results[['userId', 'amount', 'anomaly_score', 'predictedFraud']].to_dict(orient='records'))

#     return Response({"error": "Invalid action. Use 'train' or 'predict'."}, status=400)

'''
{
  "action": "train",
  "transactions": [{ "amount": 200, "hourOfDay": 14, ... }]
}
'''