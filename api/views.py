from rest_framework.response import Response
from rest_framework.decorators import api_view
from .serializers import *
from .models import User, KeystrokeFeature , Transaction
from .fraud_detector import FraudDetector
import pandas as pd
from rest_framework import status
from django.contrib.auth import authenticate, login
from rest_framework.authtoken.models import Token
from .typingModel import keyStrokeAuthenticator
from .lstm_auto_encoder import LSTMAutoEncoderFraudDetection
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
    model_path = f"models/{uname}_model.joblib"
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
        return Response({   'error': 'App bata authentic req pathau'}, status=200)

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


def get_user_data(user):
    txns = Transaction.objects.filter(user=user).order_by('created_at')
    return pd.DataFrame(list(txns.values()))


@api_view(['POST'])
def log_transaction(request):
    serializer = TransactionSerializer(data=request.data)
    if not serializer.is_valid():
        return Response(serializer.errors, status=400)

    transaction = serializer.save()
    user = transaction.user
    txns = Transaction.objects.filter(user=user)

    # Train model after 20 transactions
    model_path = f"models/{user.id}_detector.joblib"
    if txns.count() >= 20:
        if os.path.exists(model_path):
            detector = FraudDetector.load(model_path)
            print("Model exists. Loading...")
        else:
            df = get_user_data(user)
            print("Columns in DataFrame:", df.columns)
            detector = FraudDetector(contamination=0.02)
            df = detector.engineer_features(df)
            detector.fit(df)
            detector.save(model_path)

        # latest = df.iloc[-1]
        df = get_user_data(user)
        prediction = detector.predict(df.tail(1)).iloc[0]

        # Save prediction details
        transaction.anomaly_score = prediction['anomaly_score']
        transaction.predicted_fraud = bool(prediction['predicted_fraud'])
        transaction.txn_frequency = prediction['txn_frequency']
        transaction.amount_deviation = prediction['amount_deviation']
        transaction.save()

        return Response({
            'message': 'Transaction logged and prediction made.',
            'predicted_fraud': transaction.predicted_fraud,
            'anomaly_score': transaction.anomaly_score,
        })

    return Response({'message': 'Transaction logged. Need 20 transactions to start prediction.'})
def get_or_create_detector(user_id):
    detector = LSTMAutoEncoderFraudDetection(sequenceLength=20)
    feature_columns = ['amount', 'time_since_last_txn', 'merchant_category', 'distance_from_home']
    
    user_model_path = f'models/user_{user_id}_model.h5'

    if os.path.exists(user_model_path):
        detector.loadModel(user_model_path)
    else:
        detector.model = None
    
    return detector, feature_columns, user_model_path

@api_view(['POST'])
def lstm_transaction_view(request):
    serializer = TransactionSerializer(data=request.data)
    if serializer.is_valid():
        transaction = serializer.save()

        # Initialize detector
        detector, feature_columns, model_path = get_or_create_detector(transaction.user.id)

        # Get user's transaction data
        df = get_user_data(transaction.user)

        if df.shape[0] < 20:
            return Response({'message': 'Not enough transactions to predict.'}, status=status.HTTP_200_OK)

        # Prepare feature DataFrame
        df = df[[
            'amount', 
            'time_since_last_txn', 
            'merchant_category', 
            'distance_from_home'
        ]]

        try:
            
            sequences = detector.preprocessData(df, feature_columns)
            print(f"Shape of sequences: {sequences.shape}")
        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)

        # Train if model not trained
        if detector.model is None:
            detector.buildModel()
            detector.train(sequences)
            detector.setAnomalyThreshold(sequences)
            detector.saveModel(model_path)
            return Response({'message': 'Model trained. Send another transaction to predict.'}, status=status.HTTP_200_OK)

        # Predict
        detector.setAnomalyThreshold(sequences)

        anomalies, mse_scores = detector.detectAnomalies(sequences)
        is_fraud = anomalies[-1]

        # Save prediction
        transaction.predicted_fraud = is_fraud
        transaction.anomaly_score = mse_scores[-1]
        transaction.save()

        return Response({'fraud': bool(is_fraud)}, status=status.HTTP_200_OK)

    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
# def retrain_model(user):
#     detector = get_user_detector(user)  # load/create user's detector
#     transactions = Transaction.objects.filter(user=user, is_fraud=True)

#     if transactions.count() >= 20:
#         # Prepare data
#         df = pd.DataFrame(list(transactions.values(
#             'amount', 'time_since_last_txn', 'merchant_category', 'distance_from_home'
#         )))
#         feature_columns = ['amount', 'time_since_last_txn', 'merchant_category', 'distance_from_home']
#         sequences = detector.preprocessData(df, feature_columns)

#         # Retrain
#         detector.buildModel()
#         history = detector.train(sequences)
#         detector.setAnomalyThreshold(sequences)

#         # Save model
#         save_user_model(detector, user)

# def confirm_fraud(request, txn_id):
#     txn = Transaction.objects.get(id=txn_id)
#     txn.is_fraud = True
#     txn.save()

#     retrain_model(txn.user)

#     return Response({'message': 'Transaction marked as fraud and model retrained.'})





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

    "user": 1,

    "session_id": "session123",
    "keystroke_features": [0.12, 0.08, 0.09, 0.11, 0.15, 0.07]
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



'''
{
    "amount": 200.50,
    "time_since_last_txn": 3600,
    "hour_of_day": 14,
    "distance_from_home": 5.2,
    "merchant_risk": 2,
    "merchant_category": "electronics",
    "user": 1
}
'''
'''
Columns in DataFrame: Index(['id', 'user_id', 'amount', 'time_since_last_txn', 'hour_of_day',
       'distance_from_home', 'merchant_risk', 'merchant_category', 'is_fraud',
       'created_at', 'anomaly_score', 'predicted_fraud', 'txn_frequency',
       'amount_deviation'],
      dtype='object')
      '''