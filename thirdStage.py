import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
import os
import matplotlib.pyplot as plt

class LSTMAutoEncoderFraudDetection:
    def __init__(self, sequenceLength=20, encodingDim=16, lstmUnit=32):
        self.sequenceLength = sequenceLength
        self.encodingDim = encodingDim
        self.lstmUnit = lstmUnit
        self.scaler = MinMaxScaler()
        self.model = None
        self.threshold = None
        self.featureColumns = None

    def preprocessData(self, transactionData, featureColumns):
        if len(transactionData) < self.sequenceLength:
            raise ValueError(f"Need at least {self.sequenceLength} samples, got {len(transactionData)}")
        # One-hot encode categorical columns
        if 'merchant_category' in featureColumns:
            transactionData = pd.get_dummies(transactionData, columns=['merchant_category'])
            updated_feature_columns = [col for col in featureColumns if col != 'merchant_category'] + \
                                    [col for col in transactionData.columns if 'merchant_category_' in col]
        else:
            updated_feature_columns = featureColumns.copy()
        
        self.featureColumns = updated_feature_columns

        # Scale the data
        data = transactionData[self.featureColumns].values
        if len(data) > 0:  # Only fit scaler if data exists
            scaledData = self.scaler.fit_transform(data)
        else:
            scaledData = data
        
        # Create sequences
        sequences = []
        for i in range(len(scaledData) - self.sequenceLength + 1):
            sequences.append(scaledData[i:i + self.sequenceLength])
        
        return np.array(sequences)

    def buildModel(self):
        if not self.featureColumns:
            raise ValueError("Feature columns not set. Call preprocessData first.")
            
        # Encoder
        inputSequence = Input(shape=(self.sequenceLength, len(self.featureColumns)))
        encoded = LSTM(self.lstmUnit, activation='relu', return_sequences=True)(inputSequence)
        encoded = LSTM(self.encodingDim, activation='relu')(encoded)
        
        # Decoder
        decoded = RepeatVector(self.sequenceLength)(encoded)
        decoded = LSTM(self.lstmUnit, activation='relu', return_sequences=True)(decoded)
        decoded = TimeDistributed(Dense(len(self.featureColumns)))(decoded)
        
        self.model = Model(inputSequence, decoded)
        self.model.compile(optimizer=Adam(), loss="mse")

    def train(self, sequence, validationSplit=0.2, epoch=30, batchSize=64):
        if self.model is None:
            self.buildModel()
            
        X_train, X_val = train_test_split(sequence, test_size=validationSplit, shuffle=True)
        history = self.model.fit(
            X_train, X_train,
            epochs=epoch,
            batch_size=batchSize,
            validation_data=(X_val, X_val),
            verbose=1
        )
        return history
    def setAnomalyThreshold(self, sequence, percentile=95):
        reconstruction = self.model.predict(sequence)
        mse = np.mean(np.power(sequence - reconstruction, 2), axis=(1, 2))
        self.threshold = np.percentile(mse, percentile)

    def detectAnomalies(self, newSequence):
        if self.model is None:
            raise ValueError("Model not trained!")
        if self.threshold is None:
            raise ValueError("Threshold not set!")
        
        reconstruction = self.model.predict(newSequence)
        mse = np.mean(np.power(newSequence - reconstruction, 2), axis=(1, 2))
        anomalies = mse > self.threshold
        return anomalies, mse

    def evaluateFraudDetection(self, labeledData, sequence):
        sequenceLabel = labeledData['is_fraud'][self.sequenceLength - 1:].values
        anomalies, _ = self.detectAnomalies(sequence)
        
        return {
            'precision': precision_score(sequenceLabel, anomalies),
            'recall': recall_score(sequenceLabel, anomalies),
            'f1_score': f1_score(sequenceLabel, anomalies),
            'threshold': self.threshold,
            'fraud_cases': sum(sequenceLabel),
            'detected_fraud': sum(anomalies & sequenceLabel)
        }

    def saveModel(self, path):
        self.model.save(path)  # Saves model in Keras format
        print(f"Model saved to {path}")

    def loadModel(self, path):
        self.model = load_model(path)
        print(f"Model loaded from {path}")
    
    def plot_loss(self,history):
        plt.figure(figsize=(8, 5))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Autoencoder Training vs Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.show()
        
    def plot_reconstruction_error(self,detector, sequences):
        reconstruction = detector.model.predict(sequences)
        mse = np.mean(np.square(sequences - reconstruction), axis=(1, 2))
        
        plt.figure(figsize=(8, 5))
        plt.hist(mse, bins=50, color='skyblue', edgecolor='black')
        plt.axvline(detector.threshold, color='red', linestyle='dashed', linewidth=2, label=f'Threshold: {detector.threshold:.4f}')
        plt.title('Reconstruction Error Distribution')
        plt.xlabel('MSE')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True)
        plt.show()



if __name__ == "__main__":
    # Load data
    transaction_data = pd.read_csv('/Users/lalitramanmishra/GlobalIME/validTranscation/banking_transactions.csv')
    feature_columns = ['amount', 'time_since_last_transaction', 'merchant_category', 'location_distance']
    
    # Initialize and preprocess training data
    detector = LSTMAutoEncoderFraudDetection(sequenceLength=20)
    sequences = detector.preprocessData(transaction_data, feature_columns)
    
    # Train or load model
    model_path = 'fraud_detector_model.h5'
    if os.path.exists(model_path):
        detector.loadModel(model_path)
        
    else:
        detector.buildModel()
        history = detector.train(sequences)
        detector.saveModel(model_path)
        detector.plot_loss(history)
    
    # Set threshold
    detector.setAnomalyThreshold(sequences)

    detector.plot_reconstruction_error(detector, sequences)
    # Prepare new input data (example)
    new_data = pd.DataFrame([{
        'amount': 150.0,
        'time_since_last_transaction': 3600,
        'merchant_category': 'fuel',
        'location_distance': 25.0
    }])
    
    # Preprocess new data (using the same scaler and features)
    new_sequences = detector.preprocessData(new_data, feature_columns)
    if len(new_sequences) > 0:
        anomalies, scores = detector.detectAnomalies(new_sequences)
        print(f"Anomaly detection results: {anomalies}, Scores: {scores}")
    anamolies,mse=detector.detectAnomalies(newSequence=new_sequences)
    print(f"anamolies :{anamolies}")
    # Evaluate if labels exist
    if 'is_fraud' in transaction_data.columns:
        metrics = detector.evaluateFraudDetection(transaction_data, sequences)
        print(f"Metrics: {metrics}")