import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

class FraudDetector:
    def __init__(self, contamination='auto', random_state=30):
        self.contamination = contamination
        self.random_state = random_state
        self.model = None
        self.feature_names = None
        self.threshold = None
        self._initialize_pipeline()

    def _initialize_pipeline(self):
        numeric_features = [
            'amount', 'time_since_last_txn', 'distance_from_home',
            'txn_time', 'txn_frequency', 'amount_deviation', 'sineHour', 'cosHour'
        ]
        categorical_features = ['merchant_category', 'merchant_risk']

        self.preprocessor = ColumnTransformer(transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

        self.model = Pipeline([
            ('preprocessor', self.preprocessor),
            ('detector', IsolationForest(
                n_estimators=200,
                contamination=self.contamination,
                max_features=1,
                random_state=self.random_state,
                verbose=1
            ))
        ])

    def engineer_features(self, df):

        df['sineHour'] = np.sin(2 * np.pi * df['hour_of_day'] / 24)
        df['cosHour'] = np.cos(2 * np.pi * df['hour_of_day'] / 24)

        df = df.sort_values(['user_id', 'hour_of_day'])
        df['amount_deviation'] = (df['amount'] - df['amount'].rolling(window=30).mean()).abs()
        df['txn_frequency'] = df['user_id'].map(df['user_id'].value_counts())
        df['txn_time'] = np.random.randint(1, 25, len(df))  # simulate txnTime
        df = df.fillna(0).infer_objects()
        return df

    def fit(self, df):
        df = self.engineer_features(df.copy())
        self.model.fit(df)
        scores = self.model.named_steps['detector'].score_samples(
            self.model.named_steps['preprocessor'].transform(df)
        )
        self.threshold = np.percentile(scores, 100 * float(self.contamination))

        num_features = self.preprocessor.transformers_[0][2]
        cat_features = self.preprocessor.named_transformers_['cat'].get_feature_names_out()
        self.feature_names = num_features + list(cat_features)

    def predict(self, df):
        if not self.model:
            raise RuntimeError("Model not trained.")

        df = self.engineer_features(df.copy())
        transformed = self.model.named_steps['preprocessor'].transform(df)
        df['anomaly_score'] = self.model.named_steps['detector'].score_samples(transformed)
        df['predicted_fraud'] = (df['anomaly_score'] < self.threshold).astype(int)
        print(df.columns)
        return df

    def evaluate(self, df):
        if "is_fraud" not in df.columns:
            raise ValueError("Data needs an 'is_fraud' column for evaluation.")

        preds = self.predict(df)
        print(classification_report(preds['is_fraud'], preds['predicted_fraud']))

        cm = confusion_matrix(preds['is_fraud'], preds['predicted_fraud'])
        sns.heatmap(cm, annot=True, fmt='d', cmap="Blues")
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.show()

    def feature_importance(self):
        if not hasattr(self.model.named_steps['detector'], 'feature_importances_'):
            print("Feature importance is not available for this model.")
            return

        plt.figure(figsize=(12, 8))
        pd.Series(
            self.model.named_steps['detector'].feature_importances_,
            index=self.feature_names
        ).sort_values().plot.barh()
        plt.title("Fraud Detection Feature Importance")
        plt.show()

    def save(self, path):
        joblib.dump({
            "model": self.model,
            "threshold": self.threshold,
            "features": self.feature_names
        }, path)

    @classmethod
    def load(cls, path):
        data = joblib.load(path)
        detector = cls()
        detector.model = data['model']
        detector.threshold = data['threshold']
        detector.feature_names = data['features']
        return detector

# yo class mero use ko laagi maatrai ho just data genration
class TransactionSimulator:
    @staticmethod
    def generate(n_normal=9800, n_fraud=200):
        data = {
            'amount': np.concatenate([
                np.abs(np.random.normal(1000, 200, n_normal)),
                np.abs(np.random.normal(8000, 3000, n_fraud))
            ]),
            'timeSinceLastTxn': np.concatenate([
                np.random.gamma(2, 1.5, n_normal),
                np.random.exponential(0.3, n_fraud)
            ]),
            'hour_of_day': np.concatenate([
                np.random.randint(9, 18, n_normal),
                np.random.choice([0, 1, 2, 3, 4, 23], n_fraud)
            ]),
            'distanceFromHome': np.concatenate([
                np.random.randint(0, 50, n_normal),
                np.random.randint(100, 5000, n_fraud)
            ]),
            'merchantRisk': np.concatenate([
                np.random.choice([1, 2, 3], n_normal, p=[0.7, 0.25, 0.05]),
                np.random.choice([3, 4, 5], n_fraud, p=[0.3, 0.5, 0.2])
            ]),
            'merchantCategory': np.concatenate([
                np.random.choice(['Retail', 'Food', 'Utilities'], n_normal),
                np.random.choice(['Gambling', 'Cryptocurrency', 'Wire Transfer'], n_fraud)
            ]),
            'userId': np.concatenate([
                np.random.randint(1, 500, n_normal),
                np.random.randint(1, 50, n_fraud)
            ]),
            'isFraud': np.concatenate([
                np.zeros(n_normal),
                np.ones(n_fraud)
            ])
        }
        return pd.DataFrame(data)


if __name__ == "__main__":
    # mero laagi maatrai timi aafno  data use garnu
    print("Generating transaction data")
    transactions = TransactionSimulator.generate()
    # model already xa vane load garne
    # detector = joblib.load('fraud_detector.joblib')
    
    # if model xaina vane first available data ma train garne or if many more than 20 new transcation aayo vane
    detector = FraudDetector(contamination=0.02)
    print("Training fraud detection model...")
    detector.fit(transactions)
    print("hellos")
    
    # now input jun aako xa teslai predict garne and if anamoly score smallere than -0.5 xa vane anamolous flage garne if sakxau vane smaller tahn threshold calulsaate garnu
    new_data = TransactionSimulator.generate(n_normal=1000, n_fraud=20)
    new_data=new_data.fillna(0)
    results = detector.predict(new_data)

    print("\nFraud Detection Results:")
    print(results[results['isFraud'] == 1][['amount', 'merchant_category', 'anomaly_score']])
    # if model train vaako xa vane save garne
    detector.save('fraud_detector.joblib')
    print("model savved suceesfull")
