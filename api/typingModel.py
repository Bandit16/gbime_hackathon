import numpy as np
from sklearn.svm import OneClassSVM
from collections import deque
from sklearn.preprocessing import StandardScaler # ensures large valued ddata doesnt alter the model vastly, Z=x-miu/sigma
from sklearn.pipeline import Pipeline
import time
import joblib

class keyStrokeAuthenticator:
    def __init__(self, account_number=None, buffer_size=100, nu=0.5, gamma="scale"):
        self.account_number = account_number  # Store account_number
        self.buffer = deque(maxlen=buffer_size)
        self.model = None
        self.saclar = StandardScaler()
        self.nu = nu
        self.gamma = gamma
        self.trained = False
        self.model_path = f"models/{self.account_number}_model.joblib" if self.account_number else None
    def collect_typing_sample(self):
        features = np.array([
            np.random.normal(100, 20),    # Average key press duration (ms)
            np.random.normal(150, 30),    # Average digraph time like DIPESH ma D paxi I thichne time
            np.random.normal(50, 15),     # Average flight time like euta key release garepaxi arko garan lagen time
            np.random.normal(300, 50),   # Words per minute
            np.random.normal(0.05, 0.02), # Error rate user le kati choti backspace thichxa vanne pattern
            np.random.normal(100, 20),    # Key press duration variance
        ])
        return features.reshape(1,-1)
    
    
    def train_model(self):
        if len(self.buffer)<self.buffer.maxlen:
            return False
        pipeline=Pipeline([
           ("scalar",StandardScaler()),
           ("svm", OneClassSVM(kernel="rbf",nu=self.nu,gamma=self.gamma))
        ]) # data preprocessing ra modek traing ko pipeline banaako
        
        pipeline.fit(np.array(self.buffer))
        self.model=pipeline
        self.trained=True
        joblib.dump(self.model, self.model_path)
        return True
    
    def authenticate(self,sample):
        if not self.trained:
            print("model not trained")
            return None
        if self.model is None:
            try:
                self.model = joblib.load(self.model_path)
                print("model found")
            except FileNotFoundError:
                print("model not found")
                return None
        predict=self.model.predict(sample)
        decision_score=self.model.decision_function(sample)
        return{
            "isAuthenticate": predict[0]==1,
            "confidence": -decision_score[0], # jati dherai hunxa teti anamolous xa
            "decision_score": decision_score[0]
        }
    
    def update_model(self,sample):
        self.buffer.append(sample[0])
        if len(self.buffer)==self.buffer.maxlen and self.trained:
            self.train_model()
            joblib.dump(self.model, 'typingModel.joblib')

if __name__=="__main__":
    authenticator=keyStrokeAuthenticator(buffer_size=100,nu=0.5)
    for _ in range(100):
        sample=authenticator.collect_typing_sample()
        authenticator.buffer.append(sample[0])
        time.sleep(0.05)
    authenticator.train_model()
    for i in range (20):
        sample=authenticator.collect_typing_sample()   
        result=authenticator.authenticate(sample)         
    if result is not None:
        status="Authentic" if result['isAuthenticate'] else ["doubleVerify"]
        print(f"sample {i+1}: status : {status} confidence= {result['confidence']:.2f}")
    
    authenticator.update_model(sample)
    time.sleep(0.05)