from django.db import models
from django.contrib.postgres.fields import ArrayField
from django.db.models import JSONField

class User(models.Model):
    username = models.CharField(max_length=150, unique=True)
    password = models.CharField(max_length=128)  # Store hashed passwords in production
    account_number = models.CharField(max_length=20, unique=True)
    email = models.EmailField(unique=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.username



class KeystrokeFeature(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    session_id = models.CharField(max_length=50)  # identifies one full password input
    feature_vector = JSONField()
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user.username} - {self.session_id}"
    
class Transaction(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    amount = models.FloatField()
    time_since_last_txn = models.FloatField()
    hour_of_day = models.IntegerField()
    distance_from_home = models.FloatField()
    merchant_risk = models.IntegerField()
    merchant_category = models.CharField(max_length=50)
    is_fraud = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    # Prediction results
    anomaly_score = models.FloatField(null=True, blank=True)
    predicted_fraud = models.BooleanField(null=True, blank=True)
    txn_frequency = models.IntegerField(null=True, blank=True)
    amount_deviation = models.FloatField(null=True, blank=True)

    def __str__(self):
        return f"Transaction {self.id} - {self.user.username} - {'Predicted Fraud' if self.predicted_fraud else 'Not Fraud'}"