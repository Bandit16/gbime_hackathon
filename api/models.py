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