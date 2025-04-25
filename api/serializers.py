from rest_framework import serializers
from .models import User, KeystrokeFeature , Transaction

class Account_numberSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ['account_number']


class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ['username','password', 'account_number', 'email']




class KeystrokeFeatureSerializer(serializers.ModelSerializer):
    class Meta:
        model = KeystrokeFeature
        fields = ['user', 'session_id', 'feature_vector', 'timestamp']

class TransactionSerializer(serializers.ModelSerializer):
    class Meta:
        model = Transaction
        fields = '__all__'