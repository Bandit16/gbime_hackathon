from django.urls import path
from . import views

urlpatterns = [
    path('secure-login/', views.secure_login, name='login'),
    path('log_transaction/', views.log_transaction, name='log_transaction'),
    path('lstm_transaction_view/', views.lstm_transaction_view, name='lstm_transaction_view'),
    
]