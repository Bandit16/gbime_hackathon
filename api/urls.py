from django.urls import path
from . import views

urlpatterns = [
    path('secure-login/', views.secure_login, name='login'),
    path('log_transaction/', views.log_transaction, name='log_transaction'),

]