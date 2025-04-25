from django.urls import path
from . import views

urlpatterns = [
    path('secure-login/', views.secure_login, name='login'),

]