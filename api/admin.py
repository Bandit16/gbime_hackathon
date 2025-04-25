from django.contrib import admin
from .models import User, KeystrokeFeature , Transaction
# Register your models here.
admin.site.register(User)
admin.site.register(KeystrokeFeature)
admin.site.register(Transaction)