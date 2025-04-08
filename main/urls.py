from django.urls import path
from . import views

urlpatterns = [
     path('', views.test_aq10, name='main_test_aq10'),
]