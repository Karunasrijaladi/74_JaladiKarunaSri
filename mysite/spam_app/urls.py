from django.urls import path
from . import views

urlpatterns = [
    path('', views.predict_spam, name='predict_spam'),
    path('submit_feedback/', views.submit_feedback, name='submit_feedback'),
]
