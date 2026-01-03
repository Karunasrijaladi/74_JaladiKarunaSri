from django.db import models

class Feedback(models.Model):
    message = models.TextField()
    predicted_label = models.CharField(max_length=10)  # What the model thought
    actual_label = models.CharField(max_length=10)     # The correct answer (Feedback)
    is_correct = models.BooleanField()                 # Did we get it right?
    timestamp = models.DateTimeField(auto_now_add=True)
    used_for_training = models.BooleanField(default=False) # To avoid learning the same data twice

    def __str__(self):
        return f"{self.actual_label}: {self.message[:20]}..."
