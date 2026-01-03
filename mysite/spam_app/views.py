from django.shortcuts import render, redirect
import pickle
import os
import string
import nltk
from nltk.corpus import stopwords
from django.conf import settings
from .models import Feedback
import datetime

# Load model and vectorizer paths
BASE_DIR = settings.BASE_DIR
model_path = os.path.join(BASE_DIR, 'spam_model.pkl')
vectorizer_path = os.path.join(BASE_DIR, 'tfidf_vectorizer.pkl')

# Global variables
model = None
tfidf = None
last_load_time = 0

def load_model():
    """Checks if model file is newer than last load time"""
    global model, tfidf, last_load_time
    
    try:
        # Check if files exist
        if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
            print(f"DEBUG: Files not found at {model_path}")
            return

        current_mtime = os.path.getmtime(model_path)
        
        # If model is not loaded or file has changed
        if model is None or current_mtime > last_load_time:
            print(f"DEBUG: Loading model. Current mtime: {current_mtime}, Last load: {last_load_time}")
            model = pickle.load(open(model_path, 'rb'))
            tfidf = pickle.load(open(vectorizer_path, 'rb'))
            last_load_time = current_mtime
    except Exception as e:
        print(f"Error loading model: {e}")

# Initial load
load_model()

# Ensure stopwords are downloaded
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

def preprocess_text(text):
    """
    Same preprocessing function used during training
    """
    text = text.lower()
    text = "".join([char for char in text if char not in string.punctuation])
    stop_words = set(stopwords.words('english'))
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text

def predict_spam(request):
    # Check for model updates before processing request
    load_model()
    
    result = None
    message = ""
    explanation = None
    
    if request.method == 'POST':
        message = request.POST.get('message', '')
        
        if message and model and tfidf:
            # 1. Preprocess
            clean_message = preprocess_text(message)
            
            # 2. Vectorize
            # Note: transform() expects an iterable, so we pass [clean_message]
            features = tfidf.transform([clean_message]).toarray()
            
            # 3. Predict
            prediction = model.predict(features)[0] # 'spam' or 'ham'
            
            # 4. Result
            if prediction == 'spam':
                result = "Spam Detected! 🚨"
                explanation = get_ai_explanation(message, "Spam")
            else:
                result = "This message looks Safe (Ham). ✅"
                explanation = get_ai_explanation(message, "Ham")
                
    # Calculate last update time
    last_update_dt = None
    
    # Use in-memory time if available, otherwise check file directly
    if last_load_time:
        timestamp = last_load_time
    elif os.path.exists(model_path):
        timestamp = os.path.getmtime(model_path)
    else:
        timestamp = None

    if timestamp:
        # Create timezone-aware datetime from timestamp
        last_update_dt = datetime.datetime.fromtimestamp(timestamp, tz=datetime.timezone.utc)
        print(f"DEBUG: Last update DT: {last_update_dt}")
    
    return render(request, 'spam_app/index.html', {
        'result': result, 
        'message': message, 
        'explanation': explanation,
        'last_update': last_update_dt
    })

def submit_feedback(request):
    if request.method == 'POST':
        message = request.POST.get('message')
        actual_label = request.POST.get('actual_label')
        predicted_label = request.POST.get('predicted_label')
        
        # Determine correctness
        is_correct = (actual_label == predicted_label)
        
        # Save to DB
        Feedback.objects.create(
            message=message,
            predicted_label=predicted_label,
            actual_label=actual_label,
            is_correct=is_correct
        )
        
    return redirect('predict_spam')

def get_ai_explanation(message, label):
    """
    Simulates a Generative AI explanation.
    In a real hackathon, you would call OpenAI API here.
    """
    if label == "Spam":
        # Simple heuristic explanation for demo purposes
        triggers = ['free', 'win', 'winner', 'prize', 'urgent', 'call', 'click']
        found_triggers = [word for word in triggers if word in message.lower()]
        
        if found_triggers:
            return f"AI Analysis: This message is flagged as spam because it contains suspicious keywords like: {', '.join(found_triggers)}."
        return "AI Analysis: This message fits common spam patterns (promotional content, urgency)."
    else:
        return "AI Analysis: This message appears to be normal conversational text."
