AI-Powered Spam SMS Classification Using NLP and Machine Learning
📌 Problem Statement

Spam SMS messages cause inconvenience and may lead to financial fraud. Traditional rule-based filters struggle to adapt to evolving spam patterns. This project proposes an AI-powered system that classifies SMS messages as Spam or Ham (Not Spam) using Natural Language Processing (NLP) and Machine Learning techniques.

🎯 Objectives

Classify SMS messages into Spam or Ham

Build a lightweight and efficient machine learning model

Incorporate AI-driven explanations for predictions

Ensure reproducibility within a hackathon environment

📂 Dataset

SMS Spam Collection Dataset (Kaggle)

Labeled SMS messages (spam, ham)

Widely used benchmark dataset for spam detection

Dataset link:
https://www.kaggle.com/datasets/mariumfaheem666/spam-sms-classification-using-nlp

🧠 Solution Overview

Text preprocessing using NLP techniques (cleaning, tokenization, stopword removal)

Feature extraction using TF-IDF Vectorization

Classification using Multinomial Naive Bayes

Optional integration of Generative AI for explainable predictions

🏗️ System Architecture
SMS Message
   ↓
Text Preprocessing
   ↓
TF-IDF Vectorizer
   ↓
Multinomial Naive Bayes Classifier
   ↓
Prediction (Spam / Ham)
   ↓
AI-Based Explanation (Optional)

⚙️ Tech Stack

Python
Pandas
Scikit-learn
Multinomial Naive Bayes
Generative AI (optional: OpenAI / Gemini / Local LLM)
Django (for web application)

📊 Evaluation Metrics
Accuracy
Precision
Recall
F1-score
Confusion Matrix

🌐 Web Application

A Django-based web interface allows users to enter an SMS message and receive real-time spam classification results, simulating real-world SMS spam filtering systems used in telecom networks.

📌 Conclusion

This project demonstrates a practical AI-based spam SMS detection system using NLP and machine learning. The solution is efficient, interpretable, and suitable for real-world deployment as well as AI hackathon evaluation.
