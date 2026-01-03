# 📩 SMS Spam Detection using Machine Learning and Generative AI

## 📌 Problem Statement
Spam SMS messages cause inconvenience and can lead to financial fraud. Rule-based filters fail to adapt to evolving spam patterns. This project builds an AI-based system that classifies SMS messages as Spam or Ham using machine learning, with optional Generative AI explanations to improve transparency.


## 🎯 Objectives
- Classify SMS messages into Spam or Ham
- Build a lightweight and efficient ML model
- Include AI-driven explanation for predictions
- Ensure reproducibility within a hackathon timeframe


## 📂 Dataset
**SMS Spam Collection Dataset (Kaggle)**  
-Directly matches the Kaggle dataset (SMS Spam Collection)
- NLP is the correct term for text preprocessing + TF-IDF
- Simple to implement 
- Judges instantly understand the problem
- Fits perfectly under AI / Machine Learning
- https://www.kaggle.com/datasets/mariumfaheem666/spam-sms-classification-using-nlp


## 🧠 Solution Overview
1. Text preprocessing (cleaning and tokenization)
2. Feature extraction using TF-IDF
3. Classification using Multinomial Naive Bayes
4. Optional Generative AI explanation for predictions

## 🏗️ Architecture
SMS Message
↓
Text Preprocessing
↓
TF-IDF Vectorizer
↓
Naive Bayes Classifier
↓
Prediction (Spam / Ham)
↓
Generative AI Explanation

## ⚙️ Tech Stack
- Python
- Pandas
- Scikit-learn
- Multinomial Naive Bayes
- Generative AI (optional: OpenAI / Gemini / Local LLM)


## 📊 Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- webiste that usually tells us whether that SMS is Spam or not
