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
# SpamGuard AI 🛡️

A self-learning Spam SMS Detection System that improves over time. This project uses a **Naive Bayes** classifier to detect spam messages and includes a dynamic feedback loop for continuous learning.

## 🚀 Features

- **Real-time Spam Detection**: Classifies messages as "Spam" or "Ham" (Safe).
- **Self-Learning Engine**: Retrains itself automatically every few minutes to learn from new data.
- **Feedback Loop**: Users can correct the AI ("Report as Spam" / "Report as Safe").
- **Simulation Bot**: Includes a bot that simulates user traffic to test the self-learning capabilities.
- **Modern UI**: Clean, responsive interface with live retraining status.

---

## 🛠️ Setup Instructions

### 1. Prerequisites
- Python 3.8 or higher installed.

### 2. Install Dependencies
Open your terminal in the project folder and run:
```bash
pip install -r requirements.txt
```

### 3. Initialize the Database
Run the following commands to set up the database and create the necessary tables:
```bash
python mysite/manage.py makemigrations
python mysite/manage.py migrate
```

---

## 🎮 How to Run

To see the full "Self-Learning" system in action, you should run these **3 components** simultaneously in separate terminals.

### 1️⃣ Start the Web Server
This runs the main website where you can test messages manually.
```bash
python mysite/manage.py runserver
```
👉 Open your browser at: `http://127.0.0.1:8000/`

### 2️⃣ Start the Retraining Loop (The "Brain")
This script runs in the background and retrains the model every **3 minutes** using new feedback.
- **Windows**: Double-click `run_retraining_loop.bat`
- **Manual**:
  ```bash
  python mysite/manage.py retrain
  ```

### 3️⃣ Start the Simulation Bot (Optional)
This bot acts like a user, sending thousands of test messages and automatically correcting the AI when it makes mistakes.
- **Windows**: Double-click `run_simulation.bat`
- **Manual**:
  ```bash
  python mysite/manage.py simulate spam_dataset.csv
  ```

---

## 🧠 How It Works

1. **Detection**: The model predicts if a message is Spam or Ham.
2. **Feedback**:
   - **Real Users**: Click "Report as Spam" or "Confirm Correct" on the website.
   - **Simulator**: Automatically checks predictions against a validation dataset.
3. **Learning**:
   - All feedback is saved to the database.
   - The **Retraining Loop** wakes up every 3 minutes.
   - It combines the original dataset + new feedback to train a smarter model.
   - The website instantly hot-reloads the new model without restarting.

## 📂 Project Structure

- `mysite/`: Main Django project.
- `mysite/spam_app/`: The core application logic.
  - `models.py`: Database models (Feedback storage).
  - `views.py`: Prediction logic & Hot-reloading system.
  - `management/commands/`: Custom scripts (`retrain.py`, `simulate.py`).
- `run_retraining_loop.bat`: Helper script for the learning loop.
- `run_simulation.bat`: Helper script for the simulation bot.
