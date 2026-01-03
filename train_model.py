import pandas as pd
import string
import nltk
from nltk.corpus import stopwords

# Download stopwords if not already present
nltk.download('stopwords')

# Load the dataset
# usage of utf-8-sig handles the Byte Order Mark (BOM) if present
try:
    df = pd.read_csv('Spam_SMS.csv', encoding='utf-8-sig')
except UnicodeDecodeError:
    df = pd.read_csv('Spam_SMS.csv', encoding='latin-1')

# Debug: Print column names to check for issues
print("Columns found:", df.columns.tolist())

# Clean column names (remove spaces and special chars from column names if needed)
df.columns = df.columns.str.replace('ï»¿', '').str.strip()

# If columns are v1, v2 (common in this dataset), rename them
if 'v1' in df.columns and 'v2' in df.columns:
    df.rename(columns={'v1': 'Class', 'v2': 'Message'}, inplace=True)

# Data Cleaning: Remove unnamed columns if any (often happens with this dataset)
# Only keep Class and Message if they exist
if 'Class' in df.columns and 'Message' in df.columns:
    df = df[['Class', 'Message']]
else:
    print("Error: Expected columns 'Class' and 'Message' not found.")
    print("Available columns:", df.columns.tolist())
    exit()

# Function to preprocess text
def preprocess_text(text):
    # 1. Convert to lowercase
    text = text.lower()
    
    # 2. Remove punctuation
    # string.punctuation contains characters like !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
    text = "".join([char for char in text if char not in string.punctuation])
    
    # 3. Remove stopwords
    # Stopwords are common words like 'the', 'is', 'in' that don't carry much meaning for spam detection
    stop_words = set(stopwords.words('english'))
    text = " ".join([word for word in text.split() if word not in stop_words])
    
    return text

# Apply the preprocessing to the 'Message' column
print("\nPreprocessing text... This might take a moment.")
df['Clean_Message'] = df['Message'].apply(preprocess_text)

# Display the first 5 rows to see the difference
print(df[['Message', 'Clean_Message']].head())

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# Step 4: Feature Extraction (TF-IDF)
# Convert text to numbers
print("\nSplitting data and extracting features...")

X = df['Clean_Message']
y = df['Class']

# Split into Training (80%) and Testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize TF-IDF Vectorizer
# max_features=3000 means we only keep the top 3000 most frequent words to avoid making the model too complex
tfidf = TfidfVectorizer(max_features=3000)

# Fit the vectorizer on training data and transform it
# We ONLY fit on training data to avoid "data leakage" (cheating)
X_train_tfidf = tfidf.fit_transform(X_train).toarray()
X_test_tfidf = tfidf.transform(X_test).toarray()

print(f"Feature Extraction Complete.")
print(f"Training Data Shape: {X_train_tfidf.shape}")
print(f"Testing Data Shape: {X_test_tfidf.shape}")

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Step 5: Train the Model
# We use Naive Bayes because it works very well for text classification
print("\nTraining the Naive Bayes model...")
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Step 6: Model Evaluation
print("Evaluating the model...")
y_pred = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

import pickle

# Step 7: Save the Model and Vectorizer
print("\nSaving model and vectorizer...")
pickle.dump(model, open('spam_model.pkl', 'wb'))
pickle.dump(tfidf, open('tfidf_vectorizer.pkl', 'wb'))
print("Model saved as 'spam_model.pkl' and vectorizer as 'tfidf_vectorizer.pkl'")
