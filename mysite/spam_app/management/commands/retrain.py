from django.core.management.base import BaseCommand
import pandas as pd
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from django.conf import settings
from spam_app.models import Feedback
from spam_app.views import preprocess_text 

class Command(BaseCommand):
    help = 'Retrains the spam model with new feedback data'

    def handle(self, *args, **kwargs):
        self.stdout.write("Starting retraining process...")
        
        # 1. Load Original Data
        # Assuming Spam_SMS.csv is in the parent directory of mysite (i.e. Hackathon root)
        # settings.BASE_DIR is usually mysite/
        csv_path = os.path.join(settings.BASE_DIR, '../Spam_SMS.csv')
        try:
            df = pd.read_csv(csv_path, encoding='utf-8-sig')
        except FileNotFoundError:
             # Fallback if running differently or path issues
            self.stdout.write(self.style.ERROR(f"Could not find CSV at {csv_path}"))
            return
        except UnicodeDecodeError:
            df = pd.read_csv(csv_path, encoding='latin-1')
            
        if 'v1' in df.columns:
            df.rename(columns={'v1': 'Class', 'v2': 'Message'}, inplace=True)
        
        # 2. Load New Feedback Data from Database
        new_data = Feedback.objects.filter(used_for_training=False)
        
        if not new_data.exists():
            self.stdout.write("No new feedback to learn from. Skipping.")
            return

        self.stdout.write(f"Learning from {new_data.count()} new examples...")
        
        new_entries = []
        for item in new_data:
            new_entries.append({'Class': item.actual_label, 'Message': item.message})
            
        df_new = pd.DataFrame(new_entries)
        
        # 3. Combine Old + New
        # Ensure we only have relevant columns before concat
        df_original = df[['Class', 'Message']].copy()
        df_final = pd.concat([df_original, df_new], ignore_index=True)
        
        # 4. Preprocess & Train
        self.stdout.write("Preprocessing text...")
        df_final['Clean_Message'] = df_final['Message'].apply(preprocess_text)
        
        self.stdout.write("Vectorizing...")
        tfidf = TfidfVectorizer(max_features=3000)
        X = tfidf.fit_transform(df_final['Clean_Message']).toarray()
        y = df_final['Class']
        
        self.stdout.write("Training Naive Bayes model...")
        model = MultinomialNB()
        model.fit(X, y)
        
        # 5. Save the updated brain
        # Save to the root of the project where views.py expects them? 
        # views.py looks at settings.BASE_DIR for the pkl files.
        # Let's verify where views.py looks.
        
        model_out_path = os.path.join(settings.BASE_DIR, 'spam_model.pkl')
        vec_out_path = os.path.join(settings.BASE_DIR, 'tfidf_vectorizer.pkl')
        
        self.stdout.write(f"Saving model to {model_out_path}...")
        pickle.dump(model, open(model_out_path, 'wb'))
        pickle.dump(tfidf, open(vec_out_path, 'wb'))
        
        # 6. Mark feedback as used
        for item in new_data:
            item.used_for_training = True
            item.save()
            
        self.stdout.write(self.style.SUCCESS('Successfully retrained model!'))
