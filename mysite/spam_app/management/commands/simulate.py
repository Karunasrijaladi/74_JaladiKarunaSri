from django.core.management.base import BaseCommand
import pandas as pd
import pickle
import os
import time
from django.conf import settings
from spam_app.models import Feedback
from spam_app.views import preprocess_text

class Command(BaseCommand):
    help = 'Simulates user interaction using a second dataset'

    def add_arguments(self, parser):
        parser.add_argument('csv_file', type=str, help='Path to the validation CSV file')

    def handle(self, *args, **kwargs):
        csv_file = kwargs['csv_file']
        
        self.stdout.write(f"Loading simulation data from {csv_file}...")
        # Load the simulator data
        try:
            df = pd.read_csv(csv_file, encoding='utf-8-sig')
        except UnicodeDecodeError:
            df = pd.read_csv(csv_file, encoding='latin-1')
        except FileNotFoundError:
             self.stdout.write(self.style.ERROR(f"File not found: {csv_file}"))
             return
            
        if 'v1' in df.columns:
            df.rename(columns={'v1': 'Class', 'v2': 'Message'}, inplace=True)

        self.stdout.write("Starting Simulation... Press Ctrl+C to stop.")

        # Simulate a user entering 1 message every 5 seconds
        for index, row in df.iterrows():
            message = row['Message']
            true_label = row['Class']
            
            # 1. Load current model (It might have changed since last loop!)
            model_path = os.path.join(settings.BASE_DIR, 'spam_model.pkl')
            vec_path = os.path.join(settings.BASE_DIR, 'tfidf_vectorizer.pkl')
            
            try:
                model = pickle.load(open(model_path, 'rb'))
                tfidf = pickle.load(open(vec_path, 'rb'))
            except Exception as e:
                self.stdout.write(self.style.ERROR(f"Error loading model: {e}"))
                time.sleep(5)
                continue
            
            # 2. Predict
            try:
                clean_msg = preprocess_text(message)
                # transform expects iterable
                features = tfidf.transform([clean_msg]).toarray()
                prediction = model.predict(features)[0]
                
                # 3. "User" Feedback
                is_correct = (prediction == true_label)
                
                # 4. Save Feedback to DB
                Feedback.objects.create(
                    message=message,
                    predicted_label=prediction,
                    actual_label=true_label, # In simulation, we know the truth
                    is_correct=is_correct
                )
                
                status = "✅" if is_correct else "❌"
                self.stdout.write(f"[{status}] Msg: {message[:30]}... | Pred: {prediction} | True: {true_label}")
            except Exception as e:
                 self.stdout.write(self.style.ERROR(f"Error processing row: {e}"))

            # Wait to simulate human typing speed
            time.sleep(2)
