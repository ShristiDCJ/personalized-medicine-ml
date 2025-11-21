import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
import sys
import csv

def train_and_save_model():
    try:
        # Configure CSV field size limit
        maxInt = sys.maxsize
        while True:
            try:
                csv.field_size_limit(maxInt)
                break
            except OverflowError:
                maxInt = int(maxInt/10)
        
        print("Loading variants data...")
        variants_df = pd.read_csv('../../training_variants.csv')
        
        print("Loading text data...")
        # Read the text data with a custom delimiter and quoting
        text_df = pd.read_csv('../../training_text.csv', 
                             sep='\|\|',  # Using || as delimiter
                             engine='python',
                             quoting=csv.QUOTE_NONE,
                             encoding='utf-8')
        print("Data loaded successfully")
        
        # Merge the dataframes
        print("Merging data...")
        df = pd.merge(variants_df, text_df, on='ID', how='inner')
        
        # Feature engineering
        print("Creating features...")
        # Gene and Variation features
        gene_features = pd.get_dummies(df['Gene'], prefix='gene')
        variation_features = pd.get_dummies(df['Variation'], prefix='variation')
        
        # Text features
        tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
        text_features = tfidf.fit_transform(df['Text'].fillna(''))
        text_features_df = pd.DataFrame(text_features.toarray(), 
                                      columns=[f'text_{i}' for i in range(text_features.shape[1])])
        
        # Combine features
        X = pd.concat([gene_features, variation_features, text_features_df], axis=1)
        y = df['Class']
        
        # Create directory if it doesn't exist
        os.makedirs('saved_models', exist_ok=True)
        
        # Train model
        print("Training model...")
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X, y)
        
        # Save model and vectorizer
        print("Saving model and vectorizer...")
        model_path = os.path.join('saved_models', 'model.joblib')
        tfidf_path = os.path.join('saved_models', 'tfidf.joblib')
        
        joblib.dump(clf, model_path)
        joblib.dump(tfidf, tfidf_path)
        
        print(f"Model saved to {model_path}")
        print(f"Vectorizer saved to {tfidf_path}")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise e
        
        # Merge the dataframes
        df = pd.merge(variants_df, text_df, on='ID', how='inner')
        print("Data merged successfully")
        
        # Feature engineering
        print("Engineering features...")
        # Convert Gene and Variation to categorical features
        gene_features = pd.get_dummies(df['Gene'], prefix='gene')
        variation_features = pd.get_dummies(df['Variation'], prefix='variation')
        
        # Convert text to TF-IDF features
        tfidf = TfidfVectorizer(max_features=1000)
        text_features = tfidf.fit_transform(df['Text'])
        text_features_df = pd.DataFrame(text_features.toarray(), 
                                      columns=[f'text_{i}' for i in range(text_features.shape[1])])
        
        # Combine all features
        X = pd.concat([gene_features, variation_features, text_features_df], axis=1)
        y = df['Class']
        print("Features created successfully")
        
        # Create saved_models directory if it doesn't exist
        os.makedirs('saved_models', exist_ok=True)
        
        # Save the model
        model_path = os.path.join('saved_models', 'model.joblib')
        joblib.dump(model, model_path)
        print(f"Model successfully saved to {model_path}")
        
    except Exception as e:
        print(f"Error training/saving model: {str(e)}")

if __name__ == "__main__":
    train_and_save_model()