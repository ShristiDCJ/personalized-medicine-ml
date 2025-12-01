"""
Script to regenerate text_processor.joblib from training data
Run this after training to create the missing artifact
"""
import joblib
import re
import pandas as pd
from collections import Counter
import os
import csv
import sys

# Increase CSV field size limit
maxInt = sys.maxsize
while True:
    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt/10)

# Load training text data
print("Loading training text data...")
try:
    # Try reading with manual parsing to handle complex CSV
    data = []
    with open('../training_text.csv', 'r', encoding='utf-8', errors='ignore') as f:
        for line_num, line in enumerate(f):
            if line_num == 0:
                continue  # Skip header
            parts = line.strip().split('||')
            if len(parts) >= 2:
                text_id = parts[0]
                text_content = '||'.join(parts[1:])  # Rejoin in case || appears in text
                data.append({'ID': text_id, 'Text': text_content})
            if line_num % 100 == 0:
                print(f"Processed {line_num} lines...", end='\r')
    
    text_df = pd.DataFrame(data)
    print(f"\nLoaded {len(text_df)} text samples")
except Exception as e:
    print(f"Error loading training_text: {e}")
    print("Make sure training_text file exists in the parent directory")
    exit(1)

# Text cleaning function (must match training)
def clean_text(text):
    """Clean text - same as training"""
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text

# Build vocabulary
print("Building vocabulary...")
all_words = []
for text in text_df['Text']:
    cleaned = clean_text(text)
    words = cleaned.split()
    all_words.extend(words)

# Count word frequencies and take top words
word_counts = Counter(all_words)
vocab_size = 1923  # Must match model_config.json
most_common = word_counts.most_common(vocab_size - 2)  # Reserve 0 for <PAD>, 1 for <UNK>

# Create vocabulary mapping
vocab = ['<PAD>', '<UNK>'] + [word for word, _ in most_common]
word2idx = {word: idx for idx, word in enumerate(vocab)}

print(f"Vocabulary size: {len(vocab)}")
print(f"Sample words: {vocab[2:12]}")

# Save text processor
text_processor = {
    'vocab': vocab,
    'word2idx': word2idx
}

output_path = 'model/saved_models/text_processor.joblib'
joblib.dump(text_processor, output_path)
print(f"\n✅ Saved text_processor.joblib to {output_path}")
print(f"File size: {os.path.getsize(output_path) / (1024*1024):.2f} MB")
