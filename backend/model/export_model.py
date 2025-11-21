import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import re
import joblib
import json

# -------------------------------
# Model Architecture
# -------------------------------
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, num_filters=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv = nn.Conv1d(embed_dim, num_filters, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(num_filters, 64)

    def forward(self, x):
        x = self.embedding(x).transpose(1, 2)
        x = torch.relu(self.conv(x))
        x = self.pool(x).squeeze(2)
        return self.fc(x)

class PersonalizedMedicineNet(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.text_cnn = TextCNN(vocab_size)
        self.tab_net = nn.Sequential(
            nn.Linear(2, 32), nn.ReLU(), nn.Dropout(0.3), nn.Linear(32, 64)
        )
        self.classifier = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 9)
        )

    def forward(self, text, tab):
        t = self.text_cnn(text)
        s = self.tab_net(tab)
        x = torch.cat([t, s], dim=1)
        return self.classifier(x)

# -------------------------------
# Data Processing Functions
# -------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\\s]', '', text)
    return text

print("Loading and preprocessing data...")
# Load data
def safe_load_text(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or '||' not in line: continue
            parts = line.split('||', 1)
            if len(parts) != 2: continue
            id_part, text = parts
            try:
                id_val = int(id_part.split(',')[-1])
                data.append([id_val, text])
            except: continue
    return pd.DataFrame(data, columns=['ID', 'Text'])

variants = pd.read_csv('../../training_variants.csv')
text_df = safe_load_text('../../training_text.csv')
df = pd.merge(variants, text_df, on='ID', how='left')
df['Text'] = df['Text'].fillna('')
df['Text'] = df['Text'].apply(clean_text)

# Build vocabulary
print("Building vocabulary...")
all_words = ' '.join(df['Text']).split()
vocab = ['<PAD>', '<UNK>'] + [w for w, c in Counter(all_words).most_common(10000)]
word2idx = {w: i for i, w in enumerate(vocab)}
VOCAB_SIZE = len(vocab)
MAX_LEN = 256

# Prepare label encoders
print("Preparing encoders...")
le_gene = LabelEncoder()
le_var = LabelEncoder()
le_gene.fit(df['Gene'])
le_var.fit(df['Variation'])

# Initialize model
print("Initializing model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PersonalizedMedicineNet(VOCAB_SIZE).to(device)

# Train model (or load pre-trained weights)
# ... (training code here if needed)

# Save artifacts
print("Saving artifacts...")
torch.save(model.state_dict(), 'saved_models/model.pth')
joblib.dump({
    'word2idx': word2idx,
    'vocab': vocab,
    'max_len': MAX_LEN
}, 'saved_models/text_processor.joblib')
joblib.dump(le_gene, 'saved_models/le_gene.joblib')
joblib.dump(le_var, 'saved_models/le_variation.joblib')

# Save a config file
config = {
    'vocab_size': VOCAB_SIZE,
    'max_len': MAX_LEN,
    'device': 'cpu'  # We'll use CPU for inference
}
with open('saved_models/model_config.json', 'w') as f:
    json.dump(config, f)

print("Done! Saved all necessary artifacts for inference.")