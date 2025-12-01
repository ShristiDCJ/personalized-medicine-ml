import torch
import torch.nn as nn
import joblib
import re
import os
from typing import Dict, Any
import numpy as np

# Constants
MAX_LEN = 256

# Model Architecture (must match training)
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

class MutationPredictor:
    def __init__(self, model_dir: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_dir = model_dir
        self._load_artifacts()

    def _load_artifacts(self):
        """Load all required model artifacts."""
        # Define paths to model artifacts in saved_models directory
        saved_models_dir = os.path.join(self.model_dir, 'model', 'saved_models')
        model_path = os.path.join(self.model_dir, 'model', 'model.pth')
        
        # Check if text_processor exists, otherwise use model_config
        text_processor_path = os.path.join(saved_models_dir, 'text_processor.joblib')
        if os.path.exists(text_processor_path):
            text_processor = joblib.load(text_processor_path)
            self.word2idx = text_processor['word2idx']
            self.vocab = text_processor['vocab']
        else:
            # Fallback: Load vocab size from model_config.json
            import json
            config_path = os.path.join(saved_models_dir, 'model_config.json')
            with open(config_path, 'r') as f:
                config = json.load(f)
            vocab_size = config['vocab_size']
            # Create minimal vocab - this needs to be regenerated properly
            print(f"WARNING: text_processor.joblib not found. Using vocab_size={vocab_size} from config.")
            print("Note: Predictions may not work correctly without the actual vocabulary mapping.")
            self.vocab = list(range(vocab_size))
            self.word2idx = {str(i): i for i in range(vocab_size)}
        
        # Load label encoders
        self.le_gene = joblib.load(os.path.join(saved_models_dir, 'le_gene.joblib'))
        self.le_var = joblib.load(os.path.join(saved_models_dir, 'le_variation.joblib'))
        
        # Initialize and load model
        self.model = PersonalizedMedicineNet(len(self.vocab))
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def _clean_text(self, text: str) -> str:
        """Clean text using same preprocessing as training."""
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', '', text)
        return text

    def _text_to_indices(self, text: str) -> list:
        """Convert text to padded sequence of indices."""
        words = self._clean_text(text).split()
        indices = [self.word2idx.get(w, 1) for w in words[:MAX_LEN]]  # 1 is <UNK>
        padding = [0] * (MAX_LEN - len(indices))  # 0 is <PAD>
        return indices + padding

    def predict(self, gene: str, variation: str, text: str) -> Dict[str, Any]:
        """
        Make a prediction for a single mutation.
        
        Args:
            gene: Gene name (e.g., "BRCA1")
            variation: Variation (e.g., "V600E")
            text: Clinical text about the mutation
            
        Returns:
            Dictionary containing prediction probabilities and class
        """
        try:
            # Encode gene and variation
            gene_idx = self.le_gene.transform([gene])[0]
            var_idx = self.le_var.transform([variation])[0]
            tabular = torch.tensor([[gene_idx, var_idx]], dtype=torch.float).to(self.device)
            
            # Encode text
            text_indices = self._text_to_indices(text)
            text_tensor = torch.tensor([text_indices], dtype=torch.long).to(self.device)
            
            # Make prediction
            with torch.no_grad():
                output = self.model(text_tensor, tabular)
                probs = torch.softmax(output, dim=1).cpu().numpy()[0]
                pred_class = int(output.argmax(1).item()) + 1  # Add 1 since classes are 1-9
            
            return {
                'predicted_class': pred_class,
                'class_probabilities': {f'Class_{i+1}': float(p) for i, p in enumerate(probs)},
                'gene': gene,
                'variation': variation
            }
            
        except Exception as e:
            return {'error': str(e)}

if __name__ == '__main__':
    # Example usage
    model_dir = os.path.dirname(os.path.abspath(__file__))
    predictor = MutationPredictor(model_dir)
    
    # Example prediction
    sample = {
        'gene': 'BRCA1',
        'variation': 'V600E',
        'text': 'The BRCA1 mutation is a well-known variant associated with breast cancer risk.'
    }
    
    result = predictor.predict(**sample)
    print(result)