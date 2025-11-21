import sys
import json
import joblib
import torch
import os
import re
import numpy as np
import torch.nn as nn

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

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\\s]', '', text)
    return text

def text_to_indices(text, word2idx, max_len):
    return [word2idx.get(w, 1) for w in text.split()[:max_len]] + [0] * (max_len - len(text.split()[:max_len]))

def predict(input_data):
    try:
        print("Starting prediction process...")
        print(f"Input data received: {input_data}")
        
        # Get the directory of the current script
        current_dir = os.path.dirname(os.path.abspath(__file__))
        print(f"Current directory: {current_dir}")
        
        # Load artifacts
        model_path = os.path.join(current_dir, 'saved_models', 'model.pth')
        text_processor_path = os.path.join(current_dir, 'saved_models', 'text_processor.joblib')
        le_gene_path = os.path.join(current_dir, 'saved_models', 'le_gene.joblib')
        le_var_path = os.path.join(current_dir, 'saved_models', 'le_variation.joblib')
        config_path = os.path.join(current_dir, 'saved_models', 'model_config.json')
        
        print(f"Checking for model files...")
        
        # Check if all files exist
        required_files = [model_path, text_processor_path, le_gene_path, le_var_path, config_path]
        for file in required_files:
            print(f"Checking file: {file}")
            if not os.path.exists(file):
                error_msg = f'Required file not found: {os.path.basename(file)}'
                print(f"Error: {error_msg}")
                return {
                    'error': error_msg,
                    'success': False
                }
        print("All required files found")
        
        try:
            print("Loading config...")
            with open(config_path, 'r') as f:
                config = json.load(f)
            print("Config loaded successfully")
            
            print("Loading processors...")
            text_processor = joblib.load(text_processor_path)
            le_gene = joblib.load(le_gene_path)
            le_var = joblib.load(le_var_path)
            print("All processors loaded successfully")
        except Exception as e:
            error_msg = f"Error loading model files: {str(e)}"
            print(error_msg)
            return {
                'error': error_msg,
                'success': False
            }
        
        # Initialize model
        model = PersonalizedMedicineNet(config['vocab_size'])
        model.load_state_dict(torch.load(model_path))
        model.eval()
        
        # Process input
        try:
            print("Processing input data...")
            gene = input_data['geneticData'].split()[0]
            variation = ' '.join(input_data['geneticData'].split()[1:])
            text = clean_text(input_data['clinicalData'])
            print(f"Processed input - Gene: {gene}, Variation: {variation}")
        except Exception as e:
            error_msg = f"Error processing input data: {str(e)}"
            print(error_msg)
            return {
                'error': error_msg,
                'success': False
            }
        
        # Convert to features
        text_indices = text_to_indices(text, text_processor['word2idx'], text_processor['max_len'])
        gene_encoded = le_gene.transform([gene])[0]
        var_encoded = le_var.transform([variation])[0]
        
        # Convert to tensors
        text_tensor = torch.tensor([text_indices], dtype=torch.long)
        tab_tensor = torch.tensor([[gene_encoded, var_encoded]], dtype=torch.float)
        
        # Make prediction
        with torch.no_grad():
            output = model(text_tensor, tab_tensor)
            probabilities = torch.softmax(output, dim=1)
            prediction = output.argmax(1)
            
        # Convert prediction to class number (add 1 since classes are 1-9)
        pred_class = prediction.item() + 1
        probs = probabilities[0].tolist()
        
        # Define class meanings
        class_meanings = {
            1: "Likely Loss-of-function",
            2: "Likely Gain-of-function",
            3: "Likely Neutral",
            4: "Loss-of-function",
            5: "Gain-of-function",
            6: "Neutral",
            7: "Inconclusive",
            8: "Likely Pathogenic",
            9: "Pathogenic"
        }
        
        # Format probabilities as percentages
        formatted_probs = {
            class_meanings[i+1]: f"{prob * 100:.2f}%" 
            for i, prob in enumerate(probs)
        }
        
        # Get prediction explanation
        prediction_text = class_meanings[pred_class]
        
        return {
            'prediction_class': pred_class,
            'prediction_text': prediction_text,
            'class_probabilities': formatted_probs,
            'input_summary': {
                'gene': gene,
                'variation': variation,
                'clinical_summary': text[:200] + "..." if len(text) > 200 else text
            },
            'success': True
        }
    except Exception as e:
        return {
            'error': str(e),
            'success': False
        }

if __name__ == "__main__":
    # Get input data from command line argument
    input_data = json.loads(sys.argv[1])
    
    # Make prediction
    result = predict(input_data)
    
    # Print result as JSON
    print(json.dumps(result))