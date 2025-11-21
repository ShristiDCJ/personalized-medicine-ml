import os
import torch
import joblib

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model', 'model.pth')
TEXT_PROC_PATH = os.path.join(os.path.dirname(__file__), 'text_processor.pkl')
LE_GENE_PATH = os.path.join(os.path.dirname(__file__), 'le_gene.pkl')
LE_VAR_PATH = os.path.join(os.path.dirname(__file__), 'le_variation.pkl')

print('\nChecking artifact files in backend/...')
print('Model path:', MODEL_PATH)
print('Text processor path:', TEXT_PROC_PATH)
print('Label encoder gene path:', LE_GENE_PATH)
print('Label encoder variation path:', LE_VAR_PATH)

def exists(p):
    print(f"  {p} ->", os.path.exists(p))

exists(MODEL_PATH)
exists(TEXT_PROC_PATH)
exists(LE_GENE_PATH)
exists(LE_VAR_PATH)

# Try loading model
if os.path.exists(MODEL_PATH):
    try:
        state = torch.load(MODEL_PATH, map_location='cpu')
        if isinstance(state, dict):
            print('\nModel file loaded as dict. Keys:')
            print(' ', list(state.keys())[:20])
        else:
            print('\nModel file loaded (type: {}).'.format(type(state)))
    except Exception as e:
        print('\nError loading model.pth:')
        print(' ', e)

# Try loading text processor
if os.path.exists(TEXT_PROC_PATH):
    try:
        tp = joblib.load(TEXT_PROC_PATH)
        print('\nText processor loaded. Type:', type(tp))
        if isinstance(tp, dict):
            print(' Keys:', list(tp.keys()))
    except Exception as e:
        print('\nError loading text_processor.pkl:')
        print(' ', e)

# Try loading label encoders
for p, name in [(LE_GENE_PATH, 'le_gene'), (LE_VAR_PATH, 'le_variation')]:
    if os.path.exists(p):
        try:
            obj = joblib.load(p)
            print(f"\n{name} loaded. Type: {type(obj)}")
        except Exception as e:
            print(f"\nError loading {name}:")
            print(' ', e)

print('\nCheck complete.')
