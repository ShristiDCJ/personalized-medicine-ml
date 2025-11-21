import requests
import json

url = 'http://127.0.0.1:8000/predict'
payload = {
    "gene": "TP53",
    "variation": "R273C",
    "text": "Missense mutation in the TP53 tumor suppressor gene changing arginine to cysteine at codon 273."
}

try:
    r = requests.post(url, json=payload, timeout=15)
    print('Status code:', r.status_code)
    print('Response body:')
    print(r.text)
except Exception as e:
    print('Request error:', e)
