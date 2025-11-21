import requests
import json

def test_api(base_url="http://localhost:8000"):
    """Test the Personalized Medicine API."""
    
    print("Testing API Health...")
    try:
        response = requests.get(f"{base_url}/health")
        assert response.status_code == 200
        print("✓ Health check passed")
    except Exception as e:
        print(f"✗ Health check failed: {str(e)}")
        return
    
    # Test Cases
    test_cases = [
        {
            "gene": "BRCA1",
            "variation": "V600E",
            "text": "The BRCA1 mutation is a well-known variant associated with breast cancer risk.",
            "name": "Common Cancer Gene"
        },
        {
            "gene": "TP53",
            "variation": "R273C",
            "text": "This TP53 mutation leads to loss of tumor suppressor function.",
            "name": "Tumor Suppressor"
        }
    ]
    
    print("\nTesting Predictions...")
    for test in test_cases:
        print(f"\nTest Case: {test['name']}")
        try:
            response = requests.post(
                f"{base_url}/predict",
                json={
                    "gene": test["gene"],
                    "variation": test["variation"],
                    "text": test["text"]
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✓ Prediction successful")
                print(f"  Predicted Class: {result['predicted_class']}")
                print("  Top 3 Probabilities:")
                probs = result['class_probabilities']
                sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:3]
                for cls, prob in sorted_probs:
                    print(f"    {cls}: {prob:.3f}")
            else:
                print(f"✗ Prediction failed: {response.text}")
                
        except Exception as e:
            print(f"✗ Test failed: {str(e)}")
    
if __name__ == "__main__":
    test_api()