import requests
import numpy as np

NAOMI_IP = "localhost" 

def test_endpoint():

    test_data = np.random.normal(0, 1, (5, 76))
    
    print("Input shape:", test_data.shape)
    print("Sending request to model endpoint...")
    
    try:
        response = requests.post(
            f"http://{NAOMI_IP}/ray-api/nancy/",
            json=test_data.tolist(), 
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            predictions = np.array(result["predictions"])
            print("\nSuccess!")
            print("Predictions shape:", predictions.shape)
            print("\nPrediction (probabilities):", predictions[0])
        else:
            print(f"\nError: Status code {response.status_code}")
            print("Response:", response.text)
    
    except Exception as e:
        print(f"\nError connecting to endpoint: {str(e)}")

if __name__ == "__main__":
    print("Testing Nancy model endpoint...")
    test_endpoint()