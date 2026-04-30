import requests
import json

# The local URL where your Flask app is running
url = "http://127.0.0.1:5004/predict"

# Sample data based on your dataset image_f25f13.png
# Example: High traffic car count
test_data = {
    "Day of the week": "Tuesday",
    "CarCount": 14,
    "BikeCount": 16,
    "BusCount": 9,
    "TruckCount": 0,
    "Total": 39
}

print(f"Sending request to: {url}")
print(f"Input Data: {test_data}")

try:
    # Send the POST request
    response = requests.post(
        url, 
        data=json.dumps(test_data),
        headers={'Content-Type': 'application/json'}
    )

    # Print the results
    if response.status_code == 200:
        result = response.json()
        print("\n--- Prediction Result ---")
        print(f"Traffic Situation: {result['prediction']}")
        print(f"Confidence: {result['confidence']}")
    else:
        print(f"\nError: Received status code {response.status_code}")
        print(response.text)

except Exception as e:
    print(f"Failed to connect to the server: {e}")