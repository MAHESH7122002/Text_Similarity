import requests

url = 'http://127.0.0.1:5000/predict_api'
data = {
    "text1": "text1_here",
    "text2": "text2_here"
}
response = requests.post(url, json=data)
output = response.json()
print("Similarity score:", output)
