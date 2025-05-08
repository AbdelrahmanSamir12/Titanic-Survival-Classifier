import requests

url = "http://localhost:8000/predict"  # Change to your server address
data = {
    "Pclass": 3,
    "Sex": "male",
    "Age": 29,
    "SibSp": 0,
    "Parch": 0,
    "Fare": 7.25,
    "Embarked": "S"
}

# If your server expects {"input": {...}}, wrap as such:
# data = {"input": data}

response = requests.post(url, json=data)
print(response.json())