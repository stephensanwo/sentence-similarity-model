import requests
import os

token = os.environ.get('token')
print(token)

API_URL = "https://api-inference.huggingface.co/models/sentence-transformers/paraphrase-xlm-r-multilingual-v1"
headers = {"Authorization": f"Bearer {token}"}


def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()


output = query({
    "inputs": {
        "source_sentence": "Cash transfer to Stephen Sanwo",
        "sentences": [
            "Stephen Sanwo cash transfer",
            "cash payment to Adeolu Taiwo",
            "Funds transfer to Hugging Face",
            "Procurement of data from MTN",
            "Professional service fee KPMG",
               "Staff Advance Stephen Sanwo"
               ]
    },
})

print(output)
