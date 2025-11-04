# Simple test script that hits the running API. Use manually or as a pytest test.
import requests
import json

BASE = "http://localhost:8000/v1"

sample = {
    "step": 743,
    "type": "CASH_OUT",
    "amount": 181.0,
    "nameOrig": "C840083671",
    "oldbalanceOrg": 181.0,
    "newbalanceOrig": 0.0,
    "nameDest": "C38997010",
    "oldbalanceDest": 21182.0,
    "newbalanceDest": 0.0,
}

def test_predict_manual():
    resp = requests.post(f"{BASE}/predict", json=sample, timeout=10)
    print("status:", resp.status_code)
    try:
        print(json.dumps(resp.json(), indent=2))
    except Exception:
        print(resp.text)

if __name__ == "__main__":
    test_predict_manual()
