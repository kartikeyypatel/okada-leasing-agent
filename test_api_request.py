#!/usr/bin/env python3
"""
Test script to make API requests to the recommendation system
"""
import requests
import json

def test_recommendation_request():
    """Test the recommendation system via API"""
    url = "http://localhost:8000/api/chat"
    
    payload = {
        "user_id": "test@example.com",
        "message": "Suggest me a property",
        "history": []
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    print(f"Making request to: {url}")
    print(f"Payload: {json.dumps(payload, indent=2)}")
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        
        return response.json()
    except Exception as e:
        print(f"Error making request: {e}")
        return None

if __name__ == "__main__":
    test_recommendation_request()