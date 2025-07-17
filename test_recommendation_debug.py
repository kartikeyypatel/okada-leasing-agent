#!/usr/bin/env python3
"""
Debug script to test the recommendation system
"""
import asyncio
import sys
import os
sys.path.append('.')

from app.intent_detection import intent_detection_service

async def test_intent_detection():
    """Test intent detection with a simple message"""
    message = "Suggest me a property"
    print(f"Testing message: '{message}'")
    
    try:
        result = await intent_detection_service.detect_recommendation_intent(message)
        print(f"Intent detected: {result.is_recommendation_request}")
        print(f"Confidence: {result.confidence}")
        print(f"Preferences: {result.initial_preferences}")
        print(f"Triggers: {result.trigger_phrases}")
        return result
    except Exception as e:
        print(f"Error in intent detection: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    asyncio.run(test_intent_detection())