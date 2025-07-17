#!/usr/bin/env python3
"""
Debug script to test LLM configuration
"""
import asyncio
import sys
sys.path.append('.')

# Import the rag module to ensure LLM is configured
import app.rag
from llama_index.core import Settings
from llama_index.core.llms import ChatMessage

async def test_llm():
    """Test LLM configuration"""
    print(f"LLM type: {type(Settings.llm)}")
    print(f"LLM: {Settings.llm}")
    
    try:
        response = await Settings.llm.achat([ChatMessage(role="user", content="Hello, respond with just 'Hi'")])
        print(f"LLM response: {response.message.content}")
    except Exception as e:
        print(f"LLM error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_llm())