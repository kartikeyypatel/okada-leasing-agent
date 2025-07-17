#!/usr/bin/env python3
"""
Debug script to test the full recommendation workflow
"""
import asyncio
import sys
sys.path.append('.')

# Import the rag module to ensure LLM is configured
import app.rag
from app.recommendation_workflow_manager import recommendation_workflow_manager

async def test_workflow():
    """Test the full recommendation workflow"""
    user_id = "test@example.com"
    message = "Suggest me a property"
    
    print(f"Testing workflow for user: {user_id}")
    print(f"Message: '{message}'")
    
    try:
        # Start workflow
        print("Starting workflow...")
        workflow_session = await recommendation_workflow_manager.start_recommendation_workflow(user_id, message)
        print(f"Workflow session created: {workflow_session.session_id}")
        print(f"Current step: {workflow_session.current_step}")
        
        # Get next step
        print("Getting next step...")
        next_step = await recommendation_workflow_manager.get_next_step(workflow_session.session_id)
        if next_step:
            print(f"Next step: {next_step.step_name}")
            print(f"Success: {next_step.success}")
            print(f"Response: {next_step.response_message}")
        else:
            print("No next step returned")
            
    except Exception as e:
        print(f"Error in workflow: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_workflow())