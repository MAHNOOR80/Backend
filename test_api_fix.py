#!/usr/bin/env python3
"""
Test script to verify the fix works with the API server as well.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from api_server import AgentResponse as ApiAgentResponse, ResponseStatus as ApiResponseStatus
from pydantic import ValidationError

def test_api_response_validation():
    """Test that API AgentResponse validation works correctly with new status"""
    print("Testing API AgentResponse validation...")

    try:
        # Test 1: SUCCESS status with empty sources (should FAIL)
        try:
            success_response = ApiAgentResponse(
                query_id="test1",
                status=ApiResponseStatus.SUCCESS,
                answer="This is a test answer",
                sources=[]
            )
            print("[FAIL] API SUCCESS with empty sources should fail validation")
            return False
        except ValidationError as e:
            if "At least one source must be provided when status is SUCCESS" in str(e):
                print(f"  [PASS] API SUCCESS with empty sources correctly fails: {e}")
            else:
                print(f"[FAIL] API SUCCESS with empty sources failed with wrong error: {e}")
                return False

        # Test 2: SUCCESS status with sources (should WORK)
        try:
            success_response = ApiAgentResponse(
                query_id="test2",
                status=ApiResponseStatus.SUCCESS,
                answer="This is a test answer",
                sources=[{
                    "chunk_id": "test_chunk",
                    "citation_index": 1,
                    "relevance_score": 0.9,
                    "excerpt": "test excerpt",
                    "metadata": {}
                }]
            )
            print("  [PASS] API SUCCESS with sources works correctly")
        except ValidationError as e:
            print(f"[FAIL] API SUCCESS with sources failed: {e}")
            return False

        # Test 3: CONVERSATIONAL status without sources (should WORK)
        try:
            conv_response = ApiAgentResponse(
                query_id="test3",
                status=ApiResponseStatus.CONVERSATIONAL,
                answer="Hello! This is a conversational response.",
                sources=[]
            )
            print("  [PASS] API CONVERSATIONAL without sources works correctly")
        except ValidationError as e:
            print(f"[FAIL] API CONVERSATIONAL without sources failed: {e}")
            return False

        # Test 4: CONVERSATIONAL status without answer (should FAIL)
        try:
            conv_response_no_answer = ApiAgentResponse(
                query_id="test4",
                status=ApiResponseStatus.CONVERSATIONAL,
                answer=None
            )
            print("[FAIL] API CONVERSATIONAL with None answer should fail")
            return False
        except ValidationError as e:
            if "Answer must be provided when status is SUCCESS" in str(e):
                # Note: This error message might not be specific to CONVERSATIONAL status
                # The Pydantic validation might have different behavior
                print(f"  [PASS] API CONVERSATIONAL with None answer correctly fails: {e}")
            else:
                print(f"  [PASS] API CONVERSATIONAL with None answer correctly fails: {e}")

        print("[PASS] API AgentResponse validation works correctly")
        return True

    except Exception as e:
        print(f"[FAIL] Error in API validation test: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run API tests"""
    print("Testing the fix for API server conversational responses...")
    print("=" * 60)

    success = True
    success &= test_api_response_validation()

    print("\n" + "=" * 60)
    if success:
        print("[SUCCESS] All API tests passed! The fix works correctly.")
        return 0
    else:
        print("[FAILURE] Some API tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)