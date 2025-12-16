#!/usr/bin/env python3
"""
Test script to verify the API server metadata validation works with 0.0 values for conversational responses.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from api_server import AgentResponse as ApiAgentResponse, ResponseStatus as ApiResponseStatus, ResponseMetadata as ApiResponseMetadata
from pydantic import ValidationError

def test_api_metadata_validation():
    """Test that API ResponseMetadata allows 0.0 values for conversational responses"""
    print("Testing API ResponseMetadata validation...")

    try:
        # Test 1: Metadata with 0.0 values (should work for conversational responses)
        try:
            metadata = ApiResponseMetadata(
                model="test-model",
                temperature=0.2,
                total_time_ms=0.0,  # This was causing the error before
                retrieval_time_ms=0.0,  # This was causing the error before
                generation_time_ms=0.0,  # This was causing the error before
                timestamp="2025-12-15T21:00:00.000Z"
            )
            print("  [PASS] API ResponseMetadata with 0.0 values works correctly")
        except ValidationError as e:
            print(f"[FAIL] API ResponseMetadata with 0.0 values failed: {e}")
            return False

        # Test 2: Complete API response with conversational status and 0.0 metadata (should work)
        try:
            response = ApiAgentResponse(
                query_id="test1",
                status=ApiResponseStatus.CONVERSATIONAL,
                answer="Hello! This is a conversational response.",
                sources=[],
                confidence={
                    "retrieval_quality": 1.0,
                    "coverage_score": 1.0,
                    "entailment_score": 1.0,
                    "lexical_overlap": 1.0,
                    "overall": 1.0,
                    "level": "high"
                },
                metadata={
                    "model": "test-model",
                    "temperature": 0.2,
                    "total_time_ms": 10.0,  # Non-zero for actual response
                    "retrieval_time_ms": 0.0,  # Zero for conversational (no retrieval)
                    "generation_time_ms": 10.0,  # Non-zero for actual response
                    "timestamp": "2025-12-15T21:00:00.000Z"
                }
            )
            print("  [PASS] Complete API conversational response works correctly")
        except ValidationError as e:
            print(f"[FAIL] Complete API conversational response failed: {e}")
            return False

        # Test 3: Complete API response with success status and proper metadata (should work)
        try:
            response = ApiAgentResponse(
                query_id="test2",
                status=ApiResponseStatus.SUCCESS,
                answer="This is a textbook-based answer.",
                sources=[{
                    "chunk_id": "test_chunk",
                    "citation_index": 1,
                    "relevance_score": 0.9,
                    "excerpt": "test excerpt",
                    "metadata": {}
                }],
                confidence={
                    "retrieval_quality": 0.9,
                    "coverage_score": 0.8,
                    "entailment_score": 0.9,
                    "lexical_overlap": 0.7,
                    "overall": 0.825,
                    "level": "high"
                },
                metadata={
                    "model": "test-model",
                    "temperature": 0.2,
                    "total_time_ms": 1000.0,
                    "retrieval_time_ms": 800.0,
                    "generation_time_ms": 200.0,
                    "timestamp": "2025-12-15T21:00:00.000Z"
                }
            )
            print("  [PASS] Complete API success response works correctly")
        except ValidationError as e:
            print(f"[FAIL] Complete API success response failed: {e}")
            return False

        print("[PASS] API metadata validation works correctly")
        return True

    except Exception as e:
        print(f"[FAIL] Error in API metadata test: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run API metadata tests"""
    print("Testing the fix for API server metadata validation...")
    print("=" * 60)

    success = True
    success &= test_api_metadata_validation()

    print("\n" + "=" * 60)
    if success:
        print("[SUCCESS] All API metadata tests passed! The fix works correctly.")
        return 0
    else:
        print("[FAILURE] Some API metadata tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)