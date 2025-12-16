#!/usr/bin/env python3
"""
Test script to verify the fix for conversational responses in the RAG agent.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agent import RetrievalAgent, ResponseStatus

def test_greeting_response():
    """Test that greeting responses work without sources requirement"""
    print("Testing greeting response...")

    try:
        # Create a mock agent (we'll test the logic without actual Qdrant connection)
        agent = RetrievalAgent.__new__(RetrievalAgent)  # Create without calling __init__

        # Test the greeting detection logic
        greeting_queries = ["hi", "hello", "how are you", "who are you", "bye", "thanks"]

        for query in greeting_queries:
            is_greeting = agent._is_greeting_or_low_intent(query)
            print(f"  Query: '{query}' -> is_greeting: {is_greeting}")
            assert is_greeting, f"Expected '{query}' to be detected as greeting"

        # Test the greeting response handler
        greeting_response = agent._handle_greeting_query("hi")
        print(f"  Greeting response: '{greeting_response[:50]}...'")
        assert len(greeting_response) > 0, "Greeting response should not be empty"

        print("[PASS] Greeting detection and handling works correctly")
        return True

    except Exception as e:
        print(f"[FAIL] Error in greeting test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_agent_response_validation():
    """Test that AgentResponse validation works correctly with new status"""
    print("\nTesting AgentResponse validation...")

    from agent import AgentResponse, ConfidenceScore, SourceReference, ChunkMetadata

    try:
        # Test 1: SUCCESS status with empty sources (should FAIL)
        try:
            success_response = AgentResponse(
                query_id="test1",
                status=ResponseStatus.SUCCESS,
                answer="This is a test answer",
                sources=[],
                confidence=ConfidenceScore()
            )
            print("[FAIL] SUCCESS with empty sources should fail validation")
            return False
        except ValueError as e:
            print(f"  [PASS] SUCCESS with empty sources correctly fails: {e}")

        # Test 2: SUCCESS status with sources (should WORK)
        try:
            success_response = AgentResponse(
                query_id="test2",
                status=ResponseStatus.SUCCESS,
                answer="This is a test answer",
                sources=[SourceReference(
                    chunk_id="test_chunk",
                    citation_index=1,
                    relevance_score=0.9,
                    excerpt="test excerpt",
                    metadata=ChunkMetadata()
                )],
                confidence=ConfidenceScore()
            )
            print("  [PASS] SUCCESS with sources works correctly")
        except ValueError as e:
            print(f"[FAIL] SUCCESS with sources failed: {e}")
            return False

        # Test 3: CONVERSATIONAL status without sources (should WORK)
        try:
            conv_response = AgentResponse(
                query_id="test3",
                status=ResponseStatus.CONVERSATIONAL,
                answer="Hello! This is a conversational response.",
                sources=[],
                confidence=ConfidenceScore()
            )
            print("  [PASS] CONVERSATIONAL without sources works correctly")
        except ValueError as e:
            print(f"[FAIL] CONVERSATIONAL without sources failed: {e}")
            return False

        # Test 4: CONVERSATIONAL status without answer (should FAIL)
        try:
            conv_response_no_answer = AgentResponse(
                query_id="test4",
                status=ResponseStatus.CONVERSATIONAL,
                answer=None,
                sources=[],
                confidence=ConfidenceScore()
            )
            print("[FAIL] CONVERSATIONAL with None answer should fail")
            return False
        except ValueError as e:
            print(f"  [PASS] CONVERSATIONAL with None answer correctly fails: {e}")

        print("[PASS] AgentResponse validation works correctly")
        return True

    except Exception as e:
        print(f"[FAIL] Error in validation test: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("Testing the fix for conversational responses...")
    print("=" * 50)

    success = True
    success &= test_greeting_response()
    success &= test_agent_response_validation()

    print("\n" + "=" * 50)
    if success:
        print("[SUCCESS] All tests passed! The fix works correctly.")
        return 0
    else:
        print("[FAILURE] Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
