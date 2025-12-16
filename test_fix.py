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

    # Create a mock agent (we'll test the logic without actual Qdrant connection)
    try:
        # We'll test the internal methods directly to make sure they work
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

    except Exception as e:
        print(f"[FAIL] Error in greeting test: {e}")
        return False

    return True

def test_agent_response_validation():
    """Test that AgentResponse validation works correctly with new status"""
    print("\nTesting AgentResponse validation...")

    from agent import AgentResponse, ConfidenceScore, ResponseMetadata

    try:
        # Test SUCCESS status with sources (should work)
        try:
            success_response = AgentResponse(
                query_id="test1",
                status=ResponseStatus.SUCCESS,
                answer="This is a test answer",
                sources=[],
                confidence=ConfidenceScore()
            )
            print("✗ SUCCESS with empty sources should fail validation")
            return False
        except ValueError as e:
            print(f"  ✓ SUCCESS with empty sources correctly fails: {e}")

        # Test SUCCESS status with sources (should work)
        try:
            from agent import SourceReference, ChunkMetadata
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
            print("  ✓ SUCCESS with sources works correctly")
        except ValueError as e:
            print(f"✗ SUCCESS with sources failed: {e}")
            return False

        # Test CONVERSATIONAL status without sources (should work)
        try:
            conv_response = AgentResponse(
                query_id="test3",
                status=ResponseStatus.CONVERSATIONAL,
                answer="Hello! This is a conversational response.",
                sources=[],
                confidence=ConfidenceScore()
            )
            print("  ✓ CONVERSATIONAL without sources works correctly")
        except ValueError as e:
            print(f"✗ CONVERSATIONAL without sources failed: {e}")
            return False

        # Test CONVERSATIONAL status with answer required
        try:
            conv_response_no_answer = AgentResponse(
                query_id="test4",
                status=ResponseStatus.CONVERSATIONAL,
                answer=None,
                sources=[],
                confidence=ConfidenceScore()
            )
            print("✗ CONVERSATIONAL with None answer should fail")
            return False
        except ValueError as e:
            print(f"  ✓ CONVERSATIONAL with None answer correctly fails: {e}")

        print("✓ AgentResponse validation works correctly")

    except Exception as e:
        print(f"✗ Error in validation test: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True

def main():
    """Run all tests"""
    print("Testing the fix for conversational responses...")
    print("=" * 50)

    success = True
    success &= test_greeting_response()
    success &= test_agent_response_validation()

    print("\n" + "=" * 50)
    if success:
        print("✓ All tests passed! The fix should work correctly.")
    else:
        print("✗ Some tests failed. Please check the implementation.")

    return success

if __name__ == "__main__":
    main()