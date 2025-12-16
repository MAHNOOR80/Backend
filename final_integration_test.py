#!/usr/bin/env python3
"""
Final integration test to verify both issues are resolved:
1. Greetings don't cause validation errors (the original "At least one source" error)
2. Different queries return different responses (the "same response for every query" issue)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agent import RetrievalAgent, ResponseStatus

def test_different_responses():
    """Test that different queries return different responses"""
    print("Testing that different queries return different responses...")

    agent = RetrievalAgent()

    # Test queries that should return different responses
    greeting_response = agent.query('hi')
    content_response = agent.query('what is robotics')  # This should be treated as content now

    print(f"Greeting response status: {greeting_response.status.value}")
    print(f"Content response status: {content_response.status.value}")

    # Verify greeting response
    assert greeting_response.status == ResponseStatus.CONVERSATIONAL, f"Expected CONVERSATIONAL, got {greeting_response.status.value}"
    assert greeting_response.answer is not None, "Greeting response should have an answer"
    print(f"Greeting response preview: '{greeting_response.answer[:50]}...'")

    # Verify content response is processed differently (either SUCCESS or INSUFFICIENT_CONTEXT)
    assert content_response.status in [ResponseStatus.SUCCESS, ResponseStatus.INSUFFICIENT_CONTEXT], f"Content query should not be CONVERSATIONAL, got {content_response.status.value}"
    print(f"Content response status: {content_response.status.value}")
    if content_response.status == ResponseStatus.INSUFFICIENT_CONTEXT:
        print("Content response: insufficient context (expected if no matching content found)")
    else:
        print(f"Content response preview: '{content_response.answer[:50] if content_response.answer else None}...'")

    # The key test: they should be different types of responses
    assert greeting_response.status != content_response.status or greeting_response.answer != content_response.answer, "Greeting and content responses should be different"

    print("[PASS] Different query types return different responses")
    return True

def test_no_validation_error():
    """Test that greetings don't cause the original validation error"""
    print("\nTesting that greetings don't cause validation errors...")

    agent = RetrievalAgent()

    # This should not raise "At least one source must be provided when status is SUCCESS"
    try:
        response = agent.query('hello')
        print(f"✓ Greeting response successful with status: {response.status.value}")
        print(f"  Answer: '{response.answer[:50]}...'")
        print(f"  Sources: {len(response.sources)} (should be 0 for conversational)")

        # Verify it's the new conversational status
        assert response.status == ResponseStatus.CONVERSATIONAL, f"Expected CONVERSATIONAL, got {response.status.value}"
        assert len(response.sources) == 0, "Conversational responses should have no sources"
        assert response.answer is not None, "Conversational responses should have an answer"

        print("✓ No validation error for greetings")
        return True
    except ValueError as e:
        if "At least one source must be provided when status is SUCCESS" in str(e):
            print(f"✗ Original validation error still occurs: {e}")
            return False
        else:
            print(f"✗ Different validation error: {e}")
            return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False

def test_content_queries_require_sources():
    """Test that actual content queries still require sources (when they succeed)"""
    print("\nTesting that content queries still require sources...")

    from agent import AgentResponse, ConfidenceScore, SourceReference, ChunkMetadata

    # Test that SUCCESS responses (when they happen) still require sources
    try:
        # This should fail validation because SUCCESS requires sources
        bad_response = AgentResponse(
            query_id="test",
            status=ResponseStatus.SUCCESS,
            answer="Test answer",
            sources=[],  # Empty sources - should fail
            confidence=ConfidenceScore()
        )
        print("✗ SUCCESS without sources should have failed validation")
        return False
    except ValueError as e:
        if "At least one source must be provided when status is SUCCESS" in str(e):
            print("✓ SUCCESS without sources correctly fails validation")
        else:
            print(f"✗ Wrong error for SUCCESS without sources: {e}")
            return False

    # Test that SUCCESS with sources works
    try:
        good_response = AgentResponse(
            query_id="test2",
            status=ResponseStatus.SUCCESS,
            answer="Test answer with sources",
            sources=[SourceReference(
                chunk_id="test_chunk",
                citation_index=1,
                relevance_score=0.9,
                excerpt="test excerpt",
                metadata=ChunkMetadata()
            )],
            confidence=ConfidenceScore()
        )
        print("✓ SUCCESS with sources works correctly")
    except Exception as e:
        print(f"✗ SUCCESS with sources failed: {e}")
        return False

    return True

def main():
    """Run final integration tests"""
    print("Final Integration Test: Both Issues Resolved")
    print("=" * 60)
    print("Issue 1: 'At least one source must be provided when status is SUCCESS' error")
    print("Issue 2: Same response for every query")
    print("=" * 60)

    success = True
    success &= test_different_responses()
    success &= test_no_validation_error()
    success &= test_content_queries_require_sources()

    print("\n" + "=" * 60)
    if success:
        print("[FINAL RESULT] ✓ All integration tests passed!")
        print("✓ Issue 1: Greeting validation error is fixed")
        print("✓ Issue 2: Different queries return different responses")
        print("✓ Content queries still require sources as expected")
        print("✓ Chatbot now works correctly for both greeting and content queries")
        return 0
    else:
        print("[FINAL RESULT] ✗ Some integration tests failed")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)