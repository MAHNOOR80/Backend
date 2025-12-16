#!/usr/bin/env python3
"""
Test script for the fixed RAG agent.
This script tests the main functionality of the agent to ensure it properly retrieves
full content and generates grounded responses.
"""

import os
import sys
from agent import RetrievalAgent

def test_agent():
    """Test the fixed agent functionality"""
    print("Testing the fixed RAG agent...")

    # Check if required environment variables are set
    required_env_vars = ['GOOGLE_API_KEY', 'COHERE_API_KEY', 'QDRANT_URL', 'QDRANT_API_KEY']
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]

    if missing_vars:
        print(f"Missing required environment variables: {missing_vars}")
        print("Please set these variables before running the test.")
        return False

    try:
        # Initialize the agent with the fixed configuration
        print("Initializing agent...")
        agent = RetrievalAgent(
            model="gemini-pro",
            top_k=5,
            score_threshold=0.5  # Lower threshold for testing
        )
        print("Agent initialized successfully!")

        # Test retrieval first to see if we're getting any results
        print("Testing retrieval...")
        try:
            retrieval_result = agent.retrieve_information("humanoid robotics", top_k=5, score_threshold=0.3)
            print(f"Retrieved {len(retrieval_result.chunks)} chunks")

            if retrieval_result.chunks:
                print("Sample retrieved chunk:")
                first_chunk = retrieval_result.chunks[0]
                print(f"  Text length: {len(first_chunk.text)}")
                print(f"  Text preview: {first_chunk.text[:200]}...")
                print(f"  Similarity score: {first_chunk.similarity_score}")
            else:
                print("No chunks retrieved - this is likely the issue!")
                print("Check: 1) Qdrant collection name, 2) Qdrant connection, 3) Collection has data")
                return False
        except Exception as e:
            print(f"Error during retrieval: {e}")
            print("This could be due to Qdrant connection issues or incompatible Qdrant client version")
            import traceback
            traceback.print_exc()
            return False

        # Test query
        test_query = "What is inverse kinematics in humanoid robotics?"
        print(f"\nTesting query: '{test_query}'")

        response = agent.query(test_query)

        print(f"Response status: {response.status.value}")

        if response.status == "success":
            print(f"Answer: {response.answer}")
            print(f"Confidence: {response.confidence.level} ({response.confidence.overall:.2f})")
            print(f"Sources: {len(response.sources)}")

            if response.sources:
                print("\nSources:")
                for source in response.sources:
                    print(f"  [{source.citation_index}] Score: {source.relevance_score:.2f}")
                    print(f"      Excerpt: {source.excerpt[:100]}...")
        elif response.status == "insufficient_context":
            print(f"Insufficient context: {response.error_message}")
        elif response.status == "error":
            print(f"Error: {response.error_message}")

        print(f"\nTotal response time: {response.metadata.total_time_ms:.2f}ms")
        return True

    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_agent()
    if success:
        print("\n✓ Agent test completed successfully!")
    else:
        print("\n✗ Agent test failed!")
        sys.exit(1)