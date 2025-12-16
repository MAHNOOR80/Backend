#!/usr/bin/env python3
"""
Test to verify that the query classification fix works correctly.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agent import RetrievalAgent

def test_query_classification():
    """Test that queries are classified correctly"""
    print("Testing query classification fix...")

    agent = RetrievalAgent()

    # Test cases: (query, expected_classification, description)
    test_cases = [
        # Greetings should still be classified as greetings
        ("hi", True, "Simple greeting"),
        ("hello", True, "Simple greeting"),
        ("how are you", True, "Common greeting phrase"),
        ("what are you", True, "Direct question about agent"),
        ("who are you", True, "Identity question"),
        ("bye", True, "Farewell"),
        ("thanks", True, "Gratitude"),

        # Content queries should NOT be classified as greetings
        ("what is robotics", False, "Content question starting with 'what'"),
        ("what is humanoid locomotion", False, "Content question starting with 'what is'"),
        ("explain inverse kinematics", False, "Technical explanation request"),
        ("tell me about gait planning", False, "Content request"),
        ("what is the difference between forward and inverse kinematics", False, "Complex technical question"),
        ("hello world programming", False, "Non-greeting using 'hello'"),
        ("hi robot", False, "Non-greeting using 'hi'"),
        ("how does inverse kinematics work", False, "Technical question using 'how'"),
    ]

    all_passed = True
    for query, expected, description in test_cases:
        result = agent._is_greeting_or_low_intent(query)
        status = "[PASS]" if result == expected else "[FAIL]"
        print(f"  {status} {description}: '{query}' -> {result} (expected {expected})")
        if result != expected:
            all_passed = False

    return all_passed

def main():
    print("Testing query classification after fix")
    print("=" * 50)

    success = test_query_classification()

    print("\n" + "=" * 50)
    if success:
        print("[SUCCESS] All tests passed! Query classification is working correctly.")
        print("  - Greetings are still classified as greetings")
        print("  - Content queries are no longer misclassified as greetings")
        print("  - The original issue is fixed")
    else:
        print("[FAILURE] Some tests failed. Query classification still has issues.")

    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())