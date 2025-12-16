#!/usr/bin/env python3
"""
Debug script to test Qdrant connection and collection directly
"""

import os
from qdrant_client import QdrantClient

def test_qdrant_connection():
    """Test direct Qdrant connection and check collection"""
    print("Testing Qdrant connection...")

    # Check environment variables
    qdrant_url = os.getenv('QDRANT_URL')
    qdrant_api_key = os.getenv('QDRANT_API_KEY')

    if not qdrant_url or not qdrant_api_key:
        print("❌ Missing QDRANT_URL or QDRANT_API_KEY environment variables")
        return False

    try:
        # Initialize Qdrant client
        client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key,
            timeout=10.0
        )

        print("✅ Qdrant client initialized successfully")

        # List all collections to see what's available
        collections = client.get_collections()
        print(f"Available collections: {[coll.name for coll in collections.collections]}")

        # Check if rag_embedding collection exists
        try:
            collection_info = client.get_collection("rag_embedding")
            print(f"✅ Collection 'rag_embedding' found!")
            print(f"Points count: {collection_info.points_count}")
            print(f"Config: {collection_info.config}")
        except Exception as e:
            print(f"❌ Collection 'rag_embedding' not found: {e}")
            return False

        # Try to get a few sample points to verify data exists
        try:
            sample_points = client.scroll(
                collection_name="rag_embedding",
                limit=2,
                with_payload=True,
                with_vectors=False
            )

            points = list(sample_points[0])  # scroll returns (points, next_offset)
            if points:
                print(f"✅ Found {len(points)} sample points in the collection")
                for i, point in enumerate(points):
                    print(f"Sample point {i+1}:")
                    print(f"  ID: {point.id}")
                    print(f"  Payload keys: {list(point.payload.keys()) if point.payload else 'None'}")
                    if point.payload and 'content' in point.payload:
                        print(f"  Content preview: {point.payload['content'][:100]}...")
                    elif point.payload and 'content_preview' in point.payload:
                        print(f"  Content preview: {point.payload['content_preview'][:100]}...")
            else:
                print("❌ No points found in the collection - your collection might be empty!")
                return False

        except Exception as e:
            print(f"❌ Error getting sample points: {e}")
            return False

        return True

    except Exception as e:
        print(f"❌ Error connecting to Qdrant: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_qdrant_connection()
    if success:
        print("\n✅ Qdrant connection and collection check passed!")
        print("Your agent should work if the collection has data and API keys are correct.")
    else:
        print("\n❌ Qdrant connection or collection check failed!")
        print("Fix the issues above before running the agent.")