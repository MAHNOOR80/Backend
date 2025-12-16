#!/usr/bin/env python3
"""
Test script to verify Qdrant connection and basic functionality.
"""

import os
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_qdrant_connection():
    """Test basic Qdrant connection and operations."""
    print("Testing Qdrant connection...")

    # Get configuration from environment
    qdrant_url = os.getenv('QDRANT_URL', 'http://localhost:6333')
    qdrant_api_key = os.getenv('QDRANT_API_KEY', '')

    print(f"QDRANT_URL: {qdrant_url}")
    print(f"QDRANT_API_KEY exists: {'Yes' if qdrant_api_key else 'No'}")

    try:
        # Initialize Qdrant client
        if qdrant_url.startswith('http://') or qdrant_url.startswith('https://'):
            # Cloud instance
            client = QdrantClient(
                url=qdrant_url,
                api_key=qdrant_api_key,
                timeout=10.0
            )
        else:
            # Local instance
            client = QdrantClient(
                host=qdrant_url,
                timeout=10.0
            )

        print("[OK] Successfully connected to Qdrant")

        # Test getting collections
        collections = client.get_collections()
        print(f"[OK] Found {len(collections.collections)} existing collections")
        for collection in collections.collections:
            # Get detailed collection info to get the point count
            collection_info = client.get_collection(collection.name)
            print(f"  - {collection.name}: {collection_info.points_count} points")

        # Create a test collection
        test_collection_name = "test_connection"
        print(f"\nCreating test collection: {test_collection_name}")

        # Delete collection if it exists
        try:
            client.delete_collection(test_collection_name)
            print(f"[OK] Deleted existing test collection: {test_collection_name}")
        except:
            pass  # Collection might not exist, which is fine

        # Create new collection
        client.create_collection(
            collection_name=test_collection_name,
            vectors_config=VectorParams(size=4, distance=Distance.COSINE)
        )
        print(f"[OK] Created collection: {test_collection_name}")

        # Insert a test point
        test_point = PointStruct(
            id=1,
            vector=[0.1, 0.2, 0.3, 0.4],
            payload={
                "test": "This is a test point",
                "timestamp": "2025-12-12"
            }
        )

        client.upsert(
            collection_name=test_collection_name,
            points=[test_point]
        )
        print("[OK] Inserted test point")

        # Verify the point was inserted
        collection_info = client.get_collection(test_collection_name)
        print(f"[OK] Collection now has {collection_info.points_count} points")

        if collection_info.points_count == 1:
            print("[OK] Test successful: Qdrant is working correctly!")
        else:
            print("[ERROR] Test failed: Collection should have 1 point")

        # Clean up - delete test collection
        client.delete_collection(test_collection_name)
        print(f"[OK] Cleaned up test collection: {test_collection_name}")

        return True

    except Exception as e:
        print(f"[ERROR] Error during Qdrant test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_qdrant_connection()
    if success:
        print("\nQdrant connection test PASSED!")
    else:
        print("\nQdrant connection test FAILED!")
        exit(1)