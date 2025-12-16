#!/usr/bin/env python3
"""
Retrieval Pipeline Testing - Qdrant Embeddings Validation Utility

This module implements a comprehensive testing utility to validate the Qdrant retrieval pipeline:
1. Direct retrieval of embeddings by ID
2. Similarity search with configurable parameters (top-k, thresholds, filters)
3. End-to-end pipeline validation from stored embeddings to query results
4. Error handling for edge cases (empty collections, connection failures, invalid queries)

Usage:
    from retrieve import connect_qdrant, retrieve_by_id, similarity_search

    client = connect_qdrant()
    results = retrieve_by_id(client, "textbook_embeddings", ["doc_001"])

    Or via CLI:
    python retrieve.py --collection textbook_embeddings --query "What is ROS 2?" --top-k 5

Author: AI Assistant
Date: 2025-12-12
"""

import os
import sys
import logging
import time
import functools
from typing import List, Dict, Optional, Union, Any
from dataclasses import dataclass, asdict, field
from enum import Enum
from datetime import datetime

import cohere
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, Filter, FieldCondition, MatchValue
from qdrant_client.http.exceptions import UnexpectedResponse, ResponseHandlingException
from dotenv import load_dotenv


# Load environment variables
load_dotenv()


# ============================================================================
# Logging Configuration
# ============================================================================

logging.basicConfig(
    level=os.getenv('LOG_LEVEL', 'INFO'),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('retrieval_test.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ============================================================================
# Performance Tracking
# ============================================================================

performance_metrics = {
    'function_timings': {},
    'call_counts': {},
    'total_execution_times': {}
}


def timing_decorator(func):
    """
    Decorator to measure execution time of functions and track performance metrics.

    Tracks:
        - Call count for each function
        - Individual execution times
        - Total cumulative execution time

    Returns:
        Wrapped function with timing instrumentation
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        func_name = func.__name__

        # Track call count
        if func_name not in performance_metrics['call_counts']:
            performance_metrics['call_counts'][func_name] = 0
        performance_metrics['call_counts'][func_name] += 1

        try:
            result = func(*args, **kwargs)
            return result
        finally:
            end_time = time.time()
            execution_time = end_time - start_time

            # Track timing metrics
            if func_name not in performance_metrics['function_timings']:
                performance_metrics['function_timings'][func_name] = []
            performance_metrics['function_timings'][func_name].append(execution_time)

            # Update total execution time
            if func_name not in performance_metrics['total_execution_times']:
                performance_metrics['total_execution_times'][func_name] = 0
            performance_metrics['total_execution_times'][func_name] += execution_time

            logger.debug(f"{func_name} executed in {execution_time:.4f}s")

    return wrapper


def get_performance_summary() -> Dict[str, Any]:
    """
    Get a summary of performance metrics collected during execution.

    Returns:
        Dict containing performance statistics for each tracked function:
            - call_count: Number of times function was called
            - total_time: Total cumulative execution time
            - avg_time: Average execution time per call
            - min_time: Minimum execution time
            - max_time: Maximum execution time
            - p95_time: 95th percentile execution time
    """
    summary = {}

    for func_name, timings in performance_metrics['function_timings'].items():
        if timings:
            sorted_timings = sorted(timings)
            p95_index = int(len(sorted_timings) * 0.95)

            summary[func_name] = {
                'call_count': performance_metrics['call_counts'][func_name],
                'total_time': performance_metrics['total_execution_times'][func_name],
                'avg_time': sum(timings) / len(timings),
                'min_time': min(timings),
                'max_time': max(timings),
                'p95_time': sorted_timings[p95_index] if p95_index < len(sorted_timings) else sorted_timings[-1]
            }

    return summary


# ============================================================================
# Data Models
# ============================================================================

class TestStatus(Enum):
    """Test case execution status"""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"


@dataclass
class RetrievalQuery:
    """
    Represents a query request to the retrieval system.

    Attributes:
        collection_name: Target Qdrant collection name
        query_vector: Embedding vector for similarity search (1024 dimensions for Cohere)
        top_k: Maximum number of results to return (default: 10, max: 100)
        score_threshold: Minimum similarity score filter (0.0-1.0 for cosine similarity)
        filters: Optional metadata filters for narrowing search results
        with_vectors: Whether to include vector data in results
        with_payload: Whether to include payload/metadata in results
    """
    collection_name: str
    query_vector: List[float]
    top_k: int = 10
    score_threshold: Optional[float] = None
    filters: Optional[Dict] = None
    with_vectors: bool = False
    with_payload: bool = True

    def __post_init__(self):
        """Validate query parameters"""
        if self.top_k <= 0 or self.top_k > 100:
            raise ValueError(f"top_k must be between 1 and 100, got {self.top_k}")
        if self.score_threshold is not None and (self.score_threshold < 0.0 or self.score_threshold > 1.0):
            raise ValueError(f"score_threshold must be between 0.0 and 1.0, got {self.score_threshold}")


@dataclass
class RetrievalResult:
    """
    Represents a single retrieved point from Qdrant.

    Attributes:
        id: Unique identifier of the point in Qdrant
        score: Similarity score (0.0 to 1.0 for cosine distance)
        payload: Metadata associated with the point
        vector: Optional embedding vector (included if with_vectors=True)
        retrieval_time_ms: Time taken to retrieve this result in milliseconds
    """
    id: Union[str, int]
    score: float
    payload: Dict[str, Any]
    vector: Optional[List[float]] = None
    retrieval_time_ms: float = 0.0


@dataclass
class TestCase:
    """
    Represents a single test scenario for validation.

    Attributes:
        name: Descriptive test case identifier
        query_text: Human-readable query text
        collection_name: Target collection for the test
        expected_doc_ids: Expected document IDs that should appear in top results
        expected_keywords: Keywords that should appear in result content
        relevance_threshold: Minimum similarity score for relevant results
        top_k: Number of results to retrieve
        status: Test execution status
        actual_results: Results from test execution (populated after running)
        error_message: Error message if test failed
    """
    name: str
    query_text: str
    collection_name: str
    expected_doc_ids: List[str]
    expected_keywords: List[str]
    relevance_threshold: float
    top_k: int = 10
    status: TestStatus = TestStatus.PENDING
    actual_results: Optional[List[RetrievalResult]] = None
    error_message: Optional[str] = None

    def __post_init__(self):
        """Validate test case parameters"""
        if self.relevance_threshold < 0.0 or self.relevance_threshold > 1.0:
            raise ValueError(f"relevance_threshold must be between 0.0 and 1.0, got {self.relevance_threshold}")
        if self.top_k <= 0:
            raise ValueError(f"top_k must be positive, got {self.top_k}")


@dataclass
class TestSuiteResult:
    """
    Aggregates results from multiple test cases.

    Attributes:
        total_tests: Total number of test cases executed
        passed: Number of tests that passed
        failed: Number of tests that failed
        errors: Number of tests that encountered errors
        execution_time_ms: Total execution time for all tests
        test_results: Individual test case results
        performance_metrics: Performance statistics
        timestamp: ISO format timestamp of test execution
    """
    total_tests: int
    passed: int
    failed: int
    errors: int
    execution_time_ms: float
    test_results: List[TestCase]
    performance_metrics: Dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    @property
    def pass_rate(self) -> float:
        """Calculate pass rate as percentage"""
        return (self.passed / self.total_tests * 100) if self.total_tests > 0 else 0.0

    @property
    def failure_rate(self) -> float:
        """Calculate failure rate as percentage"""
        return ((self.failed + self.errors) / self.total_tests * 100) if self.total_tests > 0 else 0.0


@dataclass
class CollectionInfo:
    """
    Represents metadata about a Qdrant collection.

    Attributes:
        name: Collection name
        vector_size: Dimensionality of vectors
        distance_metric: Distance metric used ("Cosine", "Euclidean", "Dot")
        points_count: Total number of points in collection
        indexed_vectors_count: Number of indexed vectors
        status: Collection status ("green", "yellow", "red")
    """
    name: str
    vector_size: int
    distance_metric: str
    points_count: int
    indexed_vectors_count: int
    status: str


# ============================================================================
# Core Functions
# ============================================================================

@timing_decorator
def connect_qdrant(
    url: Optional[str] = None,
    api_key: Optional[str] = None,
    timeout: float = 10.0
) -> QdrantClient:
    """
    Establish connection to Qdrant instance using configuration from environment.

    Args:
        url: Qdrant instance URL. If None, reads from QDRANT_URL environment variable
        api_key: API key for authentication. If None, reads from QDRANT_API_KEY environment variable
        timeout: Connection timeout in seconds. Default: 10.0

    Returns:
        QdrantClient: Connected Qdrant client instance

    Raises:
        ConnectionError: If connection to Qdrant fails
        ValueError: If URL or API key is missing and not in environment

    Example:
        >>> client = connect_qdrant()
        >>> # Or with explicit credentials
        >>> client = connect_qdrant(url="https://qdrant.example.com:6333", api_key="key")
    """
    # Get configuration from environment if not provided
    qdrant_url = url or os.getenv('QDRANT_URL')
    qdrant_api_key = api_key or os.getenv('QDRANT_API_KEY')

    if not qdrant_url:
        raise ValueError("Qdrant URL not provided. Set QDRANT_URL environment variable or pass url parameter.")
    if not qdrant_api_key:
        raise ValueError("Qdrant API key not provided. Set QDRANT_API_KEY environment variable or pass api_key parameter.")

    logger.info(f"Connecting to Qdrant at {qdrant_url}")

    try:
        client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key,
            timeout=timeout
        )

        # Test connection by getting collections
        client.get_collections()

        logger.info(f"[OK] Successfully connected to Qdrant at {qdrant_url}")
        return client

    except (UnexpectedResponse, ResponseHandlingException) as e:
        error_msg = f"Failed to connect to Qdrant at {qdrant_url}: {e}"
        logger.error(error_msg)
        raise ConnectionError(error_msg) from e
    except Exception as e:
        error_msg = f"Unexpected error connecting to Qdrant: {e}"
        logger.error(error_msg)
        raise ConnectionError(error_msg) from e


@timing_decorator
def validate_collection(
    client: QdrantClient,
    collection_name: str
) -> Dict[str, Any]:
    """
    Validate that a collection exists and return its metadata.

    Args:
        client: Connected Qdrant client
        collection_name: Name of collection to validate

    Returns:
        Dict containing collection metadata:
            - name: Collection name
            - vector_size: Dimensionality of vectors
            - distance_metric: Distance metric used
            - points_count: Total number of points
            - indexed_vectors_count: Number of indexed vectors
            - status: Collection status
            - exists: Whether collection exists

    Raises:
        ValueError: If collection doesn't exist
        ConnectionError: If Qdrant connection fails

    Example:
        >>> info = validate_collection(client, "textbook_embeddings")
        >>> print(f"Collection has {info['points_count']} embeddings")
    """
    logger.info(f"Validating collection: {collection_name}")

    try:
        # Get collection info
        collection_info = client.get_collection(collection_name)

        # Extract metadata
        metadata = {
            'name': collection_name,
            'vector_size': collection_info.config.params.vectors.size,
            'distance_metric': collection_info.config.params.vectors.distance.name,
            'points_count': collection_info.points_count,
            'indexed_vectors_count': collection_info.indexed_vectors_count or collection_info.points_count,
            'status': collection_info.status.name,
            'exists': True
        }

        logger.info(f"[OK] Collection '{collection_name}' validated: {metadata['points_count']} points, "
                   f"{metadata['vector_size']} dimensions, {metadata['distance_metric']} distance")

        return metadata

    except Exception as e:
        # List available collections for helpful error message
        try:
            collections = client.get_collections()
            available = [col.name for col in collections.collections]
            error_msg = (f"Collection '{collection_name}' does not exist. "
                        f"Available collections: {available}")
        except:
            error_msg = f"Collection '{collection_name}' does not exist and unable to list available collections."

        logger.error(error_msg)
        raise ValueError(error_msg) from e


@timing_decorator
def retrieve_by_id(
    client: QdrantClient,
    collection_name: str,
    point_ids: List[Union[str, int]],
    with_vectors: bool = False,
    with_payload: bool = True
) -> List[RetrievalResult]:
    """
    Retrieve specific embeddings by their IDs.

    Args:
        client: Connected Qdrant client
        collection_name: Name of the target collection
        point_ids: List of point IDs to retrieve
        with_vectors: Include vector data in results. Default: False
        with_payload: Include payload/metadata in results. Default: True

    Returns:
        List[RetrievalResult]: Retrieved points with metadata

    Raises:
        ValueError: If collection doesn't exist or point_ids is empty
        ConnectionError: If Qdrant connection fails during retrieval

    Performance:
        Should complete in <100ms for collections up to 10K vectors (SC-001)

    Example:
        >>> results = retrieve_by_id(
        ...     client=client,
        ...     collection_name="textbook_embeddings",
        ...     point_ids=["doc_001", "doc_002"],
        ...     with_vectors=True
        ... )
        >>> for result in results:
        ...     print(f"ID: {result.id}, Content: {result.payload['content'][:100]}")
    """
    # Input validation
    if not point_ids:
        raise ValueError("point_ids cannot be empty")

    logger.info(f"Retrieving {len(point_ids)} points from collection '{collection_name}'")

    start_time = time.time()

    try:
        # Retrieve points from Qdrant
        records = client.retrieve(
            collection_name=collection_name,
            ids=point_ids,
            with_vectors=with_vectors,
            with_payload=with_payload
        )

        # Convert to RetrievalResult objects
        results = []
        for record in records:
            retrieval_time_ms = (time.time() - start_time) * 1000

            result = RetrievalResult(
                id=record.id,
                score=1.0,  # No score for direct retrieval
                payload=record.payload if record.payload else {},
                vector=record.vector if with_vectors and hasattr(record, 'vector') else None,
                retrieval_time_ms=retrieval_time_ms
            )
            results.append(result)

        elapsed_ms = (time.time() - start_time) * 1000
        logger.info(f"[OK] Retrieved {len(results)} points in {elapsed_ms:.2f}ms")

        return results

    except Exception as e:
        error_msg = f"Failed to retrieve points from collection '{collection_name}': {e}"
        logger.error(error_msg)
        raise ConnectionError(error_msg) from e


@timing_decorator
def similarity_search(
    client: QdrantClient,
    collection_name: str,
    query_vector: List[float],
    top_k: int = 10,
    score_threshold: Optional[float] = None,
    filters: Optional[Dict] = None,
    with_vectors: bool = False,
    with_payload: bool = True
) -> List[RetrievalResult]:
    """
    Perform semantic similarity search with query vector.

    Args:
        client: Connected Qdrant client
        collection_name: Name of the target collection
        query_vector: Query embedding vector (must match collection dimensions)
        top_k: Maximum number of results to return. Default: 10, Max: 100
        score_threshold: Minimum similarity score filter (0.0-1.0). Default: None
        filters: Metadata filters for search. Default: None
        with_vectors: Include vector data in results. Default: False
        with_payload: Include payload/metadata in results. Default: True

    Returns:
        List[RetrievalResult]: Top-k most similar points ranked by score

    Raises:
        ValueError: If query_vector dimensions don't match collection, or invalid parameters
        ConnectionError: If Qdrant connection fails during search
    """
    # Input validation
    if not isinstance(collection_name, str) or not collection_name.strip():
        raise ValueError(f"collection_name must be a non-empty string, got {collection_name}")

    if not isinstance(query_vector, list) or not query_vector:
        raise ValueError(f"query_vector must be a non-empty list, got {type(query_vector)}")

    if not isinstance(top_k, int) or top_k <= 0 or top_k > 100:
        raise ValueError(f"top_k must be an integer between 1 and 100, got {top_k}")

    if score_threshold is not None and (not isinstance(score_threshold, (int, float)) or
                                      score_threshold < 0.0 or score_threshold > 1.0):
        raise ValueError(f"score_threshold must be between 0.0 and 1.0, got {score_threshold}")

    if filters is not None and not isinstance(filters, dict):
        raise ValueError(f"filters must be a dictionary or None, got {type(filters)}")

    logger.info(f"Performing similarity search in collection '{collection_name}' with top_k={top_k}")
    if score_threshold is not None:
        logger.info(f"Using score threshold: {score_threshold}")

    start_time = time.time()

    try:
        # Validate collection exists and get its vector size
        collection_info = client.get_collection(collection_name)
        expected_vector_size = collection_info.config.params.vectors.size

        if len(query_vector) != expected_vector_size:
            raise ValueError(
                f"Query vector dimension ({len(query_vector)}) doesn't match "
                f"collection dimension ({expected_vector_size})"
            )

        # Prepare filters if provided
        qdrant_filters = None
        if filters:
            from qdrant_client.http.models import Filter
            conditions = []
            for key, value in filters.items():
                conditions.append(FieldCondition(key=key, match=MatchValue(value=value)))
            qdrant_filters = Filter(must=conditions)

        # Perform the search
        search_results = client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=top_k,
            score_threshold=score_threshold,
            query_filter=qdrant_filters,
            with_vectors=with_vectors,
            with_payload=with_payload
        )

        # Convert results to RetrievalResult objects
        results = []
        for result in search_results:
            retrieval_time_ms = (time.time() - start_time) * 1000

            retrieval_result = RetrievalResult(
                id=result.id,
                score=result.score,
                payload=result.payload if result.payload else {},
                vector=result.vector if with_vectors and hasattr(result, 'vector') else None,
                retrieval_time_ms=retrieval_time_ms
            )
            results.append(retrieval_result)

        elapsed_ms = (time.time() - start_time) * 1000
        logger.info(f"[OK] Similarity search completed in {elapsed_ms:.2f}ms, returned {len(results)} results")

        return results

    except Exception as e:
        error_msg = f"Failed to perform similarity search in collection '{collection_name}': {e}"
        logger.error(error_msg)
        raise ConnectionError(error_msg) from e


@timing_decorator
def generate_query_embedding(
    text: str,
    cohere_api_key: Optional[str] = None,
    model: str = "embed-multilingual-v3.0"
) -> List[float]:
    """
    Generate embedding vector for query text using Cohere API.

    Args:
        text: Query text to embed
        cohere_api_key: Cohere API key. If None, reads from COHERE_API_KEY environment variable
        model: Cohere embedding model. Default: "embed-multilingual-v3.0"

    Returns:
        List[float]: Embedding vector (1024 dimensions for embed-multilingual-v3.0)

    Raises:
        ValueError: If text is empty or API key is missing
        RuntimeError: If Cohere API call fails
    """
    # Input validation
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text cannot be empty")

    if len(text) > 40000:  # Cohere API limit
        raise ValueError(f"text is too long ({len(text)} chars), maximum is 40000 chars")

    # Get API key
    api_key = cohere_api_key or os.getenv('COHERE_API_KEY')
    if not api_key:
        raise ValueError("Cohere API key not provided and not found in environment variables")

    if not isinstance(model, str) or not model.strip():
        raise ValueError(f"model must be a non-empty string, got {model}")

    logger.info(f"Generating query embedding for text (first 100 chars): '{text[:100]}...'")

    try:
        # Initialize Cohere client
        co = cohere.Client(api_key)

        # Generate embedding using search_query input type for queries
        response = co.embed(
            texts=[text],
            model=model,
            input_type="search_query"  # Use search_query for query embeddings (different from document embeddings)
        )

        embedding = response.embeddings[0]

        logger.info(f"Successfully generated {len(embedding)}-dimensional embedding")
        return embedding

    except Exception as e:
        error_msg = f"Failed to generate query embedding: {e}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    # Basic CLI interface for testing
    import argparse

    parser = argparse.ArgumentParser(description="Qdrant Retrieval Pipeline Testing Utility")
    parser.add_argument("--collection", help="Collection name to test")
    parser.add_argument("--validate", action="store_true", help="Validate collection only")
    parser.add_argument("--ids", nargs="+", help="Point IDs to retrieve")

    args = parser.parse_args()

    try:
        # Connect to Qdrant
        client = connect_qdrant()

        if args.validate and args.collection:
            # Validate collection
            info = validate_collection(client, args.collection)
            print(f"\n[OK] Collection '{info['name']}' validated successfully")
            print(f"  Points: {info['points_count']}")
            print(f"  Dimensions: {info['vector_size']}")
            print(f"  Distance: {info['distance_metric']}")
            print(f"  Status: {info['status']}")

        elif args.collection and args.ids:
            # Retrieve points by ID
            results = retrieve_by_id(client, args.collection, args.ids, with_payload=True)
            print(f"\n[OK] Retrieved {len(results)} points:")
            for result in results:
                print(f"\n  ID: {result.id}")
                print(f"  Retrieval Time: {result.retrieval_time_ms:.2f}ms")
                if result.payload:
                    print(f"  Content: {result.payload.get('content', 'N/A')[:100]}...")

        else:
            print("Usage: python retrieve.py --collection <name> --validate")
            print("       python retrieve.py --collection <name> --ids <id1> <id2> ...")

    except Exception as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)
