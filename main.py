#!/usr/bin/env python3
"""
Embedding Pipeline - Docusaurus Content Extraction, Embedding Generation, and Vector Storage

This script implements a complete pipeline that:
1. Crawls Docusaurus URLs to extract text content
2. Generates embeddings using Cohere API
3. Stores embeddings in Qdrant vector database

Usage:
    python main.py --url <docusaurus_url> --collection-name <collection_name>

Author: AI Assistant
Date: 2025-12-10
"""

import os
import sys
import logging
import argparse
from typing import List, Dict, Optional, Any
import time
import requests
import hashlib
import uuid
from urllib.parse import urljoin, urlparse
import re
import asyncio
import aiohttp
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Optional
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from dotenv import load_dotenv
import functools


# Load environment variables from .env file
load_dotenv()


# Performance metrics tracking
performance_metrics = {
    'function_timings': {},
    'call_counts': {},
    'total_execution_times': {}
}


def timing_decorator(func):
    """
    Decorator to measure execution time of functions and track performance metrics.
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

            # Update total execution time for the function
            if func_name not in performance_metrics['total_execution_times']:
                performance_metrics['total_execution_times'][func_name] = 0
            performance_metrics['total_execution_times'][func_name] += execution_time

            logger.debug(f"{func_name} executed in {execution_time:.4f}s")

    return wrapper


def get_performance_summary() -> Dict:
    """
    Get a summary of performance metrics collected during execution.
    Returns:
        Dict: Performance metrics summary
    """
    summary = {}

    for func_name, timings in performance_metrics['function_timings'].items():
        if timings:
            summary[func_name] = {
                'call_count': performance_metrics['call_counts'][func_name],
                'total_time': performance_metrics['total_execution_times'][func_name],
                'avg_time': sum(timings) / len(timings),
                'min_time': min(timings),
                'max_time': max(timings),
                'total_calls': len(timings)
            }

    return summary


def cleanup_resources():
    """
    Perform cleanup of resources before application shutdown.
    Currently handles closing Qdrant client connection.
    """
    global qdrant_client

    if qdrant_client is not None:
        try:
            # Qdrant client doesn't have a close method, but we can log the cleanup
            logger.info("Cleaning up Qdrant client resources")
        except Exception as e:
            logger.error(f"Error during Qdrant client cleanup: {e}")

    logger.info("Resource cleanup completed")


def graceful_shutdown_handler(signum, frame):
    """
    Handle graceful shutdown when receiving termination signals.
    Args:
        signum: Signal number
        frame: Current stack frame
    """
    logger.info(f"Received signal {signum}, initiating graceful shutdown...")
    cleanup_resources()
    sys.exit(0)


# Set up signal handlers for graceful shutdown
try:
    import signal
    signal.signal(signal.SIGINT, graceful_shutdown_handler)   # Ctrl+C
    signal.signal(signal.SIGTERM, graceful_shutdown_handler)  # Termination signal
except Exception as e:
    logger.warning(f"Could not set up signal handlers: {e}")


@dataclass
class DocumentContent:
    """
    Represents the text extracted from Docusaurus pages, including the raw text and source URL metadata.
    """
    id: str
    url: str
    content: str
    title: str
    created_at: datetime
    chunk_index: int = 0

    def validate(self) -> List[str]:
        """Validate the document content."""
        errors = []
        if not self.url:
            errors.append("URL must be provided")
        if not self.content.strip():
            errors.append("Content must not be empty")
        if self.chunk_index < 0:
            errors.append("Chunk index must be non-negative")
        return errors


@dataclass
class EmbeddingVector:
    """
    Represents the numerical vector representation of text content generated by the embedding service.
    """
    id: str
    vector: List[float]
    document_id: str
    created_at: datetime

    def validate(self) -> List[str]:
        """Validate the embedding vector."""
        errors = []
        if not self.id:
            errors.append("ID must be provided")
        if not self.document_id:
            errors.append("Document ID must be provided")
        if not self.vector:
            errors.append("Vector must not be empty")
        if any(not isinstance(v, (int, float)) or not v for v in self.vector):
            errors.append("Vector must contain valid numbers")
        return errors


@dataclass
class VectorDatabaseCollection:
    """
    Represents the storage unit in the vector database where embeddings are stored with associated metadata for retrieval.
    """
    name: str
    size: int = 0
    dimensions: Optional[int] = None
    created_at: Optional[datetime] = None

    def validate(self) -> List[str]:
        """Validate the collection."""
        errors = []
        if not self.name:
            errors.append("Name must be provided")
        if self.size < 0:
            errors.append("Size must be non-negative")
        if self.dimensions is not None and self.dimensions <= 0:
            errors.append("Dimensions must be positive if provided")
        return errors


@dataclass
class ProcessingPipeline:
    """
    Represents the workflow that orchestrates crawling, text extraction, embedding generation, and storage operations.
    """
    status: str
    source_url: str
    total_documents: int = 0
    processed_documents: int = 0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    errors: List[str] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []

    def validate(self) -> List[str]:
        """Validate the processing pipeline."""
        errors = []
        if self.status not in ["running", "completed", "failed", "pending"]:
            errors.append("Status must be one of: running, completed, failed, pending")
        if not self.source_url:
            errors.append("Source URL must be provided")
        if self.processed_documents > self.total_documents:
            errors.append("Processed documents cannot exceed total documents")
        return errors


# Configuration class to hold all settings
class Config:
    """Configuration class to manage application settings from environment variables."""

    def __init__(self):
        self.cohere_api_key = os.getenv('COHERE_API_KEY', '')
        self.qdrant_url = os.getenv('QDRANT_URL', 'http://localhost:6333')
        self.qdrant_api_key = os.getenv('QDRANT_API_KEY', '')
        self.chunk_size = int(os.getenv('CHUNK_SIZE', '1000'))
        self.chunk_overlap = int(os.getenv('CHUNK_OVERLAP', '100'))
        self.batch_size = int(os.getenv('BATCH_SIZE', '10'))
        self.request_delay = float(os.getenv('REQUEST_DELAY', '0.5'))
        self.max_depth = int(os.getenv('MAX_DEPTH', '2'))
        self.log_level = os.getenv('LOG_LEVEL', 'INFO')
        self.log_file = os.getenv('LOG_FILE', 'embedding_pipeline.log')
        self.target_site = os.getenv('TARGET_SITE', 'https://physical-ai-humanoid-robotics-textb-dun.vercel.app/')

    def validate(self):
        """Validate configuration settings."""
        errors = []

        if not self.cohere_api_key:
            errors.append("COHERE_API_KEY is required")

        if not self.qdrant_url:
            errors.append("QDRANT_URL is required")

        if self.chunk_size <= 0:
            errors.append("CHUNK_SIZE must be positive")

        if self.chunk_size > 5000:  # Reasonable upper limit
            errors.append("CHUNK_SIZE should be less than 5000 for optimal performance")

        if self.chunk_overlap < 0:
            errors.append("CHUNK_OVERLAP cannot be negative")

        if self.chunk_overlap >= self.chunk_size:
            errors.append("CHUNK_OVERLAP must be less than CHUNK_SIZE")

        if self.batch_size <= 0:
            errors.append("BATCH_SIZE must be positive")

        if self.batch_size > 100:  # Reasonable upper limit for batch operations
            errors.append("BATCH_SIZE should be 100 or less to avoid memory issues")

        if self.request_delay < 0:
            errors.append("REQUEST_DELAY cannot be negative")

        if self.max_depth <= 0:
            errors.append("MAX_DEPTH must be positive")

        if self.max_depth > 10:  # Prevent excessive crawling depth
            errors.append("MAX_DEPTH should be 10 or less to prevent excessive crawling")

        # Validate log level
        valid_log_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if self.log_level.upper() not in valid_log_levels:
            errors.append(f"LOG_LEVEL must be one of: {', '.join(valid_log_levels)}")

        # Validate URL format
        try:
            from urllib.parse import urlparse
            result = urlparse(self.qdrant_url)
            if not all([result.scheme, result.netloc]):
                errors.append("QDRANT_URL must be a valid URL")
        except Exception:
            errors.append("QDRANT_URL must be a valid URL")

        return errors


def setup_logging():
    """Set up logging configuration based on environment variables."""
    # Get config instance to access log settings
    config = Config()
    log_level = config.log_level.upper()
    log_file = config.log_file

    # Convert string log level to logging constant
    numeric_level = getattr(logging, log_level, logging.INFO)

    # Create custom formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )

    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(numeric_level)
    file_handler.setFormatter(formatter)

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)

    # Get root logger and configure it
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Clear any existing handlers
    root_logger.handlers = []

    # Add handlers
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    return root_logger


# Set up logging
logger = setup_logging()

def get_qdrant_client():
    """Set up Qdrant client connection with error handling."""
    try:
        # Determine if we're using local or cloud Qdrant
        if config.qdrant_url.startswith('http://') or config.qdrant_url.startswith('https://'):
            # Cloud instance
            client = QdrantClient(
                url=config.qdrant_url,
                api_key=config.qdrant_api_key,
                timeout=10.0
            )
        else:
            # Local instance
            client = QdrantClient(
                host=config.qdrant_url,
                timeout=10.0
            )

        # Test the connection
        client.get_collections()
        logger.info("Successfully connected to Qdrant")
        return client
    except Exception as e:
        logger.error(f"Failed to connect to Qdrant: {e}")
        raise


@timing_decorator
def create_collection(collection_name: str = "rag_embedding", vector_size: int = 1024, distance: str = "Cosine") -> bool:
    """
    Create a collection in Qdrant for storing embeddings.
    Args:
        collection_name (str): Name of the collection to create (default: "rag_embedding")
        vector_size (int): Size of the embedding vectors (default: 1024)
        distance (str): Distance metric for similarity search (default: "Cosine")

    Returns:
        bool: True if collection was created or already exists, False otherwise

    Raises:
        ValueError: If parameters are invalid
    """
    # Input validation
    if not isinstance(collection_name, str):
        raise ValueError(f"collection_name must be a string, got {type(collection_name)}")

    if not collection_name.strip():
        raise ValueError("collection_name cannot be empty")

    if not isinstance(vector_size, int):
        raise ValueError(f"vector_size must be an integer, got {type(vector_size)}")

    if vector_size <= 0:
        raise ValueError(f"vector_size must be positive, got {vector_size}")

    if vector_size > 65536:  # Qdrant has vector size limits
        raise ValueError(f"vector_size ({vector_size}) exceeds Qdrant maximum of 65536")

    if not isinstance(distance, str):
        raise ValueError(f"distance must be a string, got {type(distance)}")

    if not distance.strip():
        raise ValueError("distance cannot be empty")

    from qdrant_client.http.models import Distance, VectorParams

    # Map distance string to Qdrant Distance enum
    distance_map = {
        "Cosine": Distance.COSINE,
        "Euclidean": Distance.EUCLID,
        "Dot": Distance.DOT
    }

    if distance not in distance_map:
        logger.warning(f"Invalid distance metric '{distance}', using 'Cosine' as default")
        distance = "Cosine"

    try:
        # Check if collection already exists
        collections = qdrant_client.get_collections()
        existing_collection_names = [collection.name for collection in collections.collections]

        if collection_name in existing_collection_names:
            logger.info(f"Collection '{collection_name}' already exists")
            return True

        # Create the collection
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=distance_map[distance])
        )

        logger.info(f"Successfully created collection '{collection_name}' with vector size {vector_size}")
        return True

    except Exception as e:
        logger.error(f"Failed to create collection '{collection_name}': {e}")
        return False


@timing_decorator
def save_chunk_to_qdrant(embedding_vector: List[float], document_content: DocumentContent, collection_name: str = "rag_embedding") -> bool:
    """
    Save an embedding vector with its metadata to Qdrant collection.
    Args:
        embedding_vector (List[float]): The embedding vector to store
        document_content (DocumentContent): The source document with metadata
        collection_name (str): Name of the collection to store in (default: "rag_embedding")

    Returns:
        bool: True if successfully saved, False otherwise

    Raises:
        ValueError: If parameters are invalid
    """
    # Input validation
    if not isinstance(embedding_vector, list):
        raise ValueError(f"embedding_vector must be a list, got {type(embedding_vector)}")

    if not embedding_vector:
        raise ValueError("embedding_vector cannot be empty")

    # Validate that all elements in the embedding vector are numbers
    for i, val in enumerate(embedding_vector):
        if not isinstance(val, (int, float)):
            raise ValueError(f"embedding_vector[{i}] must be a number, got {type(val)}: {val}")

    if not isinstance(document_content, DocumentContent):
        raise ValueError(f"document_content must be a DocumentContent instance, got {type(document_content)}")

    if not isinstance(collection_name, str):
        raise ValueError(f"collection_name must be a string, got {type(collection_name)}")

    if not collection_name.strip():
        raise ValueError("collection_name cannot be empty")

    from qdrant_client.http.models import PointStruct

    try:
        # Create a point with the embedding and metadata
        point = PointStruct(
            id=document_content.id,
            vector=embedding_vector,
            payload={
                "url": document_content.url,
                "title": document_content.title,
                "content": document_content.content,  # Store full content
                "content_preview": document_content.content[:500],  # Store first 500 chars as preview for backward compatibility
                "created_at": document_content.created_at.isoformat(),
                "chunk_index": getattr(document_content, 'chunk_index', 0)  # In case chunk_index is not defined
            }
        )

        # Upsert the point into the collection
        result = qdrant_client.upsert(
            collection_name=collection_name,
            points=[point]
        )

        # Verify the upsert was successful by checking collection size
        try:
            collection_info = qdrant_client.get_collection(collection_name)
            logger.info(f"Successfully saved embedding to collection '{collection_name}' with ID: {document_content.id}. Collection now has {collection_info.points_count} points.")
            return True
        except Exception as verification_error:
            logger.warning(f"Embedding upsert may have succeeded but verification failed: {verification_error}")
            return True  # Still return True as the upsert operation itself didn't fail

    except Exception as e:
        logger.error(f"Failed to save embedding to Qdrant: {e}")
        return False


@timing_decorator
def detect_duplicate_content(content: str, collection_name: str = "rag_embedding", threshold: float = 0.95) -> bool:
    """
    Detect if similar content already exists in the Qdrant collection to avoid redundant embeddings.
    Args:
        content (str): The content to check for duplicates
        collection_name (str): Name of the collection to check in (default: "rag_embedding")
        threshold (float): Similarity threshold above which content is considered duplicate (default: 0.95)

    Returns:
        bool: True if duplicate content is found, False otherwise
    """
    if not content or len(content.strip()) == 0:
        logger.warning("Content is empty, cannot detect duplicates")
        return False

    try:
        # Generate embedding for the content to check
        content_embedding = embed([content])

        if not content_embedding or len(content_embedding) == 0:
            logger.error("Could not generate embedding for duplicate detection")
            return False

        # Search for similar content in the collection
        search_results = qdrant_client.search(
            collection_name=collection_name,
            query_vector=content_embedding[0],
            limit=5,  # Check top 5 most similar results
            score_threshold=threshold  # Only return results above threshold
        )

        # If any results are found above the threshold, consider it a duplicate
        if search_results:
            logger.info(f"Found {len(search_results)} similar content items above threshold {threshold}")
            return True
        else:
            logger.debug("No similar content found, not a duplicate")
            return False

    except Exception as e:
        logger.warning(f"Error during duplicate detection: {e}, proceeding without duplicate check")
        # If there's an error, assume it's not a duplicate to avoid blocking the process
        return False


def get_content_hash(content: str) -> str:
    """
    Generate a hash for content to enable quick duplicate detection.
    Args:
        content (str): The content to hash

    Returns:
        str: SHA256 hash of the content
    """
    import hashlib

    if not content:
        return ""

    # Create a hash of the content
    content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()
    return content_hash


def check_content_exists_by_hash(content_hash: str, collection_name: str = "rag_embedding") -> bool:
    """
    Check if content with a specific hash already exists in the collection.

    Args:
        content_hash (str): The hash of the content to check
        collection_name (str): Name of the collection to check in (default: "rag_embedding")

    Returns:
        bool: True if content with this hash exists, False otherwise
    """
    try:
        # Search for points that have this content hash in their payload
        # Note: This assumes we store the content hash in the payload
        search_results = qdrant_client.scroll(
            collection_name=collection_name,
            scroll_filter=None,  # We would need to implement a filter for hash if we store it
            limit=1
        )

        # Since we don't currently store content hash in the payload,
        # we'll implement a simpler version that just returns False
        # The main duplicate detection happens via vector similarity
        logger.debug("Content hash-based duplicate detection not fully implemented (would require storing hashes in payload)")
        return False

    except Exception as e:
        logger.error(f"Error during hash-based duplicate detection: {e}")
        return False
def test_vector_storage(collection_name: str = "test_rag_embedding") -> Dict:
    """
    Test vector storage functionality with sample embeddings.

    Args:
        collection_name (str): Name of the test collection to use (default: "test_rag_embedding")

    Returns:
        Dict: Test results with information about success/failure
    """
    results = {
        "test_passed": False,
        "collection_created": False,
        "embeddings_stored": 0,
        "embeddings_retrieved": 0,
        "errors": [],
        "test_summary": ""
    }

    try:
        # Step 1: Create test collection
        logger.info(f"Creating test collection: {collection_name}")
        collection_created = create_collection(collection_name=collection_name, vector_size=1024)
        results["collection_created"] = collection_created

        if not collection_created:
            results["errors"].append("Failed to create test collection")
            return results

        # Step 2: Create sample content to store
        sample_contents = [
            DocumentContent(
                id="test_id_1",
                url="https://example.com/test1",
                content="This is a sample document content for testing vector storage functionality.",
                title="Test Document 1",
                created_at=datetime.now()
            ),
            DocumentContent(
                id="test_id_2",
                url="https://example.com/test2",
                content="Another sample document to test the embedding storage capabilities.",
                title="Test Document 2",
                created_at=datetime.now()
            )
        ]

        # Step 3: Generate embeddings for sample content
        logger.info("Generating embeddings for test content")
        sample_texts = [doc.content for doc in sample_contents]
        embeddings = embed(sample_texts)
        results["embeddings_stored"] = len(embeddings)

        if len(embeddings) != len(sample_contents):
            results["errors"].append(f"Mismatch: generated {len(embeddings)} embeddings for {len(sample_contents)} documents")
            return results

        # Step 4: Store embeddings in Qdrant
        logger.info("Storing embeddings in Qdrant")
        stored_count = 0
        for i, (doc, embedding) in enumerate(zip(sample_contents, embeddings)):
            success = save_chunk_to_qdrant(embedding, doc, collection_name)
            if success:
                stored_count += 1
            else:
                results["errors"].append(f"Failed to store embedding for document {i}")

        # Step 5: Verify that embeddings were stored by counting collection size
        try:
            collection_info = qdrant_client.get_collection(collection_name)
            results["embeddings_retrieved"] = collection_info.points_count
            logger.info(f"Collection contains {collection_info.points_count} points")
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            results["errors"].append(f"Error retrieving collection info: {e}")

        # Step 6: Test duplicate detection
        logger.info("Testing duplicate detection")
        is_duplicate = detect_duplicate_content(sample_contents[0].content, collection_name)
        logger.info(f"Duplicate detection test result: {is_duplicate}")

        # Step 7: Overall test result
        if stored_count == len(sample_contents) and results["embeddings_retrieved"] > 0:
            results["test_passed"] = True
            results["test_summary"] = f"Successfully stored and verified {stored_count} embeddings in collection {collection_name}"
            logger.info(results["test_summary"])
        else:
            results["test_summary"] = f"Test failed: stored {stored_count} out of {len(sample_contents)}, retrieved {results['embeddings_retrieved']}"
            logger.error(results["test_summary"])

    except Exception as e:
        logger.error(f"Error during vector storage test: {e}")
        results["errors"].append(f"Error during vector storage test: {str(e)}")

    return results


def verify_similarity_search(query_text: str, collection_name: str = "rag_embedding", top_k: int = 5) -> Dict:
    """
    Verify that embeddings can be retrieved via similarity search.
    Args:
        query_text (str): The query text to search for similar embeddings
        collection_name (str): Name of the collection to search in (default: "rag_embedding")
        top_k (int): Number of top results to return (default: 5)

    Returns:
        Dict: Search results with information about retrieved embeddings
    """
    results = {
        "query_text": query_text,
        "collection_name": collection_name,
        "top_k": top_k,
        "search_successful": False,
        "retrieved_results": [],
        "total_results": 0,
        "errors": [],
        "search_summary": ""
    }

    try:
        if not query_text or len(query_text.strip()) == 0:
            results["errors"].append("Query text cannot be empty")
            return results

        # Generate embedding for the query text
        logger.info(f"Generating embedding for query: '{query_text[:50]}...'")
        query_embedding = embed([query_text])

        if not query_embedding or len(query_embedding) == 0:
            results["errors"].append("Could not generate embedding for query text")
            return results

        # Perform similarity search in Qdrant
        logger.info(f"Performing similarity search in collection '{collection_name}'")
        search_results = qdrant_client.search(
            collection_name=collection_name,
            query_vector=query_embedding[0],
            limit=top_k
        )

        # Process search results
        results["total_results"] = len(search_results)
        results["search_successful"] = True

        for i, result in enumerate(search_results):
            result_info = {
                "rank": i + 1,
                "id": result.id,
                "score": result.score,
                "url": result.payload.get("url", "N/A"),
                "title": result.payload.get("title", "N/A"),
                "content_preview": result.payload.get("content_preview", "N/A")[:200],
                "created_at": result.payload.get("created_at", "N/A")
            }
            results["retrieved_results"].append(result_info)

        results["search_summary"] = f"Successfully retrieved {len(search_results)} similar documents from '{collection_name}'"
        logger.info(results["search_summary"])

        # Log the top result if available
        if search_results:
            top_result = search_results[0]
            logger.info(f"Top result - ID: {top_result.id}, Score: {top_result.score:.4f}, URL: {top_result.payload.get('url', 'N/A')}")

    except Exception as e:
        logger.error(f"Error during similarity search: {e}")
        results["errors"].append(f"Error during similarity search: {str(e)}")
        results["search_successful"] = False

    return results
def search_similar_content_by_text(query_text: str, collection_name: str = "rag_embedding", threshold: float = 0.5, top_k: int = 5) -> List[Dict]:
    """
    Search for similar content in the collection based on input text.
    Args:
        query_text (str): The text to find similar content for
        collection_name (str): Name of the collection to search in
        threshold (float): Minimum similarity score threshold
        top_k (int): Number of top results to return

    Returns:
        List[Dict]: List of similar content results with metadata
    """
    try:
        # Use the verify_similarity_search function to get results
        search_results = verify_similarity_search(query_text, collection_name, top_k)

        # Filter results by threshold if specified
        if threshold > 0:
            filtered_results = [r for r in search_results["retrieved_results"] if r["score"] >= threshold]
            search_results["retrieved_results"] = filtered_results
            search_results["total_results"] = len(filtered_results)

        return search_results["retrieved_results"]

    except Exception as e:
        logger.error(f"Error in search_similar_content_by_text: {e}")
        return []


def handle_qdrant_connection_error(error: Exception, context: str = "") -> Dict:
    """
    Handle Qdrant connection errors with detailed logging and recovery suggestions.
    Args:
        error (Exception): The error that occurred
        context (str): Context where the error occurred

    Returns:
        Dict: Error details and suggested actions
    """
    error_details = {
        "error_type": type(error).__name__,
        "error_message": str(error),
        "context": context,
        "timestamp": datetime.now().isoformat(),
        "suggested_actions": []
    }

    # Determine the type of error and provide appropriate suggestions
    error_msg_lower = str(error).lower()

    if "connection" in error_msg_lower or "connect" in error_msg_lower:
        error_details["suggested_actions"].extend([
            "Check if Qdrant server is running",
            "Verify QDRANT_URL in environment variables",
            "Ensure network connectivity to Qdrant server"
        ])
    elif "timeout" in error_msg_lower:
        error_details["suggested_actions"].extend([
            "Increase timeout value in configuration",
            "Check network connectivity",
            "Verify Qdrant server is not overloaded"
        ])
    elif "not found" in error_msg_lower or "404" in error_msg_lower:
        error_details["suggested_actions"].extend([
            "Verify collection name exists",
            "Create the collection if it doesn't exist",
            "Check for typos in collection name"
        ])
    elif "unauthorized" in error_msg_lower or "401" in error_msg_lower or "403" in error_msg_lower:
        error_details["suggested_actions"].extend([
            "Verify QDRANT_API_KEY in environment variables",
            "Check API key permissions",
            "Ensure API key has access to the specified collection"
        ])
    elif "memory" in error_msg_lower or "out of memory" in error_msg_lower:
        error_details["suggested_actions"].extend([
            "Reduce batch size for operations",
            "Process data in smaller chunks",
            "Increase available memory for Qdrant server"
        ])
    else:
        error_details["suggested_actions"].extend([
            "Check Qdrant server logs for more details",
            "Verify Qdrant server is healthy",
            "Review the operation that caused the error"
        ])

    logger.error(f"Qdrant connection error in {context}: {error}")
    logger.error(f"Suggested actions: {', '.join(error_details['suggested_actions'])}")

    return error_details


def safe_qdrant_operation(operation_func, *args, max_retries: int = 3, delay: float = 1.0, **kwargs) -> Any:
    """
    Execute a Qdrant operation safely with error handling and retries.
    Args:
        operation_func: The Qdrant operation function to execute
        *args: Arguments to pass to the operation function
        max_retries (int): Maximum number of retry attempts (default: 3)
        delay (float): Delay between retries in seconds (default: 1.0)
        **kwargs: Keyword arguments to pass to the operation function

    Returns:
        Any: Result of the operation or None if all retries failed
    """
    last_error = None

    for attempt in range(max_retries):
        try:
            result = operation_func(*args, **kwargs)
            logger.debug(f"Qdrant operation succeeded on attempt {attempt + 1}")
            return result

        except Exception as e:
            last_error = e
            error_context = f"{operation_func.__name__} attempt {attempt + 1}/{max_retries}"
            error_info = handle_qdrant_connection_error(e, error_context)

            if attempt < max_retries - 1:  # Don't sleep on the last attempt
                logger.info(f"Retrying Qdrant operation in {delay} seconds...")
                time.sleep(delay)
                delay *= 2  # Exponential backoff

    logger.error(f"Qdrant operation failed after {max_retries} attempts")
    return None


def validate_qdrant_connection(collection_name: str = "rag_embedding") -> tuple[bool, str]:
    """
    Validate that the Qdrant connection is working properly.
    Args:
        collection_name (str): Name of a collection to test access to

    Returns:
        tuple[bool, str]: (is_valid, message) where is_valid indicates if the connection is valid
                         and message provides details about the validation
    """
    if qdrant_client is None:
        return False, "Qdrant client not initialized"

    try:
        # Test connection by getting collections
        collections = qdrant_client.get_collections()
        logger.info(f"Successfully connected to Qdrant. Found {len(collections.collections)} collections.")

        # Check if the specified collection exists
        collection_names = [collection.name for collection in collections.collections]
        if collection_name in collection_names:
            logger.info(f"Collection '{collection_name}' exists")
        else:
            logger.info(f"Collection '{collection_name}' does not exist (this may be OK if creating new collection)")

        return True, "Qdrant connection is valid"

    except Exception as e:
        error_info = handle_qdrant_connection_error(e, "connection validation")
        return False, f"Qdrant connection validation failed: {error_info['error_message']}"


@timing_decorator
def batch_store_embeddings(embedding_vectors: List[List[float]], document_contents: List[DocumentContent],
                          collection_name: str = "rag_embedding", batch_size: int = 10) -> Dict:
    """
    Store multiple embeddings in Qdrant efficiently using batch operations.
    Args:
        embedding_vectors (List[List[float]]): List of embedding vectors to store
        document_contents (List[DocumentContent]): List of document contents with metadata
        collection_name (str): Name of the collection to store in (default: "rag_embedding")
        batch_size (int): Number of items to process in each batch (default: 10)

    Returns:
        Dict: Results of the batch operation with success/failure counts
    """
    from qdrant_client.http.models import PointStruct

    results = {
        "total_to_process": len(embedding_vectors),
        "successful_batches": 0,
        "failed_batches": 0,
        "successful_items": 0,
        "failed_items": 0,
        "errors": [],
        "batch_operation_summary": ""
    }

    # Validate inputs
    if len(embedding_vectors) != len(document_contents):
        error_msg = f"Mismatch: {len(embedding_vectors)} vectors but {len(document_contents)} documents"
        logger.error(error_msg)
        results["errors"].append(error_msg)
        return results

    if not embedding_vectors:
        logger.warning("No embeddings to store")
        results["batch_operation_summary"] = "No embeddings provided to store"
        return results

    try:
        # Process in batches
        for batch_start in range(0, len(embedding_vectors), batch_size):
            batch_end = min(batch_start + batch_size, len(embedding_vectors))
            batch_vectors = embedding_vectors[batch_start:batch_end]
            batch_docs = document_contents[batch_start:batch_end]

            try:
                # Create points for this batch
                points = []
                for vector, doc in zip(batch_vectors, batch_docs):
                    point = PointStruct(
                        id=doc.id,
                        vector=vector,
                        payload={
                            "url": doc.url,
                            "title": doc.title,
                            "content": doc.content,  # Store full content
                            "content_preview": doc.content[:500],  # Store first 500 chars as preview for backward compatibility
                            "created_at": doc.created_at.isoformat(),
                            "chunk_index": getattr(doc, 'chunk_index', 0)
                        }
                    )
                    points.append(point)

                # Store the batch in Qdrant
                qdrant_client.upsert(
                    collection_name=collection_name,
                    points=points
                )

                results["successful_batches"] += 1
                results["successful_items"] += len(points)
                logger.debug(f"Successfully stored batch {batch_start//batch_size + 1} with {len(points)} items")

            except Exception as batch_error:
                results["failed_batches"] += 1
                results["failed_items"] += len(batch_vectors)
                error_msg = f"Batch {batch_start//batch_size + 1} failed: {str(batch_error)}"
                logger.error(error_msg)
                results["errors"].append(error_msg)

        # Summary
        results["batch_operation_summary"] = (
            f"Batch operation completed: {results['successful_items']} successful items, "
            f"{results['failed_items']} failed items across {results['successful_batches']} "
            f"successful batches and {results['failed_batches']} failed batches"
        )
        logger.info(results["batch_operation_summary"])

    except Exception as e:
        logger.error(f"Error during batch storage operation: {e}")
        results["errors"].append(f"Critical error during batch storage: {str(e)}")

    return results


def store_document_embeddings(document_content: DocumentContent, embeddings: List[List[float]],
                            collection_name: str = "rag_embedding", batch_size: int = 10) -> Dict:
    """
    Store all embeddings for a single document (potentially multiple chunks) in Qdrant.
    Args:
        document_content (DocumentContent): The source document
        embeddings (List[List[float]]): List of embeddings for the document (one per chunk)
        collection_name (str): Name of the collection to store in
        batch_size (int): Number of items to process in each batch

    Returns:
        Dict: Results of the storage operation
    """
    if not embeddings:
        logger.warning("No embeddings provided for document storage")
        return {"success": False, "message": "No embeddings to store", "stored_count": 0}

    # Create document content objects for each chunk, with appropriate IDs and chunk indices
    chunk_contents = []
    for i, _ in enumerate(embeddings):
        # Create a copy of the document content for each chunk, with a unique ID and chunk index
        chunk_id = f"{document_content.id}_chunk_{i}"
        chunk_doc = DocumentContent(
            id=chunk_id,
            url=document_content.url,
            content=document_content.content,  # The full content, but this embedding represents a chunk
            title=document_content.title,
            created_at=document_content.created_at,
            chunk_index=i
        )
        chunk_contents.append(chunk_doc)

    # Store all embeddings for this document in batch
    batch_results = batch_store_embeddings(
        embedding_vectors=embeddings,
        document_contents=chunk_contents,
        collection_name=collection_name,
        batch_size=batch_size
    )

    return batch_results

# Load configuration
config = Config()

# Initialize Qdrant client
try:
    qdrant_client = get_qdrant_client()
except Exception as e:
    logger.error(f"Could not initialize Qdrant client: {e}")
    qdrant_client = None


def is_valid_url(url: str) -> bool:
    """Check if the provided string is a valid URL."""
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except Exception:
        return False 
def is_docusaurus_site(url: str) -> bool:
    """Check if the URL is likely a Docusaurus site by looking for common indicators."""
    try:
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            return False

        content = response.text.lower()

        # Look for common Docusaurus indicators
        docusaurus_indicators = [
            'docusaurus',
            'data-theme',
            'navbar',
            'doc-sidebar',
            'footer'
        ]

        return any(indicator in content for indicator in docusaurus_indicators)
    except Exception:
        return False


def validate_docusaurus_site(url: str) -> tuple[bool, str]:
    """
    Validate that the provided URL is accessible and likely a Docusaurus site.
    Returns:
        tuple[bool, str]: (is_valid, message) where is_valid indicates if the site is valid
                         and message provides details about the validation
    """
    if not is_valid_url(url):
        return False, f"Invalid URL format: {url}"

    try:
        # Check if site is accessible
        response = requests.head(url, timeout=10)
        if response.status_code >= 400:
            return False, f"Site returned error status: {response.status_code}"
    except requests.exceptions.RequestException as e:
        return False, f"Site is not accessible: {str(e)}"

    # Check if it's likely a Docusaurus site
    if not is_docusaurus_site(url):
        logger.warning(f"URL {url} may not be a Docusaurus site, but will proceed anyway")

    return True, "Site is accessible and appears to be valid"


def validate_and_access_site(url: str) -> tuple[bool, str]:
    """
    Comprehensive validation of a Docusaurus site including accessibility check.
    Returns:
        tuple[bool, str]: (is_valid, message) where is_valid indicates if the site is valid
                         and message provides details about the validation
    """
    # First validate the URL format
    if not is_valid_url(url):
        return False, f"Invalid URL format: {url}"

    # Check if the site is accessible
    try:
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            return False, f"Site returned status code: {response.status_code}"
    except requests.exceptions.RequestException as e:
        return False, f"Could not access site: {str(e)}"

    return True, "Site is accessible"


def retry_api_call(func, max_retries=3, delay=1, backoff=2, exceptions=(Exception,)):
    """
    Decorator to retry API calls with exponential backoff.
    Args:
        func: The function to retry
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff: Multiplier for delay after each retry
        exceptions: Tuple of exceptions to catch and retry on
    """
    def wrapper(*args, **kwargs):
        retries = 0
        current_delay = delay

        while retries < max_retries:
            try:
                return func(*args, **kwargs)
            except exceptions as e:
                retries += 1
                if retries >= max_retries:
                    logger.error(f"Failed after {max_retries} attempts: {str(e)}")
                    raise e

                logger.warning(f"Attempt {retries} failed: {str(e)}. Retrying in {current_delay}s...")
                time.sleep(current_delay)
                current_delay *= backoff

        return None  # This line should never be reached
    return wrapper


def handle_api_error(error: Exception, context: str = ""):
    """
    Standardized error handling for API calls.
    Args:
        error: The exception that occurred
        context: Additional context about where the error occurred
    """
    error_msg = f"API Error in {context}: {str(error)}"
    logger.error(error_msg)

    # Different handling based on error type
    if isinstance(error, requests.exceptions.Timeout):
        logger.error("Request timed out. Consider increasing timeout or checking network connectivity.")
    elif isinstance(error, requests.exceptions.ConnectionError):
        logger.error("Connection error. Check network connectivity and URL.")
    elif isinstance(error, requests.exceptions.HTTPError):
        logger.error(f"HTTP error occurred: {error.response.status_code}")
    elif "rate limit" in str(error).lower() or "429" in str(error):
        logger.error("Rate limit exceeded. Consider implementing rate limiting.")

    return error_msg


def safe_api_call(func, context: str = "", default_return=None, max_retries=3):
    """
    Execute an API call safely with error handling and retries.
    Args:
        func: The API function to call
        context: Context description for logging
        default_return: Value to return if all retries fail
        max_retries: Maximum number of retry attempts

    Returns:
        The result of the function call or default_return if it fails
    """
    try:
        # Use the retry decorator
        retry_func = retry_api_call(
            func,
            max_retries=max_retries,
            exceptions=(requests.exceptions.RequestException, ConnectionError)
        )
        return retry_func()
    except Exception as e:
        handle_api_error(e, context)
        return default_return


def rate_limit(calls_per_second=1):
    """
    Decorator to limit the rate of function calls.

    Args:
        calls_per_second: Maximum number of calls allowed per second
    """
    min_interval = 1.0 / calls_per_second
    last_called = [0.0]

    def decorator(func):
        def wrapper(*args, **kwargs):
            elapsed = time.time() - last_called[0]
            left_to_wait = min_interval - elapsed
            if left_to_wait > 0:
                time.sleep(left_to_wait)
            ret = func(*args, **kwargs)
            last_called[0] = time.time()
            return ret
        return wrapper
    return decorator


def is_content_page(url: str) -> bool:
    """
    Determine if a URL likely contains content that should be processed.
    Args:
        url (str): The URL to check

    Returns:
        bool: True if the URL is likely a content page, False otherwise
    """
    # Convert to lowercase for case-insensitive matching
    url_lower = url.lower()

    # URLs to exclude (non-content pages)
    excluded_patterns = [
        # Navigation elements
        '/nav/',
        '/navbar/',
        '/menu/',
        '/header/',
        '/footer/',
        # Common non-content extensions
        '.pdf',
        '.jpg',
        '.jpeg',
        '.png',
        '.gif',
        '.css',
        '.js',
        '.xml',
        '.json',
        # Common non-content paths
        '/api/',
        '/admin/',
        '/login/',
        '/logout/',
        '/register/',
        '/signup/',
        '/search/',
        '/tag/',
        '/category/',
        '/feed',
        # Query parameters that suggest non-content
        'share=',
        'print=',
        # File extensions without slash (to catch files at root)
        'sitemap.xml',
        'robots.txt',
    ]

    # Check for excluded patterns
    for pattern in excluded_patterns:
        if pattern in url_lower:
            return False

    # Additional check: if URL has no path (just domain) or ends with common content indicators
    path = urlparse(url).path
    if path == '/' or path.endswith(('/', '.html', '.htm')):
        return True

    # If it doesn't match any exclusion pattern and has a reasonable path, consider it content
    return True


# Global variable to track request times for rate limiting
_request_times = []

def respectful_delay(min_delay: float = 0.5, max_requests_per_second: float = 2):
    """
    Implements respectful delay between requests to not overload the server.

    Args:
        min_delay (float): Minimum delay in seconds between requests
        max_requests_per_second (float): Maximum number of requests per second
    """
    import time
    global _request_times

    now = time.time()
    # Remove requests older than 1 second to calculate current rate
    _request_times = [req_time for req_time in _request_times if now - req_time < 1]

    # If we're making too many requests per second, wait
    if len(_request_times) >= max_requests_per_second:
        sleep_time = 1 - (now - _request_times[0]) if _request_times else min_delay
        if sleep_time > 0:
            time.sleep(sleep_time)

    # Add current request time
    _request_times.append(time.time())

    # Also ensure minimum delay
    time.sleep(min_delay)


def get_all_urls(root_url: str, max_depth: int = 2, delay: float = 0.5) -> List[str]:
    """
    Discovers and returns all accessible URLs from a given Docusaurus site.

    Args:
        root_url (str): The root URL of the Docusaurus site to crawl
        max_depth (int, optional): Maximum depth to crawl (default: 2)
        delay (float, optional): Delay between requests in seconds (default: 0.5)

    Returns:
        List[str]: A list of all discovered URLs from the site
    """
    from urllib.parse import urljoin, urlparse
    import time

    # Validate the root URL
    is_valid, msg = validate_and_access_site(root_url)
    if not is_valid:
        logger.error(f"Invalid root URL: {msg}")
        return []

    # Set to store discovered URLs
    discovered_urls = set()
    # Set to track visited URLs to avoid infinite loops
    visited_urls = set()

    # Queue for BFS crawling: (current_url, current_depth)
    queue = [(root_url, 0)]

    while queue:
        current_url, current_depth = queue.pop(0)

        # Skip if already visited or max depth reached
        if current_url in visited_urls or current_depth > max_depth:
            continue

        visited_urls.add(current_url)
        logger.info(f"Crawling: {current_url} (depth: {current_depth})")

        try:
            # Apply respectful delay to not overload the server
            respectful_delay(min_delay=delay)

            # Fetch the page content
            response = safe_api_call(
                lambda: requests.get(current_url, timeout=10),
                context=f"fetching {current_url}",
                default_return=None
            )

            if response is None:
                logger.warning(f"Failed to fetch {current_url}: No response received")
                continue
            elif response.status_code == 429:
                logger.warning(f"Rate limited (429) by server: {current_url}. Waiting longer before continuing...")
                time.sleep(5)  # Wait longer if rate limited
                continue
            elif response.status_code == 403:
                logger.warning(f"Access forbidden (403) for {current_url}, possibly due to crawling restrictions")
                continue
            elif response.status_code != 200:
                logger.warning(f"Failed to fetch {current_url}: {response.status_code}")
                continue

            # Parse the HTML content
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(response.text, 'html.parser')

            # Find all links on the page
            links = soup.find_all('a', href=True)

            for link in links:
                href = link['href']

                # Convert relative URLs to absolute URLs
                absolute_url = urljoin(current_url, href)

                # Only process URLs from the same domain as the root URL
                root_domain = urlparse(root_url).netloc
                link_domain = urlparse(absolute_url).netloc

                if root_domain == link_domain and absolute_url not in discovered_urls:
                    # Check if it's a valid URL and a content page
                    if is_valid_url(absolute_url) and is_content_page(absolute_url):
                        discovered_urls.add(absolute_url)
                        # Add to queue if not exceeding max depth
                        if current_depth < max_depth:
                            queue.append((absolute_url, current_depth + 1))

        except Exception as e:
            logger.error(f"Error crawling {current_url}: {str(e)}")
            continue

    logger.info(f"Discovered {len(discovered_urls)} URLs from {root_url}")
    return list(discovered_urls)


def extract_text_from_url(url: str) -> Dict:
    """
    Extracts and cleans text content from a single URL.
    Args:
        url (str): The URL to extract text from

    Returns:
        dict: Contains the following keys:
            - "url": The original URL
            - "title": The page title
            - "content": The cleaned text content
            - "status": Extraction status ("success", "error")
    """
    # Validate URL first
    if not is_valid_url(url):
        logger.error(f"Invalid URL format: {url}")
        return {
            "url": url,
            "title": "",
            "content": "",
            "status": "error"
        }

    try:
        # Fetch the page content
        response = safe_api_call(
            lambda: requests.get(url, timeout=10),
            context=f"fetching {url}",
            default_return=None
        )

        if response is None:
            logger.error(f"Failed to fetch {url}: No response received")
            return {
                "url": url,
                "title": "",
                "content": "",
                "status": "error"
            }

        if response.status_code == 404:
            logger.warning(f"Page not found (404): {url}")
            return {
                "url": url,
                "title": "",
                "content": "",
                "status": "error"
            }
        elif response.status_code == 403:
            logger.warning(f"Access forbidden (403) - may be blocked by robots.txt: {url}")
            return {
                "url": url,
                "title": "",
                "content": "",
                "status": "error"
            }
        elif response.status_code == 429:
            logger.warning(f"Rate limited (429) by server: {url}")
            return {
                "url": url,
                "title": "",
                "content": "",
                "status": "error"
            }
        elif response.status_code != 200:
            logger.error(f"Failed to fetch {url}: HTTP {response.status_code}")
            return {
                "url": url,
                "title": "",
                "content": "",
                "status": "error"
            }

        from bs4 import BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract title
        title_tag = soup.find('title')
        title = title_tag.get_text().strip() if title_tag else "No Title"

        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()

        # Try to find main content area (common in Docusaurus sites)
        main_content = None
        # Look for common Docusaurus content containers
        for selector in ['.main-wrapper', '.theme-doc-markdown', '.markdown', 'main', '.container', '.post', '.doc-content']:
            main_content = soup.select_one(selector)
            if main_content:
                break

        # If no specific content container found, use body
        if not main_content:
            main_content = soup.find('body')

        # Extract text from the main content
        if main_content:
            content = main_content.get_text()
        else:
            content = soup.get_text()

        # Clean up the text (remove extra whitespace)
        lines = (line.strip() for line in content.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        content = ' '.join(chunk for chunk in chunks if chunk)

        # Verify content is substantial
        if not content or len(content.strip()) < 50:
            logger.warning(f"Content too short from {url} ({len(content)} chars)")
            return {
                "url": url,
                "title": title,
                "content": content,
                "status": "error"
            }

        logger.info(f"Successfully extracted content from {url}")
        return {
            "url": url,
            "title": title,
            "content": content,
            "status": "success"
        }

    except requests.exceptions.Timeout:
        logger.error(f"Timeout error extracting text from {url}")
        return {
            "url": url,
            "title": "",
            "content": "",
            "status": "error"
        }
    except requests.exceptions.ConnectionError:
        logger.error(f"Connection error extracting text from {url}")
        return {
            "url": url,
            "title": "",
            "content": "",
            "status": "error"
        }
    except Exception as e:
        logger.error(f"Unexpected error extracting text from {url}: {str(e)}")
        return {
            "url": url,
            "title": "",
            "content": "",
            "status": "error"
        }


def test_content_extraction(urls: List[str], sample_size: int = 5) -> Dict:
    """
    Test content extraction with sample Docusaurus URLs.
    Args:
        urls: List of URLs to test extraction on
        sample_size: Number of URLs to test (default: 5)

    Returns:
        Dict with test results
    """
    results = {
        "total_tested": 0,
        "successful": 0,
        "failed": 0,
        "errors": [],
        "sample_results": []
    }

    # Limit to sample size
    test_urls = urls[:sample_size] if len(urls) > sample_size else urls

    logger.info(f"Testing content extraction on {len(test_urls)} sample URLs")

    for url in test_urls:
        results["total_tested"] += 1
        extraction_result = extract_text_from_url(url)

        if extraction_result["status"] == "success":
            results["successful"] += 1
            results["sample_results"].append(extraction_result)
        else:
            results["failed"] += 1
            results["errors"].append(f"Failed to extract from {url}")

    logger.info(f"Content extraction test results: {results['successful']} successful, {results['failed']} failed")
    return results


def verify_clean_content(content: str) -> Dict:
    """
    Verify that extracted content is clean (no HTML tags, navigation elements).
    Args:
        content: Content to verify

    Returns:
        Dict with verification results
    """
    import re

    verification = {
        "is_clean": True,
        "issues": [],
        "content_length": len(content),
        "html_tags_found": 0,
        "suspicious_patterns": 0
    }

    # Check for HTML tags
    html_tag_pattern = r'<[^>]+>'
    html_tags = re.findall(html_tag_pattern, content)
    verification["html_tags_found"] = len(html_tags)

    if html_tags:
        verification["issues"].append(f"Found {len(html_tags)} HTML tags")
        verification["is_clean"] = False

    # Check for common navigation patterns that might not have been removed
    nav_patterns = [
        r'navigation',
        r'menu',
        r'footer',
        r'header',
        r'sidebar',
        r'nav',
        r'breadcrumb',
    ]

    content_lower = content.lower()
    for pattern in nav_patterns:
        if pattern in content_lower:
            verification["issues"].append(f"Found potential navigation pattern: {pattern}")
            verification["suspicious_patterns"] += 1

    # Check content quality
    if len(content.strip()) == 0:
        verification["issues"].append("Content is empty")
        verification["is_clean"] = False
    elif len(content.strip()) < 50:
        verification["issues"].append(f"Content is very short ({len(content.strip())} chars)")
        verification["is_clean"] = False

    # Check for excessive special characters that might indicate poor extraction
    special_char_ratio = len(re.findall(r'[^a-zA-Z0-9\s]', content)) / len(content) if content else 0
    if special_char_ratio > 0.3:  # More than 30% special characters
        verification["issues"].append(f"High ratio of special characters: {special_char_ratio:.2%}")
        verification["is_clean"] = False

    return verification


@timing_decorator
def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 100) -> List[str]:
    """
    Split large text content into smaller chunks of specified size with overlap.
    Args:
        text (str): The text to be chunked
        chunk_size (int): Maximum size of each chunk (default: 1000)
        chunk_overlap (int): Number of characters to overlap between chunks (default: 100)

    Returns:
        List[str]: A list of text chunks

    Raises:
        ValueError: If parameters are invalid
    """
    # Input validation
    if not isinstance(text, str):
        raise ValueError(f"text must be a string, got {type(text)}")

    if not text or len(text.strip()) == 0:
        logger.debug("Empty text provided for chunking, returning empty list")
        return []

    if not isinstance(chunk_size, int):
        raise ValueError(f"chunk_size must be an integer, got {type(chunk_size)}")

    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")

    if not isinstance(chunk_overlap, int):
        raise ValueError(f"chunk_overlap must be an integer, got {type(chunk_overlap)}")

    if chunk_overlap < 0:
        raise ValueError(f"chunk_overlap cannot be negative, got {chunk_overlap}")

    if chunk_overlap >= chunk_size:
        raise ValueError(f"chunk_overlap ({chunk_overlap}) must be less than chunk_size ({chunk_size})")

    # Additional validation for reasonable limits
    if chunk_size > 10000:
        raise ValueError(f"chunk_size ({chunk_size}) seems too large, maximum recommended is 10000")

    chunks = []
    start = 0

    while start < len(text):
        # Calculate end position for this chunk
        end = start + chunk_size

        # If we're near the end of the text, adjust end to avoid exceeding text length
        if end >= len(text):
            # This is the last chunk
            chunks.append(text[start:])
            break

        # Extract the chunk
        chunk = text[start:end]
        chunks.append(chunk)

        # Move start position by chunk_size minus overlap
        start += chunk_size - chunk_overlap

    logger.info(f"Text chunked into {len(chunks)} pieces (chunk_size: {chunk_size}, overlap: {chunk_overlap})")
    return chunks 
@timing_decorator
def embed(texts: List[str], model: str = "embed-english-v3.0") -> List[List[float]]:
    """
    Generate embeddings for a list of texts using Cohere API.

    Args:
        texts (List[str]): List of texts to generate embeddings for
        model (str): The embedding model to use (default: "embed-english-v3.0")

    Returns:
        List[List[float]]: List of embedding vectors, where each vector is a list of floats

    Raises:
        ValueError: If parameters are invalid
    """
    # Input validation
    if not isinstance(texts, list):
        raise ValueError(f"texts must be a list, got {type(texts)}")

    if not texts:
        logger.debug("Empty texts list provided for embedding")
        return []

    # Validate each text in the list
    for i, text in enumerate(texts):
        if not isinstance(text, str):
            raise ValueError(f"text at index {i} must be a string, got {type(text)}")

    if not isinstance(model, str):
        raise ValueError(f"model must be a string, got {type(model)}")

    if not model.strip():
        raise ValueError("model cannot be empty")

    import cohere

    # Validate API key
    if not config.cohere_api_key:
        logger.error("COHERE_API_KEY not set in environment variables")
        raise ValueError("Cohere API key is required")

    # Initialize Cohere client
    try:
        co = cohere.Client(config.cohere_api_key)
    except Exception as e:
        logger.error(f"Failed to initialize Cohere client: {e}")
        raise

    # Validate input texts
    # Filter out empty texts
    valid_texts = [text for text in texts if text and len(text.strip()) > 0]
    if not valid_texts:
        logger.warning("No valid texts found for embedding after filtering")
        return []

    # Check if we have too many texts (Cohere has limits)
    # For Cohere's v3 models, the batch size is typically limited to 96 texts
    max_batch_size = 96
    all_embeddings = []

    try:
        # Process in batches if needed
        for i in range(0, len(valid_texts), max_batch_size):
            batch = valid_texts[i:i + max_batch_size]

            # Use Cohere's embed API
            response = co.embed(
                texts=batch,
                model=model,
                input_type="search_document"  # Appropriate for document search
            )

            # Extract embeddings from response
            batch_embeddings = [embedding for embedding in response.embeddings]
            all_embeddings.extend(batch_embeddings)

            logger.info(f"Generated embeddings for batch {i//max_batch_size + 1} of {len(valid_texts)} texts")

    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        raise

    logger.info(f"Successfully generated embeddings for {len(valid_texts)} texts")
    return all_embeddings


def validate_cohere_api_key(api_key: Optional[str] = None) -> tuple[bool, str]:
    """
    Validate the Cohere API key by making a test request.

    Args:
        api_key (Optional[str]): API key to validate. If None, uses the one from config.

    Returns:
        tuple[bool, str]: (is_valid, message) where is_valid indicates if the API key is valid
                         and message provides details about the validation
    """
    import cohere

    # Use provided API key or fall back to config
    key_to_validate = api_key if api_key is not None else config.cohere_api_key

    if not key_to_validate:
        return False, "No API key provided for validation"

    try:
        # Initialize Cohere client with the provided key
        co = cohere.Client(key_to_validate)

        # Make a simple API call to test the key (using the embed endpoint with minimal input)
        # We'll use a very short text to avoid hitting usage limits during validation
        response = co.embed(
            texts=["test"],
            model="embed-english-v3.0",
            input_type="search_document"
        )

        # If we get a response without error, the API key is valid
        if response and hasattr(response, 'embeddings') and len(response.embeddings) > 0:
            logger.info("Cohere API key validation successful")
            return True, "API key is valid and working correctly"
        else:
            return False, "API key validation failed - no embeddings returned"

    except cohere.UnauthorizedError:
        return False, "Invalid API key - unauthorized access"
    except Exception as e:
        error_msg = str(e).lower()
        if "rate limit" in error_msg or "429" in error_msg:
            # If it's a rate limit error, the API key is technically valid but we're limited
            logger.warning(f"API key appears valid but hit rate limit during validation: {e}")
            return True, "API key is valid but rate limited during validation"
        else:
            logger.error(f"Unexpected error during API key validation: {e}")
            return False, f"Unexpected error during validation: {e}"


# Global variable to track Cohere API request times for rate limiting
_cohere_request_times = []


def cohere_rate_limit(max_requests_per_minute: int = 60):
    """
    Decorator to limit the rate of Cohere API calls to respect API limits.
    Args:
        max_requests_per_minute (int): Maximum number of requests allowed per minute (default: 60)
    """
    import time
    from functools import wraps

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            global _cohere_request_times
            now = time.time()

            # Remove requests older than 1 minute to calculate current rate
            _cohere_request_times = [req_time for req_time in _cohere_request_times if now - req_time < 60]

            # If we're making too many requests per minute, wait
            if len(_cohere_request_times) >= max_requests_per_minute:
                sleep_time = 60 - (now - _cohere_request_times[0]) if _cohere_request_times else 60
                if sleep_time > 0:
                    logger.info(f"Approaching Cohere API rate limit, sleeping for {sleep_time:.2f} seconds")
                    time.sleep(sleep_time)

            # Add current request time
            _cohere_request_times.append(time.time())

            return func(*args, **kwargs)
        return wrapper
    return decorator


def make_rate_limited_cohere_request(texts: List[str], model: str = "embed-english-v3.0",
                                   input_type: str = "search_document", max_requests_per_minute: int = 60) -> Any:
    """
    Make a rate-limited request to Cohere API for embedding generation.
    Args:
        texts (List[str]): List of texts to generate embeddings for
        model (str): The embedding model to use (default: "embed-english-v3.0")
        input_type (str): The input type for the model (default: "search_document")
        max_requests_per_minute (int): Maximum requests per minute (default: 60)

    Returns:
        Any: The Cohere API response
    """
    import cohere
    from functools import wraps

    # Validate API key first
    if not config.cohere_api_key:
        logger.error("COHERE_API_KEY not set in environment variables")
        raise ValueError("Cohere API key is required")

    # Initialize Cohere client
    co = cohere.Client(config.cohere_api_key)

    # Apply rate limiting
    rate_limited_func = cohere_rate_limit(max_requests_per_minute)

    @rate_limited_func
    def _make_request():
        return co.embed(texts=texts, model=model, input_type=input_type)

    return _make_request()


def test_embedding_generation(sample_texts: Optional[List[str]] = None) -> Dict:
    """
    Test embedding generation with sample text content.

    Args:
        sample_texts (Optional[List[str]]): Optional list of sample texts to test with. If None, uses default sample texts.

    Returns:
        Dict: Test results with information about success/failure
    """
    if sample_texts is None:
        # Default sample texts for testing
        sample_texts = [
            "This is a sample text for testing embedding generation.",
            "Artificial intelligence and machine learning are transforming technology.",
            "Python is a versatile programming language used for various applications.",
            "Natural language processing enables computers to understand human language.",
            "Vector embeddings represent text in high-dimensional space for similarity matching."
        ]

    logger.info(f"Testing embedding generation with {len(sample_texts)} sample texts")

    results = {
        "total_texts": len(sample_texts),
        "successful_embeddings": 0,
        "failed_embeddings": 0,
        "total_embeddings": 0,
        "average_embedding_size": 0,
        "errors": [],
        "sample_results": []
    }

    try:
        # Generate embeddings for the sample texts
        embeddings = embed(sample_texts)

        results["total_embeddings"] = len(embeddings)

        if len(embeddings) == len(sample_texts):
            results["successful_embeddings"] = len(sample_texts)
            logger.info(f"Successfully generated embeddings for all {len(sample_texts)} texts")

            # Calculate average embedding size (dimensionality)
            if embeddings and len(embeddings) > 0:
                total_dimensions = sum(len(embedding) for embedding in embeddings)
                results["average_embedding_size"] = total_dimensions / len(embeddings)
                logger.info(f"Average embedding dimensionality: {results['average_embedding_size']:.2f}")
        else:
            results["failed_embeddings"] = len(sample_texts) - len(embeddings)
            results["errors"].append(f"Mismatch in number of embeddings: expected {len(sample_texts)}, got {len(embeddings)}")

        # Sample some results for verification
        for i, (text, embedding) in enumerate(zip(sample_texts[:3], embeddings[:3])):
            results["sample_results"].append({
                "text_preview": text[:50] + "..." if len(text) > 50 else text,
                "embedding_length": len(embedding) if embedding else 0,
                "first_5_values": embedding[:5] if embedding else []
            })

    except Exception as e:
        logger.error(f"Error during embedding generation test: {e}")
        results["failed_embeddings"] = len(sample_texts)
        results["errors"].append(f"Error during embedding generation: {str(e)}")

    logger.info(f"Embedding generation test completed: {results['successful_embeddings']} successful, {results['failed_embeddings']} failed")
    return results


def verify_embeddings_within_limits(texts: List[str], model: str = "embed-english-v3.0") -> Dict:
    """
    Verify that embeddings are generated within Cohere API size limits.
    Args:
        texts (List[str]): List of texts to verify
        model (str): The embedding model to use (default: "embed-english-v3.0")

    Returns:
        Dict: Verification results with information about API limits compliance
    """
    import cohere

    results = {
        "total_texts": len(texts),
        "texts_within_limits": 0,
        "texts_exceeding_limits": 0,
        "total_text_chars": 0,
        "max_text_chars": 0,
        "max_allowed_chars": 40000,  # Cohere API limit for embed endpoint
        "verification_passed": True,
        "issues": [],
        "model": model
    }

    # Calculate text statistics
    for text in texts:
        text_len = len(text) if text else 0
        results["total_text_chars"] += text_len
        results["max_text_chars"] = max(results["max_text_chars"], text_len)

        # Check if individual text exceeds API limits
        if text_len > results["max_allowed_chars"]:
            results["texts_exceeding_limits"] += 1
            results["issues"].append(f"Text with {text_len} chars exceeds API limit of {results['max_allowed_chars']}")
            results["verification_passed"] = False
        else:
            results["texts_within_limits"] += 1

    # Check if total batch exceeds limits (for batch processing)
    total_chars = sum(len(text) for text in texts if text)
    if total_chars > results["max_allowed_chars"]:
        results["issues"].append(f"Total text length ({total_chars} chars) exceeds API limit of {results['max_allowed_chars']} when processed as a single batch")
        # This is acceptable if we chunk properly, so we don't set verification_passed to False for this

    logger.info(f"Embedding size verification: {results['texts_within_limits']} within limits, {results['texts_exceeding_limits']} exceeding limits")
    return results


def handle_large_documents_for_embedding(texts: List[str], chunk_size: int = 1000, chunk_overlap: int = 100,
                                       model: str = "embed-english-v3.0") -> List[List[float]]:
    """
    Handle documents that exceed API size limits by chunking them before embedding generation.
    Args:
        texts (List[str]): List of texts to process (some may exceed API limits)
        chunk_size (int): Size of each chunk in characters (default: 1000)
        chunk_overlap (int): Number of characters to overlap between chunks (default: 100)
        model (str): The embedding model to use (default: "embed-english-v3.0")

    Returns:
        List[List[float]]: Combined embeddings for all input texts (flattened)
    """
    # Verify the texts are within API limits
    verification = verify_embeddings_within_limits(texts, model)

    # If all texts are within limits, process them directly
    if verification["verification_passed"] and verification["texts_exceeding_limits"] == 0:
        logger.info("All texts are within API limits, processing directly")
        return embed(texts)

    logger.info(f"Found {verification['texts_exceeding_limits']} texts exceeding API limits, processing with chunking")

    all_embeddings = []

    for i, text in enumerate(texts):
        if not text or len(text.strip()) == 0:
            logger.warning(f"Skipping empty text at index {i}")
            continue

        # If text is within limits, process directly
        if len(text) <= verification["max_allowed_chars"]:
            logger.debug(f"Text {i} is within limits, processing directly")
            text_embeddings = embed([text], model=model)
            all_embeddings.extend(text_embeddings)
        else:
            # Text exceeds limits, chunk it
            logger.info(f"Text {i} exceeds API limits ({len(text)} chars), chunking for embedding")
            chunks = chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

            logger.info(f"Text {i} chunked into {len(chunks)} pieces")

            # Generate embeddings for each chunk
            chunk_embeddings = embed(chunks, model=model)
            all_embeddings.extend(chunk_embeddings)

    logger.info(f"Successfully processed {len(texts)} texts with {len(all_embeddings)} total embeddings")
    return all_embeddings


def main():
    """Main function to orchestrate the entire embedding pipeline."""
    parser = argparse.ArgumentParser(description='Docusaurus Embedding Pipeline')
    parser.add_argument('--url', required=True, help='Root URL of the Docusaurus site to process')
    parser.add_argument('--collection-name', default='rag_embedding', help='Name of Qdrant collection (default: rag_embedding)')
    parser.add_argument('--chunk-size', type=int, default=1000, help='Size of text chunks (default: 1000)')
    parser.add_argument('--chunk-overlap', type=int, default=100, help='Overlap between chunks (default: 100)')
    parser.add_argument('--batch-size', type=int, default=10, help='Batch size for vector storage (default: 10)')
    parser.add_argument('--test-mode', action='store_true', help='Run in test mode with limited processing')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode with additional logging')

    args = parser.parse_args()

    # Set debug logging if debug mode is enabled
    if args.debug:
        logger.setLevel(logging.DEBUG)
        for handler in logger.handlers:
            handler.setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled")

    # Initialize pipeline status tracking
    pipeline_status = ProcessingPipeline(
        status="running",
        source_url=args.url,
        start_time=datetime.now()
    )
    logger.info(f"Starting embedding pipeline for URL: {args.url}")
    logger.debug(f"Arguments: url={args.url}, collection_name={args.collection_name}, chunk_size={args.chunk_size}, chunk_overlap={args.chunk_overlap}, batch_size={args.batch_size}, test_mode={args.test_mode}, debug={args.debug}")

    try:
        # Validate the URL
        logger.info("Validating URL...")
        is_valid, validation_msg = validate_and_access_site(args.url)
        if not is_valid:
            logger.error(f"URL validation failed: {validation_msg}")
            pipeline_status.status = "failed"
            pipeline_status.errors.append(f"URL validation failed: {validation_msg}")
            return

        # Validate the Cohere API key
        logger.info("Validating Cohere API key...")
        api_key_valid, api_msg = validate_cohere_api_key()
        if not api_key_valid:
            logger.error(f"Cohere API validation failed: {api_msg}")
            pipeline_status.status = "failed"
            pipeline_status.errors.append(f"Cohere API validation failed: {api_msg}")
            return
        else:
            logger.info("Cohere API key validated successfully")

        # Validate Qdrant connection
        logger.info("Validating Qdrant connection...")
        qdrant_valid, qdrant_msg = validate_qdrant_connection(args.collection_name)
        if not qdrant_valid:
            logger.error(f"Qdrant connection validation failed: {qdrant_msg}")
            pipeline_status.status = "failed"
            pipeline_status.errors.append(f"Qdrant connection validation failed: {qdrant_msg}")
            return
        else:
            logger.info("Qdrant connection validated successfully")

        # Create the collection in Qdrant
        logger.info(f"Creating collection '{args.collection_name}' in Qdrant...")
        collection_created = create_collection(args.collection_name)
        if not collection_created:
            logger.error(f"Failed to create collection '{args.collection_name}' in Qdrant")
            pipeline_status.status = "failed"
            pipeline_status.errors.append(f"Failed to create collection '{args.collection_name}'")
            return
        else:
            logger.info(f"Collection '{args.collection_name}' created or already exists")

        # Get all URLs from the Docusaurus site
        logger.info("Discovering URLs from the Docusaurus site...")
        urls = get_all_urls(args.url, max_depth=config.max_depth, delay=config.request_delay)
        logger.info(f"Discovered {len(urls)} URLs to process")
        pipeline_status.total_documents = len(urls)

        if not urls:
            logger.error("No URLs discovered, exiting")
            pipeline_status.status = "failed"
            pipeline_status.errors.append("No URLs discovered")
            return

        # Limit URLs if in test mode
        if args.test_mode:
            urls = urls[:5]  # Limit to first 5 URLs in test mode
            logger.info(f"Test mode enabled: processing only {len(urls)} URLs")

        # Extract text content from each URL
        logger.info("Extracting text content from URLs...")
        documents = []
        for i, url in enumerate(urls):
            logger.info(f"Processing ({i+1}/{len(urls)}): {url}")
            extraction_result = extract_text_from_url(url)

            if extraction_result["status"] == "success":
                # Create a DocumentContent object for successful extractions
                # Generate UUID from content hash for deterministic IDs
                content_hash = hashlib.md5((url + extraction_result["content"][:100]).encode()).hexdigest()
                doc_uuid = str(uuid.UUID(content_hash))
                doc_content = DocumentContent(
                    id=doc_uuid,
                    url=url,
                    content=extraction_result["content"],
                    title=extraction_result["title"],
                    created_at=datetime.now()
                )

                # Validate the document content
                validation_errors = doc_content.validate()
                if not validation_errors:
                    documents.append(doc_content)
                    logger.debug(f"Successfully extracted and validated content from {url}")
                    pipeline_status.processed_documents += 1
                else:
                    logger.warning(f"Validation errors for {url}: {validation_errors}")
                    pipeline_status.errors.extend([f"{url}: {error}" for error in validation_errors])
            else:
                logger.warning(f"Failed to extract content from {url}")
                pipeline_status.errors.append(f"Failed to extract content from {url}")

        logger.info(f"Successfully extracted content from {len(documents)} URLs out of {len(urls)}")

        if not documents:
            logger.error("No valid documents extracted, exiting")
            pipeline_status.status = "failed"
            pipeline_status.errors.append("No valid documents extracted")
            return

        # Store embeddings in Qdrant - process each document individually to avoid memory issues
        logger.info(f"Processing {len(documents)} documents for embedding and storage...")
        successful_stores = 0
        failed_stores = 0
        total_embeddings_generated = 0

        # For each document, chunk it and store each chunk's embedding
        for doc in documents:
            # Skip documents that are too large to avoid memory issues
            if len(doc.content) > 500000:  # 500KB limit
                logger.warning(f"Skipping document {doc.id} - content too large ({len(doc.content)} chars)")
                failed_stores += 1
                continue

            # Chunk the document content
            try:
                chunks = chunk_text(doc.content, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
            except MemoryError:
                logger.error(f"MemoryError while chunking document {doc.id} ({len(doc.content)} chars)")
                failed_stores += 1
                continue

            if len(chunks) == 1 and chunks[0] == doc.content:
                # No chunking was needed, process as single document
                logger.debug(f"Document {doc.id} did not require chunking")

                # Generate embedding for the single content
                doc_embeddings = embed([doc.content])
                total_embeddings_generated += len(doc_embeddings) if doc_embeddings else 0

                if doc_embeddings:
                    # Store this single embedding
                    success = save_chunk_to_qdrant(doc_embeddings[0], doc, args.collection_name)
                    if success:
                        successful_stores += 1
                    else:
                        failed_stores += 1
                        pipeline_status.errors.append(f"Failed to store embedding for document {doc.id}")
            else:
                # Document was chunked, process each chunk
                logger.debug(f"Document {doc.id} was chunked into {len(chunks)} pieces")

                # Generate embeddings for all chunks at once (more efficient)
                chunk_embeddings = embed(chunks)
                total_embeddings_generated += len(chunk_embeddings) if chunk_embeddings else 0

                if not chunk_embeddings or len(chunk_embeddings) != len(chunks):
                    logger.error(f"Failed to generate embeddings for document {doc.id}: expected {len(chunks)}, got {len(chunk_embeddings) if chunk_embeddings else 0}")
                    failed_stores += len(chunks)
                    continue

                # Prepare lists for batch processing
                chunk_docs = []
                for chunk_idx, chunk in enumerate(chunks):
                    # Create unique ID for each chunk
                    content_hash = hashlib.md5((doc.url + str(chunk_idx) + chunk[:100]).encode()).hexdigest()
                    chunk_uuid = str(uuid.UUID(content_hash))

                    # Create DocumentContent object for the chunk
                    chunk_doc = DocumentContent(
                        id=chunk_uuid,
                        url=doc.url,
                        content=chunk,
                        title=f"{doc.title} - Chunk {chunk_idx + 1}",
                        created_at=doc.created_at,
                        chunk_index=chunk_idx
                    )
                    chunk_docs.append(chunk_doc)

                # FIX: Use the existing batch function instead of the loop
                batch_result = batch_store_embeddings(
                    embedding_vectors=chunk_embeddings,
                    document_contents=chunk_docs,
                    collection_name=args.collection_name,
                    batch_size=args.batch_size
                )

                successful_stores += batch_result["successful_items"]
                failed_stores += batch_result["failed_items"]

                if batch_result["errors"]:
                    pipeline_status.errors.extend(batch_result["errors"])

        logger.info(f"Finished processing: {successful_stores} embeddings stored successfully, {failed_stores} failed")

        # Create a simplified storage results dict
        storage_results = {
            "successful_items": successful_stores,
            "failed_items": failed_stores,
            "errors": pipeline_status.errors[-failed_stores:] if failed_stores > 0 else []
        }

        if storage_results["successful_items"] == 0:
            logger.error("No embeddings were successfully stored in Qdrant")
            pipeline_status.status = "failed"
            pipeline_status.errors.append("No embeddings were successfully stored in Qdrant")
            return
        else:
            logger.info(f"Successfully stored {storage_results['successful_items']} embeddings in Qdrant")

        # Update pipeline status to completed
        pipeline_status.status = "completed"
        pipeline_status.end_time = datetime.now()

        # Verify final collection size
        try:
            final_collection_info = qdrant_client.get_collection(args.collection_name)
            final_count = final_collection_info.points_count
            logger.info(f"Final verification: Collection '{args.collection_name}' contains {final_count} points")
        except Exception as e:
            logger.error(f"Could not verify final collection size: {e}")
            final_count = -1  # Indicate verification failed

        # Generate summary report
        duration = (pipeline_status.end_time - pipeline_status.start_time).total_seconds()
        summary = {
            "status": pipeline_status.status,
            "source_url": pipeline_status.source_url,
            "total_documents_processed": pipeline_status.processed_documents,
            "total_embeddings_generated": total_embeddings_generated,
            "total_embeddings_stored": storage_results["successful_items"],
            "final_collection_size": final_count,
            "collection_name": args.collection_name,
            "processing_duration_seconds": duration,
            "errors_count": len(pipeline_status.errors),
            "storage_results": storage_results
        }

        # Print summary report
        print("\n" + "="*60)
        print("PIPELINE EXECUTION SUMMARY")
        print("="*60)
        print(f"Status: {summary['status']}")
        print(f"Source URL: {summary['source_url']}")
        print(f"Documents Processed: {summary['total_documents_processed']}")
        print(f"Embeddings Generated: {summary['total_embeddings_generated']}")
        print(f"Embeddings Stored: {summary['total_embeddings_stored']}")
        print(f"Final Collection Size: {summary['final_collection_size']}")
        print(f"Collection Name: {summary['collection_name']}")
        print(f"Processing Duration: {summary['processing_duration_seconds']:.2f} seconds")
        print(f"Errors Encountered: {summary['errors_count']}")
        if storage_results["errors"]:
            print("Storage Errors:")
            for error in storage_results["errors"][:3]:  # Show first 3 errors
                print(f"  - {error}")
            if len(storage_results["errors"]) > 3:
                print(f"  ... and {len(storage_results['errors']) - 3} more")
        print("="*60)

        logger.info(f"Embedding pipeline completed successfully. Processed {summary['total_documents_processed']} documents, stored {summary['total_embeddings_stored']} embeddings in collection '{args.collection_name}'")

        # Generate and log performance summary
        perf_summary = get_performance_summary()
        if perf_summary:
            logger.info("PERFORMANCE METRICS SUMMARY:")
            for func_name, metrics in perf_summary.items():
                logger.info(f"  {func_name}: {metrics['call_count']} calls, "
                          f"total time: {metrics['total_time']:.4f}s, "
                          f"avg time: {metrics['avg_time']:.4f}s")

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        logger.error(f"Pipeline execution failed with error: {e}")
        logger.error(f"Traceback: {error_details}")
        pipeline_status.status = "failed"
        pipeline_status.errors.append(f"Pipeline execution failed: {str(e)}")
        # Don't raise, allow the script to complete and show summary
        # raise

    finally:
        # Final logging
        if pipeline_status.end_time is None:
            pipeline_status.end_time = datetime.now()

        duration = (pipeline_status.end_time - pipeline_status.start_time).total_seconds()
        logger.info(f"Pipeline execution completed in {duration:.2f} seconds with status: {pipeline_status.status}")

        # Generate and log performance summary in finally block as well
        perf_summary = get_performance_summary()
        if perf_summary:
            print("\nPERFORMANCE METRICS SUMMARY:")
            print("-" * 50)
            for func_name, metrics in perf_summary.items():
                print(f"{func_name:30} | Calls: {metrics['call_count']:3} | "
                      f"Total: {metrics['total_time']:.4f}s | "
                      f"Avg: {metrics['avg_time']:.4f}s")
            print("-" * 50)


if __name__ == "__main__":
    main()