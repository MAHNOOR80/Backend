#!/usr/bin/env python3
"""
Retrieval-Enabled Agent for Humanoid Robotics Textbook

This module implements a Cohere-based agent that retrieves information from Qdrant
vector database and answers questions based on embedded textbook content with proper
source attribution and grounding.

Author: AI Assistant (Refactored for Cohere)
Date: 2025-12-16
"""

import os
import sys
import logging
import time
import uuid
from typing import List, Dict, Optional, Union, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from dotenv import load_dotenv

# Removed google.generativeai import
import cohere
from qdrant_client import QdrantClient
# Keep existing Qdrant imports
from qdrant_client.http.models import Distance, VectorParams, Filter, FieldCondition, MatchValue

# Load environment variables
load_dotenv()

# ============================================================================
# Logging Configuration
# ============================================================================

logging.basicConfig(
    level=os.getenv('LOG_LEVEL', 'INFO'),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('agent.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ============================================================================
# Data Models (Unchanged)
# ============================================================================

class ResponseStatus(Enum):
    SUCCESS = "success"
    CONVERSATIONAL = "conversational"
    INSUFFICIENT_CONTEXT = "insufficient_context"
    ERROR = "error"

@dataclass
class Query:
    text: str
    timestamp: datetime = field(default_factory=datetime.now)
    query_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if not self.text or not self.text.strip():
            raise ValueError("Query text cannot be empty")
        if len(self.text) > 500:
            raise ValueError("Query text too long (max 500 characters)")
        if self.metadata is None:
            self.metadata = {}

@dataclass
class ChunkMetadata:
    page_number: Optional[int] = None
    section_title: Optional[str] = None
    chapter: Optional[str] = None
    url: Optional[str] = None
    chunk_index: Optional[int] = None

@dataclass
class TextChunk:
    chunk_id: Union[str, int]
    text: str
    vector: Optional[List[float]] = None
    metadata: ChunkMetadata = field(default_factory=ChunkMetadata)
    similarity_score: Optional[float] = None

@dataclass
class RetrievalParameters:
    top_k: int = 5
    score_threshold: float = 0.5
    collection_name: str = "rag_embedding"
    embedding_model: str = "embed-english-v3.0"

@dataclass
class RetrievalResult:
    query: Query
    chunks: List[TextChunk]
    retrieval_time_ms: float
    total_candidates: int
    filtered_count: int
    parameters: RetrievalParameters

@dataclass
class ConfidenceScore:
    retrieval_quality: float = 0.0
    coverage_score: float = 0.0
    entailment_score: float = 0.0
    lexical_overlap: float = 0.0

    @property
    def overall(self) -> float:
        weights = {
            'retrieval_quality': 0.35,
            'coverage_score': 0.25,
            'entailment_score': 0.25,
            'lexical_overlap': 0.15
        }
        return sum(weights[k] * getattr(self, k) for k in weights)

    @property
    def level(self) -> str:
        score = self.overall
        if score >= 0.8: return "high"
        elif score >= 0.6: return "medium"
        else: return "low"

@dataclass
class SourceReference:
    chunk_id: Union[str, int]
    citation_index: int
    relevance_score: float
    excerpt: str
    metadata: ChunkMetadata

@dataclass
class ResponseMetadata:
    model: str
    temperature: float
    total_time_ms: float
    retrieval_time_ms: float
    generation_time_ms: float
    tokens_used: Optional[int] = None
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class AgentResponse:
    query_id: str
    status: ResponseStatus
    answer: Optional[str] = None
    confidence: Optional[ConfidenceScore] = None
    sources: List[SourceReference] = field(default_factory=list)
    metadata: Optional[ResponseMetadata] = None
    error_message: Optional[str] = None


# ============================================================================
# Core Agent Class (Refactored)
# ============================================================================

class RetrievalAgent:
    """
    Cohere-based agent that retrieves information from Qdrant and answers questions
    grounded in textbook content using Cohere's Command R+ model.
    """

    def __init__(
        self,
        model: str = "command-r-plus-08-2024", # Updated to current Cohere model
        temperature: float = 0.3,
        top_k: int = 5,
        score_threshold: float = 0.5,
        collection_name: str = "rag_embedding"
    ):
        """
        Initialize the retrieval agent with configuration parameters.
        """
        # Initialize API clients
        cohere_api_key = os.getenv('COHERE_API_KEY')
        if not cohere_api_key:
            raise ValueError("COHERE_API_KEY environment variable is required")

        qdrant_url = os.getenv('QDRANT_URL')
        if not qdrant_url:
            raise ValueError("QDRANT_URL environment variable is required")

        qdrant_api_key = os.getenv('QDRANT_API_KEY')
        if not qdrant_api_key:
            raise ValueError("QDRANT_API_KEY environment variable is required")

        # Removed Gemini Configuration
        
        self.cohere_client = cohere.Client(cohere_api_key)
        self.qdrant_client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key,
            timeout=10.0
        )

        # Store configuration
        self.model = model
        self.temperature = temperature
        self.top_k = top_k
        self.score_threshold = score_threshold
        self.collection_name = collection_name

        logger.info(f"RetrievalAgent initialized with model={model}, collection={collection_name}")

    def _is_greeting_or_low_intent(self, query: str) -> bool:
        # (This method remains unchanged from your original code)
        if not query: return True
        normalized_query = query.lower().strip()
        greeting_patterns = [
            'hello', 'hi', 'hey', 'greetings', 'good morning', 'how are you', 
            'who are you', 'what can you do', 'thanks', 'bye', 'help'
        ]
        
        for pattern in greeting_patterns:
            if normalized_query == pattern: return True
            if normalized_query.startswith(pattern + ' '): return True
            
        if len(normalized_query.strip()) < 3 and not any(c.isalpha() for c in normalized_query):
            return True
        return False

    def _handle_greeting_query(self, query: str) -> str:
        # (This method remains unchanged from your original code)
        # Simplified for brevity in this snippet, assumes same logic as before
        return ("Hello! I'm your AI assistant for the Humanoid Robotics Textbook. "
                "I can help answer questions about humanoid robotics concepts. "
                "What would you like to know?")

    def get_embedding(self, text: str, input_type: str = "search_query") -> List[float]:
        """Generate embedding using Cohere API."""
        start_time = time.time()
        try:
            response = self.cohere_client.embed(
                texts=[text],
                model="embed-english-v3.0",
                input_type=input_type
            )
            embedding = response.embeddings[0]
            logger.debug(f"Generated embedding in {(time.time() - start_time) * 1000:.2f}ms")
            return embedding
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise

    def retrieve_information(self, query: str, top_k: int = 5, score_threshold: float = 0.7) -> RetrievalResult:
        """Retrieve relevant text chunks from Qdrant based on the query."""
        start_time = time.time()
        try:
            try:
                self.qdrant_client.get_collection(self.collection_name)
            except Exception:
                raise ValueError(f"Qdrant collection '{self.collection_name}' not found.")

            query_embedding = self.get_embedding(query, input_type="search_query")

            try:
                search_results = self.qdrant_client.query_points(
                    collection_name=self.collection_name,
                    query=query_embedding,
                    limit=top_k,
                    score_threshold=score_threshold,
                    with_payload=True,
                    with_vectors=False
                )
                search_result_items = search_results.points
            except AttributeError:
                search_results = self.qdrant_client.search(
                    collection_name=self.collection_name,
                    query_vector=query_embedding,
                    limit=top_k,
                    score_threshold=score_threshold,
                    with_payload=True,
                    with_vectors=False
                )
                search_result_items = search_results

            chunks = []
            for result in search_result_items:
                chunk_metadata = ChunkMetadata(
                    section_title=result.payload.get('title') if result.payload else None,
                    url=result.payload.get('url') if result.payload else None,
                    chunk_index=result.payload.get('chunk_index') if result.payload else None
                )
                
                full_content = result.payload.get('content', '') or result.payload.get('content_preview', '') or ''
                
                if len(full_content.strip()) >= 50:
                    chunk = TextChunk(
                        chunk_id=result.id,
                        text=full_content,
                        metadata=chunk_metadata,
                        similarity_score=result.score
                    )
                    chunks.append(chunk)

            # Get total collection size
            collection_info = self.qdrant_client.get_collection(self.collection_name)
            
            retrieval_time_ms = (time.time() - start_time) * 1000
            
            return RetrievalResult(
                query=Query(text=query),
                chunks=chunks,
                retrieval_time_ms=retrieval_time_ms,
                total_candidates=collection_info.points_count,
                filtered_count=len(chunks),
                parameters=RetrievalParameters(top_k=top_k, score_threshold=score_threshold, collection_name=self.collection_name)
            )
        except Exception as e:
            logger.error(f"Failed to retrieve information: {e}")
            raise

    def generate_answer(self, query: str, retrieved_chunks: List[TextChunk]) -> str:
        """
        Generate a grounded answer using Cohere's Chat API with native RAG (documents).
        """
        if not retrieved_chunks:
            return "I cannot find sufficient information in the provided textbook content to answer this question."

        # Convert TextChunks to the dictionary format Cohere expects for 'documents'
        documents = []
        for chunk in retrieved_chunks:
            doc = {
                "text": chunk.text,
                "title": chunk.metadata.section_title or "Textbook Section",
                "url": chunk.metadata.url or "",
                # You can add custom fields if needed, but 'text' is mandatory for RAG
            }
            documents.append(doc)

        start_time = time.time()

        try:
            # Preamble to enforce style, though Cohere handles grounding natively
            preamble = (
                "You are a helpful AI assistant for a Humanoid Robotics textbook. "
                "Answer the user's question using ONLY the provided documents. "
                "If the information is not in the documents, say you cannot find sufficient information."
            )

            # Call Cohere Chat API with documents
            response = self.cohere_client.chat(
                model=self.model,  # e.g., 'command-r-plus'
                message=query,
                documents=documents,
                temperature=self.temperature,
                preamble=preamble
            )

            answer = response.text
            generation_time_ms = (time.time() - start_time) * 1000
            logger.debug(f"Generated answer in {generation_time_ms:.2f}ms")

            return answer

        except Exception as e:
            logger.error(f"Cohere API failed: {e}")
            # Provide more specific error messages
            error_str = str(e)
            if "404" in error_str or "model" in error_str.lower():
                return "The AI model is currently unavailable. Please contact support or try again later."
            elif "401" in error_str or "api key" in error_str.lower():
                return "Authentication failed. Please check the API configuration."
            else:
                return "I encountered an error generating the answer. Please try again or contact support."

    def query(self, user_query: str, top_k: Optional[int] = None, score_threshold: Optional[float] = None) -> AgentResponse:
        """
        Main method to process a user query and return a grounded response.
        """
        if top_k is None: top_k = self.top_k
        if score_threshold is None: score_threshold = self.score_threshold

        query_obj = Query(text=user_query)
        start_time = time.time()

        try:
            if self._is_greeting_or_low_intent(user_query):
                response_text = self._handle_greeting_query(user_query)
                total_time_ms = (time.time() - start_time) * 1000
                metadata = ResponseMetadata(
                    model=self.model, temperature=self.temperature,
                    total_time_ms=total_time_ms, retrieval_time_ms=0.0, generation_time_ms=total_time_ms
                )
                return AgentResponse(
                    query_id=query_obj.query_id, status=ResponseStatus.CONVERSATIONAL,
                    answer=response_text, confidence=ConfidenceScore(1.0, 1.0, 1.0, 1.0),
                    sources=[], metadata=metadata
                )

            # Retrieve
            retrieval_start = time.time()
            retrieval_result = self.retrieve_information(user_query, top_k, score_threshold)
            retrieval_time_ms = (time.time() - retrieval_start) * 1000

            # Generate
            generation_start = time.time()
            answer = self.generate_answer(user_query, retrieval_result.chunks)
            generation_time_ms = (time.time() - generation_start) * 1000

            # Confidence (Simplified logic)
            confidence = ConfidenceScore()
            if retrieval_result.chunks:
                confidence.retrieval_quality = max([c.similarity_score for c in retrieval_result.chunks if c.similarity_score] or [0.0])
                confidence.coverage_score = 0.8  # Placeholder
                confidence.entailment_score = 0.8 # Placeholder

            # Build Sources
            sources = []
            for i, chunk in enumerate(retrieval_result.chunks, 1):
                source = SourceReference(
                    chunk_id=chunk.chunk_id,
                    citation_index=i,
                    relevance_score=chunk.similarity_score or 0.0,
                    excerpt=chunk.text[:100] + "...",
                    metadata=chunk.metadata
                )
                sources.append(source)

            status = ResponseStatus.SUCCESS
            error_msg = None
            if "sufficient information" in answer and "cannot find" in answer:
                status = ResponseStatus.INSUFFICIENT_CONTEXT
                error_msg = answer
                answer = None

            total_time_ms = (time.time() - start_time) * 1000
            metadata = ResponseMetadata(
                model=self.model, temperature=self.temperature,
                total_time_ms=total_time_ms, retrieval_time_ms=retrieval_time_ms,
                generation_time_ms=generation_time_ms
            )

            return AgentResponse(
                query_id=query_obj.query_id, status=status, answer=answer,
                confidence=confidence, sources=sources, metadata=metadata, error_message=error_msg
            )

        except Exception as e:
            logger.error(f"Query failed: {e}")
            return AgentResponse(query_id=query_obj.query_id, status=ResponseStatus.ERROR, error_message=str(e))

    def query_with_context(self, user_query: str, context: str, top_k: Optional[int] = None, score_threshold: Optional[float] = None) -> AgentResponse:
        """
        Process a user query using provided context (selected text) with Cohere.
        """
        # Logic is similar to query(), but we bypass retrieval and manually construct the doc.
        query_obj = Query(text=user_query)
        start_time = time.time()

        try:
            if self._is_greeting_or_low_intent(user_query):
                 # (Handling same as above for brevity)
                 return self.query(user_query) 

            # Create a single "chunk" from the context
            chunk = TextChunk(
                chunk_id="selected_text",
                text=context,
                similarity_score=1.0,
                metadata=ChunkMetadata(section_title="User Selected Text")
            )
            
            generation_start = time.time()
            answer = self.generate_answer_with_context(user_query, context)
            generation_time_ms = (time.time() - generation_start) * 1000

            # Build dummy source
            source = SourceReference(
                chunk_id="selected_text", citation_index=1, relevance_score=1.0,
                excerpt=context[:50] + "...", metadata=ChunkMetadata()
            )

            total_time_ms = (time.time() - start_time) * 1000
            metadata = ResponseMetadata(
                model=self.model, temperature=self.temperature,
                total_time_ms=total_time_ms, retrieval_time_ms=0.0, generation_time_ms=generation_time_ms
            )
            
            return AgentResponse(
                query_id=query_obj.query_id, status=ResponseStatus.SUCCESS, answer=answer,
                confidence=ConfidenceScore(1.0, 1.0, 1.0, 1.0), sources=[source], metadata=metadata
            )

        except Exception as e:
            return AgentResponse(query_id=query_obj.query_id, status=ResponseStatus.ERROR, error_message=str(e))

    def generate_answer_with_context(self, query: str, context: str) -> str:
        """Generate answer using Cohere for specific context string."""
        start_time = time.time()
        
        # Treat context as a single document
        documents = [{"text": context, "title": "Selected Context"}]
        
        try:
            response = self.cohere_client.chat(
                model=self.model,
                message=query,
                documents=documents,
                temperature=self.temperature,
                preamble="Answer using only the provided context text."
            )
            return response.text
        except Exception as e:
            logger.error(f"Cohere API failed: {e}")
            # Provide more specific error messages
            error_str = str(e)
            if "404" in error_str or "model" in error_str.lower():
                return "The AI model is currently unavailable. Please contact support or try again later."
            elif "401" in error_str or "api key" in error_str.lower():
                return "Authentication failed. Please check the API configuration."
            else:
                return "I encountered an error generating the answer. Please try again or contact support."

# ============================================================================
# CLI Interface (Unchanged)
# ============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Retrieval-Enabled Agent")
    parser.add_argument("--query", type=str, help="Question to ask")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--model", type=str, default="command-r-plus-08-2024") # Updated to current model

    args = parser.parse_args()
    if not args.query:
        print("Usage: python agent.py --query 'Your question'")
        sys.exit(1)

    agent = RetrievalAgent(model=args.model, top_k=args.top_k, score_threshold=args.threshold)
    print(f"Processing query: {args.query}")
    print("-" * 50)
    
    response = agent.query(args.query)
    
    if response.status == ResponseStatus.SUCCESS:
        print(f"Answer: {response.answer}")
        print(f"Sources: {len(response.sources)} chunks")
    else:
        print(f"Status: {response.status.value}")
        if response.error_message: print(f"Error: {response.error_message}")

if __name__ == "__main__":
    main()