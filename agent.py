#!/usr/bin/env python3
"""
Retrieval-Enabled Agent for Humanoid Robotics Textbook

This module implements an OpenAI-based agent that retrieves information from Qdrant
vector database and answers questions based on embedded textbook content with proper
source attribution and grounding.

The agent uses OpenAI Chat Completions API with function calling to:
1. Retrieve relevant text chunks from Qdrant using semantic search
2. Generate grounded answers based only on retrieved content
3. Provide source citations and confidence scores
4. Handle insufficient context scenarios appropriately

Author: AI Assistant
Date: 2025-12-13
"""

import os
import sys
import logging
import time
import uuid
import json
from typing import List, Dict, Optional, Union, Any
from dataclasses import dataclass, asdict, field
from enum import Enum
from datetime import datetime
from dotenv import load_dotenv

import google.generativeai as genai 
import cohere
from qdrant_client import QdrantClient
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
# Data Models
# ============================================================================

class ResponseStatus(Enum):
    """Response status for agent responses"""
    SUCCESS = "success"
    CONVERSATIONAL = "conversational"
    INSUFFICIENT_CONTEXT = "insufficient_context"
    ERROR = "error"


@dataclass
class Query:
    """
    Represents a user's natural language question about humanoid robotics topics.

    Attributes:
        text: The natural language query text
        timestamp: When the query was submitted
        query_id: Unique identifier for the query
        metadata: Optional metadata (user_id, session_id, etc.)
    """
    text: str
    timestamp: datetime = field(default_factory=datetime.now)
    query_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Validate query parameters"""
        if not self.text or not self.text.strip():
            raise ValueError("Query text cannot be empty")
        if len(self.text) > 500:
            raise ValueError("Query text too long (max 500 characters)")
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ChunkMetadata:
    """
    Metadata for text chunks from the textbook.

    Attributes:
        page_number: Page number in original textbook
        section_title: Section or chapter title
        chapter: Chapter name
        url: Source URL from the textbook website
        chunk_index: Sequential index of chunk in document
    """
    page_number: Optional[int] = None
    section_title: Optional[str] = None
    chapter: Optional[str] = None
    url: Optional[str] = None
    chunk_index: Optional[int] = None


@dataclass
class TextChunk:
    """
    Represents a segment of the embedded textbook content stored in Qdrant vector database.

    Attributes:
        chunk_id: Unique identifier from Qdrant
        text: The actual text content of the chunk
        vector: 1024-dim Cohere embedding (optional in retrieval)
        metadata: Source metadata from the textbook
        similarity_score: Cosine similarity score (when retrieved)
    """
    chunk_id: Union[str, int]
    text: str
    vector: Optional[List[float]] = None
    metadata: ChunkMetadata = field(default_factory=ChunkMetadata)
    similarity_score: Optional[float] = None

    def __post_init__(self):
        """Validate text chunk parameters"""
        if not self.text or not self.text.strip():
            raise ValueError("Text chunk content cannot be empty")
        if self.similarity_score is not None and (self.similarity_score < 0.0 or self.similarity_score > 1.0):
            raise ValueError(f"Similarity score must be between 0.0 and 1.0, got {self.similarity_score}")


@dataclass
class RetrievalParameters:
    """
    Parameters for the retrieval function.

    Attributes:
        top_k: Number of chunks to retrieve
        score_threshold: Minimum similarity score threshold
        collection_name: Qdrant collection name
        embedding_model: Cohere model used for embeddings
    """
    top_k: int = 5
    score_threshold: float = 0.5
    collection_name: str = "rag_embedding"
    embedding_model: str = "embed-english-v3.0"

    def __post_init__(self):
        """Validate retrieval parameters"""
        if self.top_k < 1 or self.top_k > 20:
            raise ValueError(f"top_k must be between 1 and 20, got {self.top_k}")
        if self.score_threshold < 0.0 or self.score_threshold > 1.0:
            raise ValueError(f"score_threshold must be between 0.0 and 1.0, got {self.score_threshold}")


@dataclass
class RetrievalResult:
    """
    Represents the output from Qdrant semantic search operation.

    Attributes:
        query: Original query that triggered retrieval
        chunks: Retrieved chunks ranked by similarity
        retrieval_time_ms: Time taken for retrieval operation
        total_candidates: Total chunks in collection
        filtered_count: Chunks above score_threshold
        parameters: Search parameters used
    """
    query: Query
    chunks: List[TextChunk]
    retrieval_time_ms: float
    total_candidates: int
    filtered_count: int
    parameters: RetrievalParameters


@dataclass
class ConfidenceScore:
    """
    Multi-factor confidence assessment for agent responses.

    Attributes:
        retrieval_quality: Max similarity score from retrieval (0-1)
        coverage_score: % of query topics covered (0-1)
        entailment_score: Answer-context entailment (0-1)
        lexical_overlap: Lexical overlap answer/chunks (0-1)
    """
    retrieval_quality: float = 0.0
    coverage_score: float = 0.0
    entailment_score: float = 0.0
    lexical_overlap: float = 0.0

    @property
    def overall(self) -> float:
        """Weighted average of all confidence factors."""
        weights = {
            'retrieval_quality': 0.35,
            'coverage_score': 0.25,
            'entailment_score': 0.25,
            'lexical_overlap': 0.15
        }
        return sum(weights[k] * getattr(self, k) for k in weights)

    @property
    def level(self) -> str:
        """Human-readable confidence level."""
        score = self.overall
        if score >= 0.8:
            return "high"
        elif score >= 0.6:
            return "medium"
        else:
            return "low"


@dataclass
class SourceReference:
    """
    Citation linking answer to a specific chunk.

    Attributes:
        chunk_id: References TextChunk.chunk_id
        citation_index: Position in answer (e.g., [1], [2])
        relevance_score: Similarity score from retrieval
        excerpt: Short excerpt from chunk (for verification)
        metadata: Full metadata from TextChunk
    """
    chunk_id: Union[str, int]
    citation_index: int
    relevance_score: float
    excerpt: str
    metadata: ChunkMetadata


@dataclass
class ResponseMetadata:
    """
    Generation metadata for agent responses.

    Attributes:
        model: OpenAI model used
        temperature: Generation temperature (0.1-0.3)
        total_time_ms: End-to-end response time
        retrieval_time_ms: Time spent on Qdrant retrieval
        generation_time_ms: Time spent on LLM generation
        tokens_used: Total tokens consumed
        timestamp: Response generation timestamp
    """
    model: str
    temperature: float
    total_time_ms: float
    retrieval_time_ms: float
    generation_time_ms: float
    tokens_used: Optional[int] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class AgentResponse:
    """
    Structured output from the retrieval agent containing the grounded answer with source attribution.

    Attributes:
        query_id: Links back to original Query
        status: Success, insufficient_context, or error
        answer: The generated answer (None if status != success)
        confidence: Multi-factor confidence assessment
        sources: List of source references
        metadata: Generation metadata
        error_message: Error details if status == error
    """
    query_id: str
    status: ResponseStatus
    answer: Optional[str] = None
    confidence: Optional[ConfidenceScore] = None
    sources: List[SourceReference] = field(default_factory=list)
    metadata: Optional[ResponseMetadata] = None
    error_message: Optional[str] = None

    def __post_init__(self):
        """Validate agent response parameters"""
        if self.status == ResponseStatus.SUCCESS:
            if self.answer is None:
                raise ValueError("Answer must be provided when status is SUCCESS")
            if len(self.sources) == 0:
                raise ValueError("At least one source must be provided when status is SUCCESS")
        elif self.status == ResponseStatus.CONVERSATIONAL:
            if self.answer is None:
                raise ValueError("Answer must be provided when status is CONVERSATIONAL")
            # Conversational responses don't require sources
        elif self.status == ResponseStatus.INSUFFICIENT_CONTEXT:
            if self.answer is not None:
                raise ValueError("Answer should be None when status is INSUFFICIENT_CONTEXT")
        elif self.status == ResponseStatus.ERROR:
            if self.error_message is None:
                raise ValueError("Error message must be provided when status is ERROR")


# ============================================================================
# Core Agent Class
# ============================================================================

class RetrievalAgent:
    """
    OpenAI-based agent that retrieves information from Qdrant and answers questions
    grounded in textbook content with proper source attribution.
    """

    def __init__(
        self,
        model: str = "gpt-4-turbo-preview",
        temperature: float = 0.2,
        top_k: int = 5,
        score_threshold: float = 0.5,
        collection_name: str = "rag_embedding"
    ):
        """
        Initialize the retrieval agent with configuration parameters.

        Args:
            model: OpenAI model to use for generation
            temperature: Generation temperature (0.1-0.3 for factual responses)
            top_k: Maximum number of chunks to retrieve
            score_threshold: Minimum similarity score for retrieved chunks
            collection_name: Name of the Qdrant collection with textbook embeddings
        """
        # Initialize API clients
        google_api_key = os.getenv('OPENAI_API_KEY')  # Using the existing env var name for Google API key
        if not google_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")

        cohere_api_key = os.getenv('COHERE_API_KEY')
        if not cohere_api_key:
            raise ValueError("COHERE_API_KEY environment variable is required")

        qdrant_url = os.getenv('QDRANT_URL')
        if not qdrant_url:
            raise ValueError("QDRANT_URL environment variable is required")

        qdrant_api_key = os.getenv('QDRANT_API_KEY')
        if not qdrant_api_key:
            raise ValueError("QDRANT_API_KEY environment variable is required")

        genai.configure(api_key=google_api_key)
        self.google_model = genai.GenerativeModel('gemini-pro')  # Use gemini-pro model
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

        # Define tools for function calling
        self.tools = [{
            "type": "function",
            "function": {
                "name": "retrieve_relevant_chunks",
                "description": "Retrieve relevant text chunks from the humanoid robotics textbook knowledge base using semantic search. This function searches the Qdrant vector database for content semantically similar to the user's query and returns the most relevant passages with similarity scores.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query or question to find relevant textbook content for. This will be embedded using the same model as the stored chunks (Cohere embed-english-v3.0) and used for semantic similarity search."
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Maximum number of relevant chunks to retrieve. Defaults to 5. Higher values provide more context but may introduce noise.",
                            "default": 5,
                            "minimum": 1,
                            "maximum": 20
                        },
                        "score_threshold": {
                            "type": "number",
                            "description": "Minimum similarity score threshold (0.0 to 1.0) for retrieved chunks. Only chunks with cosine similarity above this threshold will be returned. Defaults to 0.7 (high relevance). Lower values retrieve more chunks but with potentially lower quality.",
                            "default": 0.7,
                            "minimum": 0.0,
                            "maximum": 1.0
                        }
                    },
                    "required": ["query"]
                }
            }
        }]

        logger.info(f"RetrievalAgent initialized with model={model}, collection={collection_name}")

    def _is_greeting_or_low_intent(self, query: str) -> bool:
        """
        Check if the query is a greeting or low-intent query that doesn't require RAG retrieval.

        Args:
            query: The user query to check

        Returns:
            True if the query is a greeting or low-intent query, False otherwise
        """
        if not query:
            return True

        # Normalize the query for comparison
        normalized_query = query.lower().strip()

        # Common greetings and low-intent phrases
        greeting_patterns = [
            'hello', 'hi', 'hey', 'greetings', 'good morning', 'good afternoon', 'good evening', 'good night',
            'how are you', 'how do you do', 'what\'s up', 'how\'s it going', 'how are things',
            'who are you', 'what are you', 'what is your name', 'introduce yourself', 'tell me about yourself',
            'what can you do', 'what are your capabilities', 'what are you capable of', 'help',
            'thanks', 'thank you', 'thx', 'appreciate it', 'gracias', 'merci', 'danke',
            'bye', 'goodbye', 'see you', 'farewell', 'ciao', 'adios', 'au revoir',
            'ok', 'okay', 'sure', 'yes', 'no', 'maybe', 'indeed', 'exactly', 'right',
            'good', 'great', 'nice', 'awesome', 'excellent', 'wonderful', 'fantastic',
            'please', 'please help', 'can you help', 'can you assist', 'assist me',
            '',  # Empty query
            '...', '..', '....', '.....',  # Dots only
            'um', 'uh', 'hmm', 'hmmm', 'er', 'erm',  # Hesitations
        ]

        # Check for direct matches
        for pattern in greeting_patterns:
            if normalized_query == pattern:
                return True

        # For multi-word patterns, check if query starts with the phrase followed by space
        for pattern in greeting_patterns:
            pattern_words = pattern.split()
            if len(pattern_words) > 1:  # Multi-word pattern
                if normalized_query.startswith(pattern + ' '):
                    return True

        # For single-word patterns, be more restrictive about prefix matching
        # Only match if the next word is also likely to be part of a greeting
        single_word_greetings = [p for p in greeting_patterns if len(p.split()) == 1]
        for word in single_word_greetings:
            if normalized_query.startswith(word + ' '):
                # Check if the following word is also likely a greeting continuation
                remaining = normalized_query[len(word + ' '):]
                next_word = remaining.split()[0] if remaining.split() else ""

                # Common greeting continuations (avoiding words that might appear in content queries)
                greeting_continuations = {
                    'hello', 'hi', 'hey', 'there', 'everyone', 'folks', 'team',
                    'thanks', 'please', 'sorry', 'excuse', 'good', 'morning', 'afternoon',
                    'evening', 'night', 'bye', 'goodbye', 'welcome', 'again', 'you', 'my'
                }

                # If the next word is a common greeting continuation, treat as greeting
                if next_word in greeting_continuations or next_word == "":
                    return True

        # Check for very short queries that start with greeting words (avoiding content queries like "hi robot")
        words = normalized_query.split()
        if len(words) <= 2 and words and words[0] in ['hello', 'hi', 'hey', 'help']:
            # Only consider it a greeting if it's just the greeting word or followed by common greeting words
            if len(words) == 1:
                return True  # Single greeting word
            else:  # Two words
                second_word = words[1]
                common_greeting_followers = {
                    'there', 'world', 'everyone', 'folks', 'team', 'again', 'you', 'there!', 'world!',
                    'morning', 'afternoon', 'evening', 'night', 'my', 'dear', 'good'
                }
                if second_word in common_greeting_followers:
                    return True

        # Check for queries that are just punctuation or very short
        if len(normalized_query.strip()) < 3 and not any(c.isalpha() for c in normalized_query):
            return True

        return False

    def _handle_greeting_query(self, query: str) -> str:
        """
        Generate a response for greeting or low-intent queries.

        Args:
            query: The user query (greeting or low-intent)

        Returns:
            A friendly response to the greeting or low-intent query
        """
        normalized_query = query.lower().strip()

        # Handle different types of greetings
        if any(greeting in normalized_query for greeting in ['hello', 'hi', 'hey', 'greetings']):
            return ("Hello! I'm your AI assistant for the Humanoid Robotics Textbook. "
                   "I can help answer questions about humanoid robotics concepts, theories, and applications. "
                   "What would you like to know about humanoid robotics?")

        elif any(phrase in normalized_query for phrase in ['how are you', 'how do you do', 'what\'s up', 'how\'s it going']):
            return ("I'm doing well, thank you for asking! I'm here to help you learn about humanoid robotics. "
                   "Feel free to ask me any questions about the textbook content.")

        elif any(phrase in normalized_query for phrase in ['who are you', 'what are you', 'what is your name', 'introduce yourself', 'tell me about yourself']):
            return ("I'm an AI assistant designed specifically for the Humanoid Robotics Textbook. "
                   "I can answer questions about humanoid robotics by retrieving and analyzing content from the textbook. "
                   "Ask me anything about the book!")

        elif any(phrase in normalized_query for phrase in ['what can you do', 'what are you capable of', 'what are your capabilities']):
            return ("I can help you with questions about humanoid robotics by searching the textbook content. "
                   "I can explain concepts, find specific information, and provide detailed answers grounded in the textbook. "
                   "Try asking me about specific topics like 'inverse kinematics', 'gait planning', or 'humanoid control systems'.")

        elif any(thanks in normalized_query for thanks in ['thanks', 'thank you', 'thx', 'appreciate it', 'gracias', 'merci', 'danke']):
            return ("You're welcome! I'm happy to help. If you have any questions about humanoid robotics, feel free to ask!")

        elif any(bye in normalized_query for bye in ['bye', 'goodbye', 'see you', 'farewell', 'ciao', 'adios', 'au revoir']):
            return ("Goodbye! Feel free to come back if you have any questions about humanoid robotics. Have a great day!")

        elif any(help_word in normalized_query for help_word in ['help', 'please help', 'can you help', 'can you assist', 'assist me']):
            return ("I'm here to help! You can ask me questions about humanoid robotics topics covered in the textbook. "
                   "For example, you could ask 'What is inverse kinematics?' or 'Explain gait planning in humanoid robots.' "
                   "I'll search the textbook content to provide accurate answers.")

        else:
            # Default response for other low-intent queries
            return ("I'm here to help with questions about humanoid robotics from the textbook. "
                   "Could you ask me something specific about the content? For example: "
                   "'What are the main challenges in humanoid locomotion?' or "
                   "'Explain the control systems used in humanoid robots.'")

    def get_embedding(self, text: str, input_type: str = "search_query") -> List[float]:
        """
        Generate embedding for the given text using Cohere API.

        Args:
            text: Text to embed
            input_type: Type of input ("search_query" for queries, "search_document" for storage)

        Returns:
            List of float representing the embedding vector
        """
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
        """
        Retrieve relevant text chunks from Qdrant based on the query.

        Args:
            query: Natural language query to search for
            top_k: Maximum number of chunks to retrieve
            score_threshold: Minimum similarity score threshold

        Returns:
            RetrievalResult containing ranked text chunks with metadata
        """
        start_time = time.time()

        try:
            # Check if collection exists before attempting search
            try:
                self.qdrant_client.get_collection(self.collection_name)
            except Exception:
                raise ValueError(f"Qdrant collection '{self.collection_name}' not found. Please create it first.")

            # Generate embedding for the query
            query_embedding = self.get_embedding(query, input_type="search_query")

            # Try the newer query_points method first (newer Qdrant versions)
            try:
                search_results = self.qdrant_client.query_points(
                    collection_name=self.collection_name,
                    query=query_embedding,
                    limit=top_k,
                    score_threshold=score_threshold,
                    with_payload=True,
                    with_vectors=False
                )
                # For query_points, results are in search_results.points
                search_result_items = search_results.points
            except AttributeError:
                # Fall back to search method (older Qdrant versions)
                search_results = self.qdrant_client.search(
                    collection_name=self.collection_name,
                    query_vector=query_embedding,
                    limit=top_k,
                    score_threshold=score_threshold,
                    with_payload=True,
                    with_vectors=False
                )
                search_result_items = search_results

            # Convert results to TextChunk objects
            chunks = []
            for result in search_result_items:
                chunk_metadata = ChunkMetadata(
                    page_number=None,  # Not available in this collection
                    section_title=result.payload.get('title') if result.payload else None,
                    chapter=None,  # Not available in this collection
                    url=result.payload.get('url') if result.payload else None,
                    chunk_index=result.payload.get('chunk_index') if result.payload else None
                )

                # Get the full content from the 'content' field, with fallback to content_preview if needed
                full_content = result.payload.get('content', '') if result.payload else ''
                if not full_content:  # Fallback to content_preview if content is not available
                    full_content = result.payload.get('content_preview', '') if result.payload else ''

                # Skip chunks that are too short (less than 50 characters)
                if len(full_content.strip()) >= 50:
                    chunk = TextChunk(
                        chunk_id=result.id,
                        text=full_content,
                        metadata=chunk_metadata,
                        similarity_score=result.score
                    )
                    chunks.append(chunk)

            # Get total collection size for metadata
            collection_info = self.qdrant_client.get_collection(self.collection_name)
            total_candidates = collection_info.points_count

            retrieval_time_ms = (time.time() - start_time) * 1000

            retrieval_result = RetrievalResult(
                query=Query(text=query),
                chunks=chunks,
                retrieval_time_ms=retrieval_time_ms,
                total_candidates=total_candidates,
                filtered_count=len(chunks),
                parameters=RetrievalParameters(
                    top_k=top_k,
                    score_threshold=score_threshold,
                    collection_name=self.collection_name
                )
            )

            logger.info(f"Retrieved {len(chunks)} chunks in {retrieval_time_ms:.2f}ms")
            return retrieval_result

        except Exception as e:
            logger.error(f"Failed to retrieve information: {e}")
            raise

    def generate_answer(self, query: str, retrieved_chunks: List[TextChunk]) -> str:
        """
        Generate a grounded answer based on the query and retrieved chunks.

        Args:
            query: Original user query
            retrieved_chunks: List of relevant text chunks from Qdrant

        Returns:
            Generated answer string with source citations
        """
        if not retrieved_chunks:
            return "I cannot find sufficient information in the provided textbook content to answer this question. The most relevant chunks found had similarity scores below the 0.7 threshold."

        # Build context from retrieved chunks with citations
        context_parts = []
        for i, chunk in enumerate(retrieved_chunks, 1):
            context_parts.append(f"Source [{i}]: {chunk.text}")

        context = "\n\n".join(context_parts)

        # Create system prompt that enforces grounding
        system_prompt = f"""You are a retrieval-based question answering assistant for a humanoid robotics textbook.

CRITICAL RULES:
1. Answer ONLY using information from the provided context chunks below
2. DO NOT use your general knowledge or training data
3. If the context does not contain enough information to answer the question, respond: "I cannot find sufficient information in the provided textbook content to answer this question."
4. Never make assumptions or inferences beyond what is explicitly stated in the context
5. Include source citations in your answer using bracket notation [1], [2], etc. to reference the provided sources
6. Be concise but comprehensive in your answers

CONTEXT CHUNKS:
{context}

USER QUESTION:
{query}"""

        start_time = time.time()

        try:
            # For Google's Gemini, we need to adapt the prompt format
            # Combine system prompt and user query
            full_prompt = f"{system_prompt}\n\nPlease answer the user's question based on the provided context."

            response = self.google_model.generate_content(
                full_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=self.temperature,
                    max_output_tokens=1000
                )
            )

            answer = response.text if response.text else "I couldn't generate a response based on the provided context."
            # Google API doesn't provide token count in the same way as OpenAI
            tokens_used = None
            generation_time_ms = (time.time() - start_time) * 1000

            logger.debug(f"Generated answer in {generation_time_ms:.2f}ms")

            return answer
        except Exception as e:
            # If Google API fails, return a summary of the retrieved chunks as a fallback
            logger.warning(f"Google API failed: {e}. Returning retrieved content as fallback.")

            # Create a simple summary from the retrieved chunks
            if retrieved_chunks:
                fallback_answer = "Based on the retrieved content:\n\n"
                for i, chunk in enumerate(retrieved_chunks, 1):
                    fallback_answer += f"Source [{i}]: {chunk.text[:500]}...\n\n"
                fallback_answer += f"\nTotal sources: {len(retrieved_chunks)}"
            else:
                fallback_answer = "No relevant content found in the textbook to answer this question."

            generation_time_ms = (time.time() - start_time) * 1000
            logger.debug(f"Generated fallback answer in {generation_time_ms:.2f}ms")

            return fallback_answer

    def query(self, user_query: str, top_k: Optional[int] = None, score_threshold: Optional[float] = None) -> AgentResponse:
        """
        Main method to process a user query and return a grounded response.

        Args:
            user_query: Natural language question from user
            top_k: Number of chunks to retrieve (uses default if None)
            score_threshold: Minimum similarity threshold (uses default if None)

        Returns:
            AgentResponse with answer, sources, confidence, and metadata
        """
        if top_k is None:
            top_k = self.top_k
        if score_threshold is None:
            score_threshold = self.score_threshold

        query_obj = Query(text=user_query)
        start_time = time.time()

        try:
            # Check if the query is a greeting or low-intent query first
            if self._is_greeting_or_low_intent(user_query):
                # Handle greetings and low-intent queries directly
                response = self._handle_greeting_query(user_query)

                # Create response metadata
                total_time_ms = (time.time() - start_time) * 1000
                metadata = ResponseMetadata(
                    model=self.model,
                    temperature=self.temperature,
                    total_time_ms=total_time_ms,
                    retrieval_time_ms=0.0,  # No retrieval needed
                    generation_time_ms=total_time_ms
                )

                agent_response = AgentResponse(
                    query_id=query_obj.query_id,
                    status=ResponseStatus.CONVERSATIONAL,
                    answer=response,
                    confidence=ConfidenceScore(
                        retrieval_quality=1.0,  # High confidence for direct response
                        coverage_score=1.0,
                        entailment_score=1.0,
                        lexical_overlap=1.0
                    ),
                    sources=[],  # No sources needed for greetings
                    metadata=metadata
                )

                logger.info(f"Greeting query completed in {total_time_ms:.2f}ms")
                return agent_response

            # Retrieve relevant chunks for content-related queries
            retrieval_start = time.time()
            retrieval_result = self.retrieve_information(user_query, top_k, score_threshold)
            retrieval_time_ms = (time.time() - retrieval_start) * 1000

            # Generate answer based on retrieved chunks
            generation_start = time.time()
            answer = self.generate_answer(user_query, retrieval_result.chunks)
            generation_time_ms = (time.time() - generation_start) * 1000

            # Calculate confidence scores
            confidence = ConfidenceScore()
            if retrieval_result.chunks:
                confidence.retrieval_quality = max([chunk.similarity_score for chunk in retrieval_result.chunks if chunk.similarity_score])
                # Simplified coverage and entailment scores for now
                confidence.coverage_score = min(1.0, len(retrieval_result.chunks) / 5.0)  # Scale based on number of chunks
                confidence.entailment_score = confidence.retrieval_quality  # Use similarity as proxy
                confidence.lexical_overlap = 0.7  # Placeholder value

            # Create source references with citation indices
            sources = []
            for i, chunk in enumerate(retrieval_result.chunks, 1):
                source = SourceReference(
                    chunk_id=chunk.chunk_id,
                    citation_index=i,
                    relevance_score=chunk.similarity_score or 0.0,
                    excerpt=chunk.text[:100] + "..." if len(chunk.text) > 100 else chunk.text,
                    metadata=chunk.metadata
                )
                sources.append(source)

            # Determine response status
            status = ResponseStatus.SUCCESS
            error_message = None

            if "I cannot find sufficient information" in answer:
                status = ResponseStatus.INSUFFICIENT_CONTEXT
                error_message = answer
                answer = None

            # Create response metadata
            total_time_ms = (time.time() - start_time) * 1000
            metadata = ResponseMetadata(
                model=self.model,
                temperature=self.temperature,
                total_time_ms=total_time_ms,
                retrieval_time_ms=retrieval_time_ms,
                generation_time_ms=generation_time_ms
            )

            response = AgentResponse(
                query_id=query_obj.query_id,
                status=status,
                answer=answer,
                confidence=confidence,
                sources=sources,
                metadata=metadata,
                error_message=error_message
            )

            logger.info(f"Query completed in {total_time_ms:.2f}ms, status: {status.value}")
            return response

        except Exception as e:
            logger.error(f"Query failed: {e}")

            error_response = AgentResponse(
                query_id=query_obj.query_id,
                status=ResponseStatus.ERROR,
                error_message=str(e)
            )

            return error_response

    def query_with_context(self, user_query: str, context: str, top_k: Optional[int] = None, score_threshold: Optional[float] = None) -> AgentResponse:
        """
        Process a user query using provided context instead of retrieving from Qdrant.
        Used for selected text mode where the user has highlighted specific content.

        Args:
            user_query: Natural language question from user
            context: Provided context to answer from (selected text)
            top_k: Number of chunks to retrieve (not used in this method)
            score_threshold: Minimum similarity threshold (not used in this method)

        Returns:
            AgentResponse with answer, sources, confidence, and metadata
        """
        if top_k is None:
            top_k = self.top_k
        if score_threshold is None:
            score_threshold = self.score_threshold

        query_obj = Query(text=user_query)
        start_time = time.time()

        try:
            # Check if the query is a greeting or low-intent query first
            if self._is_greeting_or_low_intent(user_query):
                # Handle greetings and low-intent queries directly
                response = self._handle_greeting_query(user_query)

                # Create response metadata
                total_time_ms = (time.time() - start_time) * 1000
                metadata = ResponseMetadata(
                    model=self.model,
                    temperature=self.temperature,
                    total_time_ms=total_time_ms,
                    retrieval_time_ms=0.0,  # No retrieval needed
                    generation_time_ms=total_time_ms
                )

                agent_response = AgentResponse(
                    query_id=query_obj.query_id,
                    status=ResponseStatus.CONVERSATIONAL,
                    answer=response,
                    confidence=ConfidenceScore(
                        retrieval_quality=1.0,  # High confidence for direct response
                        coverage_score=1.0,
                        entailment_score=1.0,
                        lexical_overlap=1.0
                    ),
                    sources=[],  # No sources needed for greetings
                    metadata=metadata
                )

                logger.info(f"Greeting query with context completed in {total_time_ms:.2f}ms")
                return agent_response

            # Create a single TextChunk from the provided context
            # This simulates what would be returned from retrieval
            chunk_metadata = ChunkMetadata()
            text_chunk = TextChunk(
                chunk_id="selected_text_1",
                text=context,
                metadata=chunk_metadata,
                similarity_score=1.0  # Perfect match since user provided this text
            )
            retrieved_chunks = [text_chunk]

            # Generate answer based on the provided context
            generation_start = time.time()
            answer = self.generate_answer_with_context(user_query, context)
            generation_time_ms = (time.time() - generation_start) * 1000

            # Calculate confidence scores based on the provided context
            confidence = ConfidenceScore()
            confidence.retrieval_quality = 1.0  # We have the exact context
            confidence.coverage_score = 0.9  # High confidence as context is provided
            confidence.entailment_score = 0.9  # High confidence as context is provided
            confidence.lexical_overlap = 0.8  # High confidence as context is provided

            # Create source reference for the selected text
            source = SourceReference(
                chunk_id=text_chunk.chunk_id,
                citation_index=1,
                relevance_score=text_chunk.similarity_score or 0.0,
                excerpt=context[:100] + "..." if len(context) > 100 else context,
                metadata=text_chunk.metadata
            )
            sources = [source]

            # Determine response status
            status = ResponseStatus.SUCCESS
            error_message = None

            if "I cannot find sufficient information" in answer:
                status = ResponseStatus.INSUFFICIENT_CONTEXT
                error_message = answer
                answer = None

            # Create response metadata
            total_time_ms = (time.time() - start_time) * 1000
            metadata = ResponseMetadata(
                model=self.model,
                temperature=self.temperature,
                total_time_ms=total_time_ms,
                retrieval_time_ms=0.0,  # No retrieval needed
                generation_time_ms=generation_time_ms
            )

            response = AgentResponse(
                query_id=query_obj.query_id,
                status=status,
                answer=answer,
                confidence=confidence,
                sources=sources,
                metadata=metadata,
                error_message=error_message
            )

            logger.info(f"Query with context completed in {total_time_ms:.2f}ms, status: {status.value}")
            return response

        except Exception as e:
            logger.error(f"Query with context failed: {e}")

            error_response = AgentResponse(
                query_id=query_obj.query_id,
                status=ResponseStatus.ERROR,
                error_message=str(e)
            )

            return error_response

    def generate_answer_with_context(self, query: str, context: str) -> str:
        """
        Generate a grounded answer based on the query and provided context.

        Args:
            query: Original user query
            context: Provided context to answer from (selected text)

        Returns:
            Generated answer string with source citations
        """
        # Create system prompt that enforces using only the provided context
        system_prompt = f"""You are a retrieval-based question answering assistant for a humanoid robotics textbook.

CRITICAL RULES:
1. Answer ONLY using information from the provided context below
2. DO NOT use your general knowledge or training data
3. If the provided context does not contain enough information to answer the question, respond: "I cannot find sufficient information in the selected text to answer this question."
4. Never make assumptions or inferences beyond what is explicitly stated in the provided context
5. Include source citations in your answer using bracket notation [1] to reference the provided source
6. Be concise but comprehensive in your answers

PROVIDED CONTEXT:
{context}

USER QUESTION:
{query}"""

        start_time = time.time()

        try:
            # For Google's Gemini, we need to adapt the prompt format
            # Combine system prompt and user query
            full_prompt = f"{system_prompt}\n\nPlease answer the user's question based on the provided context."

            response = self.google_model.generate_content(
                full_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=self.temperature,
                    max_output_tokens=1000
                )
            )

            answer = response.text if response.text else "I couldn't generate a response based on the provided context."
            generation_time_ms = (time.time() - start_time) * 1000

            logger.debug(f"Generated answer with context in {generation_time_ms:.2f}ms")

            return answer
        except Exception as e:
            # If Google API fails, return a summary of the provided context as a fallback
            logger.warning(f"Google API failed: {e}. Returning provided context as fallback.")

            # Create a simple response from the provided context
            fallback_answer = f"Based on the provided text: {context[:500]}... [1]"
            generation_time_ms = (time.time() - start_time) * 1000
            logger.debug(f"Generated fallback answer in {generation_time_ms:.2f}ms")

            return fallback_answer


# ============================================================================
# CLI Interface
# ============================================================================

def main():
    """Command-line interface for the retrieval agent."""
    import argparse

    parser = argparse.ArgumentParser(description="Retrieval-Enabled Agent for Humanoid Robotics Textbook")
    parser.add_argument("--query", type=str, help="Question to ask the agent")
    parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to retrieve (default: 5)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Similarity threshold (default: 0.5)")
    parser.add_argument("--model", type=str, default="gpt-4-turbo-preview", help="OpenAI model to use")

    args = parser.parse_args()

    if not args.query:
        print("Usage: python agent.py --query 'Your question here'")
        print("Example: python agent.py --query 'What is inverse kinematics in humanoid robotics?'")
        sys.exit(1)

    try:
        # Initialize agent
        agent = RetrievalAgent(
            model=args.model,
            top_k=args.top_k,
            score_threshold=args.threshold
        )

        # Process query
        print(f"Processing query: {args.query}")
        print("-" * 50)

        response = agent.query(args.query, top_k=args.top_k, score_threshold=args.threshold)

        # Display results
        if response.status == ResponseStatus.SUCCESS:
            print(f"Answer: {response.answer}")
            print(f"Confidence: {response.confidence.level} ({response.confidence.overall:.2f})")
            print(f"Sources: {len(response.sources)} chunks")
            print(f"Response time: {response.metadata.total_time_ms:.2f}ms")

            if response.sources:
                print("\nSources:")
                for source in response.sources:
                    print(f"  [{source.citation_index}] {source.metadata.section_title or 'N/A'} "
                          f"(Score: {source.relevance_score:.2f}, Page: {source.metadata.page_number or 'N/A'})")
        elif response.status == ResponseStatus.INSUFFICIENT_CONTEXT:
            print(f"Cannot answer: {response.error_message}")
        elif response.status == ResponseStatus.ERROR:
            print(f"Error: {response.error_message}")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()