#!/usr/bin/env python3
"""
Retrieval-Enabled Agent for Humanoid Robotics Textbook

Cohere-based conversational RAG agent that retrieves textbook content from Qdrant
and explains concepts in a natural, instructor-like manner.

Author: Refactored with Conversational RAG Fixes
Date: 2025-12-19
"""

import os
import sys
import logging
import time
import uuid
from typing import List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from dotenv import load_dotenv

from openai import OpenAI
from qdrant_client import QdrantClient
from fastembed import TextEmbedding

# Load environment variables
load_dotenv()

# ============================================================================ 
# Logging Configuration
# ============================================================================

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# ============================================================================ 
# Data Models
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

@dataclass
class ChunkMetadata:
    section_title: Optional[str] = None
    url: Optional[str] = None
    chunk_index: Optional[int] = None

@dataclass
class TextChunk:
    chunk_id: Union[str, int]
    text: str
    metadata: ChunkMetadata
    similarity_score: float

@dataclass
class ConfidenceScore:
    retrieval_quality: float = 0.0
    coverage_score: float = 0.0
    entailment_score: float = 0.0

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
# Core Agent
# ============================================================================

class RetrievalAgent:
    """
    Conversational textbook tutor using Cohere Command R+ and Qdrant.
    """

    def __init__(
        self,
        model: str = "command-r-plus-08-2024",
        temperature: float = 0.3,
        top_k: int = 7,
        score_threshold: float = 0.3,
        collection_name: str = "textbook_embeddings",
    ):
        self.model = model
        self.temperature = temperature
        self.top_k = top_k
        self.score_threshold = score_threshold
        self.collection_name = collection_name

        self.openai_client = OpenAI(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1"
        )
        self.qdrant_client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
        )
        self.embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

    # ---------------------------------------------------------------------

    def _is_greeting(self, query: str) -> bool:
        return query.lower().strip() in {
            "hi", "hello", "hey", "help", "who are you", "what can you do"
        }

    def _greeting_response(self) -> str:
        return (
            "Hello! I'm your Humanoid Robotics textbook assistant. "
            "Ask me any concept and I’ll explain it clearly using the book."
        )

    # ---------------------------------------------------------------------

    def get_embedding(self, text: str) -> List[float]:
        # Use FastEmbed model to generate embeddings
        embeddings = list(self.embedding_model.embed([text]))
        return embeddings[0].tolist()  # Convert to list format

    # ---------------------------------------------------------------------

    def retrieve_information(self, query: str, top_k: Optional[int] = None, score_threshold: Optional[float] = None) -> List[TextChunk]:
        embedding = self.get_embedding(query)

        # Use provided parameters or fall back to instance defaults
        actual_top_k = top_k if top_k is not None else self.top_k
        actual_score_threshold = score_threshold if score_threshold is not None else self.score_threshold

        results = self.qdrant_client.query_points(
            collection_name=self.collection_name,
            query=embedding,
            limit=actual_top_k,
            query_filter=None,
            score_threshold=actual_score_threshold,
        ).points

        chunks = []
        for r in results:
            content = r.payload.get("content", "")

            # FILTER: Only ignore empty or extremely short content
            if len(content.split()) < 20:
                continue

            chunks.append(
                TextChunk(
                    chunk_id=r.id,
                    text=content,
                    similarity_score=r.score,
                    metadata=ChunkMetadata(
                        section_title=r.payload.get("title"),
                        url=r.payload.get("url"),
                        chunk_index=r.payload.get("chunk_index"),
                    ),
                )
            )

        return chunks

    # ---------------------------------------------------------------------

    def generate_answer(self, query: str, chunks: List[TextChunk]) -> str:
        if not chunks:
            return "I cannot find sufficient information in the textbook to answer that."

        # Format the context from retrieved chunks
        context = "\n\n".join([f"Document {i+1}: {c.text}" for i, c in enumerate(chunks)])

        system_message = (
            "You are a knowledgeable and conversational tutor for a Humanoid Robotics textbook.\n\n"
            "Instructions:\n"
            "- Explain concepts clearly and naturally, as a human instructor would.\n"
            "- Use ONLY the provided context for factual content.\n"
            "- Include definitions, explanations, and examples if available.\n"
            "- If the answer is not present in the context, say so clearly.\n\n"
            "Write in a helpful, textbook-explanation style."
        )

        user_message = f"Based on the following context, please explain:\n\n{context}\n\nQuestion: {query}"

        response = self.openai_client.chat.completions.create(
            model="mistralai/devstral-2512:free",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            temperature=self.temperature,
        )

        return response.choices[0].message.content

    # ---------------------------------------------------------------------

    def query(self, user_query: str, top_k: Optional[int] = None, score_threshold: Optional[float] = None) -> AgentResponse:
        start = time.time()

        if self._is_greeting(user_query):
            return AgentResponse(
                query_id=str(uuid.uuid4()),
                status=ResponseStatus.CONVERSATIONAL,
                answer=self._greeting_response(),
            )

        chunks = self.retrieve_information(user_query, top_k=top_k, score_threshold=score_threshold)
        answer = self.generate_answer(user_query, chunks)

        status = (
            ResponseStatus.INSUFFICIENT_CONTEXT
            if "cannot find sufficient information" in answer.lower()
            else ResponseStatus.SUCCESS
        )

        sources = [
            SourceReference(
                chunk_id=c.chunk_id,
                citation_index=i + 1,
                relevance_score=c.similarity_score,
                excerpt=c.text[:120] + "...",
                metadata=c.metadata,
            )
            for i, c in enumerate(chunks)
        ]

        return AgentResponse(
            query_id=str(uuid.uuid4()),
            status=status,
            answer=answer,  # ALWAYS RETURN ANSWER
            sources=sources,
            confidence=ConfidenceScore(
                retrieval_quality=max((c.similarity_score for c in chunks), default=0.0),
                coverage_score=0.8 if chunks else 0.0,
                entailment_score=0.8 if chunks else 0.0,
            ),
            metadata=ResponseMetadata(
                model=self.model,
                temperature=self.temperature,
                total_time_ms=(time.time() - start) * 1000,
                retrieval_time_ms=0.0,
                generation_time_ms=0.0,
            ),
        )

    # ---------------------------------------------------------------------

    def query_with_context(self, user_query: str, context: str, top_k: Optional[int] = None, score_threshold: Optional[float] = None) -> AgentResponse:
        """
        Query method that uses provided context instead of retrieving from vector database.
        This is used for the 'selected_text' mode in the API.
        """
        start = time.time()

        if self._is_greeting(user_query):
            return AgentResponse(
                query_id=str(uuid.uuid4()),
                status=ResponseStatus.CONVERSATIONAL,
                answer=self._greeting_response(),
            )

        # Create a TextChunk from the provided context
        # We'll use a dummy chunk_id and assign a high similarity score since it's user-provided context
        chunk = TextChunk(
            chunk_id="selected_context",
            text=context,
            similarity_score=1.0,  # High score since user provided this context
            metadata=ChunkMetadata(
                section_title="Selected Text",
                url="",
                chunk_index=0,
            ),
        )
        chunks = [chunk]

        answer = self.generate_answer(user_query, chunks)

        status = (
            ResponseStatus.INSUFFICIENT_CONTEXT
            if "cannot find sufficient information" in answer.lower()
            else ResponseStatus.SUCCESS
        )

        sources = [
            SourceReference(
                chunk_id=c.chunk_id,
                citation_index=i + 1,
                relevance_score=c.similarity_score,
                excerpt=c.text[:120] + "...",
                metadata=c.metadata,
            )
            for i, c in enumerate(chunks)
        ]

        return AgentResponse(
            query_id=str(uuid.uuid4()),
            status=status,
            answer=answer,  # ALWAYS RETURN ANSWER
            sources=sources,
            confidence=ConfidenceScore(
                retrieval_quality=max((c.similarity_score for c in chunks), default=0.0),
                coverage_score=0.8 if chunks else 0.0,
                entailment_score=0.8 if chunks else 0.0,
            ),
            metadata=ResponseMetadata(
                model=self.model,
                temperature=self.temperature,
                total_time_ms=(time.time() - start) * 1000,
                retrieval_time_ms=0.0,
                generation_time_ms=0.0,
            ),
        )

# ============================================================================
# CLI
# ============================================================================

def main():
    if len(sys.argv) < 2:
        print("Usage: python agent.py 'your question'")
        sys.exit(1)

    agent = RetrievalAgent()
    response = agent.query(sys.argv[1])

    print("\nAnswer:\n", response.answer)
    print("\nSources used:", len(response.sources))

if __name__ == "__main__":
    main()
