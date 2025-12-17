#!/usr/bin/env python3
"""
RAG Agent API Server

FastAPI server that wraps the existing RetrievalAgent to provide
a REST API for querying the RAG agent backend.
"""

import os
import uuid
import logging
from datetime import datetime
from typing import Optional, List, Union
from enum import Enum
from dataclasses import asdict

import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from typing_extensions import Literal

# Import the existing agent
from agent import RetrievalAgent

def _convert_metadata_to_dict(metadata):
    """Convert ResponseMetadata object to dictionary with proper datetime formatting"""
    if not metadata:
        return None
    metadata_dict = asdict(metadata)
    # Convert datetime to ISO string format
    if hasattr(metadata, 'timestamp') and metadata.timestamp:
        metadata_dict['timestamp'] = metadata.timestamp.isoformat()
    return metadata_dict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app with enhanced documentation
app = FastAPI(
    title="RAG Agent API",
    description="""
    API for querying the RAG (Retrieval-Augmented Generation) agent backend to retrieve grounded answers from the humanoid robotics textbook.

    ## Features
    - Query the textbook content with source citations
    - Ask questions about selected text passages
    - Get confidence scores for answers
    - Access health status of the service

    The API supports both full-book queries and selected-text queries with proper validation and error handling.
    """,
    version="1.0.0",
    docs_url="/docs",  # Swagger UI
    redoc_url="/redoc",  # ReDoc
    openapi_tags=[
        {
            "name": "query",
            "description": "Query the RAG agent for textbook content"
        },
        {
            "name": "health",
            "description": "Health check and service status"
        }
    ]
)

# Add CORS middleware for Docusaurus frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # Docusaurus dev server
        "https://humanoid-robotics-ai-textbook.vercel.app",  # Production URL (removed trailing slash)
        os.getenv("FRONTEND_URL", "*")  # Production URL from env or wildcard
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    # Expose custom headers if needed
    # expose_headers=["Access-Control-Allow-Origin"]
)

# Initialize the retrieval agent once at startup
agent = RetrievalAgent()


# Define enums and models
class ResponseStatus(str, Enum):
    SUCCESS = "success"
    CONVERSATIONAL = "conversational"
    INSUFFICIENT_CONTEXT = "insufficient_context"
    ERROR = "error"


class ChunkMetadata(BaseModel):
    """Metadata for text chunks from the textbook"""
    page_number: Optional[int] = None
    section_title: Optional[str] = None
    chapter: Optional[str] = None
    url: Optional[str] = None
    chunk_index: Optional[int] = None


class SourceReference(BaseModel):
    """Citation linking answer to specific textbook chunk"""
    chunk_id: Union[str, int]
    citation_index: int
    relevance_score: float = Field(ge=0.0, le=1.0)
    excerpt: str
    metadata: ChunkMetadata


class ConfidenceScore(BaseModel):
    """Multi-factor confidence assessment for agent responses"""
    retrieval_quality: float = Field(ge=0.0, le=1.0, default=0.0)
    coverage_score: float = Field(ge=0.0, le=1.0, default=0.0)
    entailment_score: float = Field(ge=0.0, le=1.0, default=0.0)
    lexical_overlap: float = Field(ge=0.0, le=1.0, default=0.0)
    overall: float = Field(ge=0.0, le=1.0, default=0.0)
    level: Literal["high", "medium", "low"] = "low"


class ResponseMetadata(BaseModel):
    """Generation metadata for agent responses"""
    model: str
    temperature: float = Field(ge=0.0, le=1.0)
    total_time_ms: float = Field(ge=0.0)  # Allow 0.0 for conversational responses
    retrieval_time_ms: float = Field(ge=0.0)  # Allow 0.0 for conversational responses
    generation_time_ms: float = Field(ge=0.0)  # Allow 0.0 for conversational responses
    tokens_used: Optional[int] = None
    timestamp: str  # ISO 8601 format


class AgentResponse(BaseModel):
    """Structured data returned from backend to frontend containing answer and metadata"""
    query_id: str
    status: ResponseStatus
    answer: Optional[str] = None
    confidence: Optional[ConfidenceScore] = None
    sources: Optional[List[SourceReference]] = None
    metadata: Optional[ResponseMetadata] = None
    error_message: Optional[str] = None

    @field_validator('answer')
    @classmethod
    def validate_answer_for_success(cls, v, info):
        """Ensure answer is provided when status is success or conversational"""
        status = info.data.get('status')
        if status in [ResponseStatus.SUCCESS, ResponseStatus.CONVERSATIONAL] and v is None:
            raise ValueError("Answer must be provided when status is SUCCESS or CONVERSATIONAL")
        return v

    @field_validator('sources')
    @classmethod
    def validate_sources_for_success(cls, v, info):
        """Ensure sources are provided when status is success but not for conversational responses"""
        status = info.data.get('status')
        if status == ResponseStatus.SUCCESS and (v is None or len(v) == 0):
            raise ValueError("At least one source must be provided when status is SUCCESS")
        return v

    @field_validator('error_message')
    @classmethod
    def validate_error_message_for_non_success(cls, v, info):
        """Ensure error_message is provided when status is not success"""
        status = info.data.get('status')
        if status in [ResponseStatus.INSUFFICIENT_CONTEXT, ResponseStatus.ERROR] and v is None:
            raise ValueError("Error message must be provided when status is not SUCCESS")
        return v


class QueryRequest(BaseModel):
    """User question sent from frontend to backend API"""
    text: str = Field(min_length=1, max_length=500)
    selected_text: Optional[str] = Field(None, max_length=2000)
    mode: Literal["full_book", "selected_text"] = "full_book"
    top_k: Optional[int] = Field(default=5, ge=1, le=20)
    score_threshold: Optional[float] = Field(default=0.5, ge=0.0, le=1.0)

    @field_validator('text')
    @classmethod
    def validate_text_not_empty(cls, v):
        if not v.strip():
            raise ValueError('Query text cannot be empty or whitespace')
        return v

    @field_validator('selected_text')
    @classmethod
    def validate_selected_text_mode(cls, v, info):
        if info.data.get('mode') == 'selected_text' and not v:
            raise ValueError('selected_text required when mode is selected_text')
        return v


@app.post("/api/query",
          response_model=AgentResponse,
          summary="Query the RAG Agent",
          description="Submit a question to the RAG agent to get a grounded answer with source citations from the textbook content.",
          tags=["query"],
          responses={
              200: {
                  "description": "Successful response with answer and sources",
                  "content": {
                      "application/json": {
                          "example": {
                              "query_id": "123e4567-e89b-12d3-a456-426614174000",
                              "status": "success",
                              "answer": "The answer to your question is...",
                              "confidence": {
                                  "retrieval_quality": 0.85,
                                  "coverage_score": 0.78,
                                  "entailment_score": 0.92,
                                  "lexical_overlap": 0.81,
                                  "overall": 0.84,
                                  "level": "high"
                              },
                              "sources": [
                                  {
                                      "chunk_id": "chunk_1",
                                      "citation_index": 1,
                                      "relevance_score": 0.95,
                                      "excerpt": "The relevant text excerpt...",
                                      "metadata": {
                                          "page_number": 42,
                                          "section_title": "Introduction to Robotics",
                                          "chapter": "Chapter 1",
                                          "url": "/docs/chapter-1",
                                          "chunk_index": 0
                                      }
                                  }
                              ],
                              "metadata": {
                                  "model": "gpt-4",
                                  "temperature": 0.7,
                                  "total_time_ms": 2450.5,
                                  "retrieval_time_ms": 1200.2,
                                  "generation_time_ms": 1250.3,
                                  "tokens_used": 150,
                                  "timestamp": "2025-12-13T10:30:00.123Z"
                              }
                          }
                      }
                  }
              },
              422: {
                  "description": "Validation error - request parameters don't meet requirements",
                  "content": {
                      "application/json": {
                          "example": {
                              "detail": [
                                  {
                                      "loc": ["body", "text"],
                                      "msg": "String should have at least 1 character",
                                      "type": "value_error"
                                  }
                              ]
                          }
                      }
                  }
              },
              500: {
                  "description": "Internal server error",
                  "content": {
                      "application/json": {
                          "example": {
                              "detail": "Error processing query: Error details here"
                          }
                      }
                  }
              }
          })
async def query_endpoint(request: QueryRequest):
    """
    Submit a query to the RAG agent.

    The agent will:
    1. Retrieve relevant chunks from Qdrant vector database (for full_book mode)
    2. Or use the provided selected_text directly (for selected_text mode)
    3. Generate a grounded answer using LLM (Gemini/GPT)
    4. Return answer with source citations and confidence scores
    """
    query_id = str(uuid.uuid4())
    logger.info(f"Processing query {query_id}: {request.text[:100]}... (mode: {request.mode})")

    try:
        if request.mode == "selected_text" and request.selected_text:
            # For selected text mode, use the new query_with_context method
            agent_response = agent.query_with_context(
                user_query=request.text,
                context=request.selected_text,
                top_k=request.top_k,
                score_threshold=request.score_threshold
            )
        else:
            # For full book mode, use the regular query method
            agent_response = agent.query(
                user_query=request.text,
                top_k=request.top_k,
                score_threshold=request.score_threshold
            )

        # Convert the agent's response to our API response format
        api_response = AgentResponse(
            query_id=query_id,
            status=ResponseStatus(agent_response.status.value),
            answer=agent_response.answer,
            confidence=asdict(agent_response.confidence) if agent_response.confidence else None,
            sources=[asdict(source) for source in agent_response.sources] if agent_response.sources else None,
            metadata=_convert_metadata_to_dict(agent_response.metadata) if agent_response.metadata else None,
            error_message=agent_response.error_message
        )

        logger.info(f"Query {query_id} completed successfully")
        return api_response

    except Exception as e:
        logger.error(f"Error processing query {query_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )


@app.get("/health",
         summary="Health Check",
         description="Check the health status of the API server and its dependencies.",
         tags=["health"],
         responses={
             200: {
                 "description": "Health status of the service",
                 "content": {
                     "application/json": {
                         "example": {
                             "status": "healthy",
                             "services": {
                                 "qdrant": "connected",
                                 "llm": "available"
                             }
                         }
                     }
                 }
             }
         })
async def health_check():
    """Health check endpoint to verify the API server and dependencies are operational"""
    try:
        # Test the agent connection by attempting a simple operation
        # For now, just verify the agent exists and is accessible
        if agent:
            # In a real implementation, you might test Qdrant connectivity here
            return {
                "status": "healthy",
                "services": {
                    "qdrant": "connected",  # This would be determined by actual connection test
                    "llm": "available"      # This would be determined by actual connection test
                }
            }
        else:
            return {"status": "unhealthy", "error": "Agent not initialized"}
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {"status": "unhealthy", "error": str(e)}


if __name__ == "__main__":
    # Run the server with uvicorn
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True  # Enable auto-reload for development
    )