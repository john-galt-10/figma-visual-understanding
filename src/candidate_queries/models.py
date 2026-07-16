"""Pydantic models for candidate retrieval-query generation."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class QueryResponse(BaseModel):
    """Defines the JSON schema requested from a vision-language provider."""

    retrieval_queries: list[str] = Field(
        description="Distinct, concise queries for retrieving official Figma documentation."
    )


class QueryResponseWithReasoning(QueryResponse):
    """Extends a query response with a short, inspectable query-selection summary."""

    reasoning_summary: str = Field(
        min_length=1,
        description=(
            "Brief explanation of the visible evidence and terminology used to choose "
            "the retrieval queries."
        ),
    )


class ImageMetadata(BaseModel):
    """Describes the validated screenshot sent to a query-generation provider."""

    path: str
    filename: str
    format: str
    width: int
    height: int


class CandidateQueryInput(BaseModel):
    """Captures the caller inputs used to generate candidate queries."""

    image: ImageMetadata
    textual_query: str | None = None


class GeneratorMetadata(BaseModel):
    """Identifies the provider and model used for a generation attempt."""

    provider: str
    model: str
    sdk_version: str | None = None


class CandidateQueryResult(BaseModel):
    """Contains normalized candidate queries and inspectable execution metadata."""

    model_config = ConfigDict(extra="forbid")

    input: CandidateQueryInput
    generator: GeneratorMetadata
    retrieval_queries: list[str]
    reasoning_summary: str | None = None
    processing_time_seconds: float
    metadata: dict[str, Any] = Field(default_factory=dict)
