"""Pydantic models for candidate retrieval-query generation."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class QueryResponse(BaseModel):
    """Defines the JSON schema requested from a vision-language provider."""

    retrieval_queries: list[str] = Field(
        description="Distinct, concise queries for retrieving official Figma documentation."
    )


class FocusBox(BaseModel):
    """Locate the focused image within a full context screenshot in pixel coordinates."""

    x: int = Field(ge=0, description="Zero-based horizontal coordinate in the context image.")
    y: int = Field(ge=0, description="Zero-based vertical coordinate in the context image.")
    width: int = Field(gt=0, description="Positive focus-box width in pixels.")
    height: int = Field(gt=0, description="Positive focus-box height in pixels.")


class ScreenContext(BaseModel):
    """Store optional, observable context inferred from the full screenshot."""

    selected_layer_type: str | None = Field(
        default=None,
        description="Generic visible selected-layer type, such as rectangle; never a layer name.",
    )
    referenced_panel: str | None = Field(
        default=None,
        description="Visible Figma panel name when it helps identify the focused target.",
    )
    visible_evidence_summary: str | None = Field(
        default=None,
        description="Concise, directly observable evidence supporting the contextual fields.",
    )


class ContextQueryResponse(QueryResponse):
    """Extend the provider response only for requests that include a context screenshot."""

    selected_layer_type: str | None = Field(
        default=None,
        description="Generic visible selected-layer type, such as rectangle; never a Figma layer name.",
    )
    referenced_panel: str | None = Field(
        default=None,
        description="Visible Figma panel name when relevant to the focused target.",
    )
    visible_evidence_summary: str | None = Field(
        default=None,
        description="Concise observable contextual evidence, without unsupported inference.",
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


class ContextQueryResponseWithReasoning(ContextQueryResponse):
    """Add the optional inspectable query-selection summary to a context response."""

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


class ContextInput(BaseModel):
    """Describe the source and submitted full-screen context image for one request."""

    source_image: ImageMetadata
    submitted_image: ImageMetadata
    focus_bbox: FocusBox | None = None
    annotated_context_image_path: str | None = None


class CandidateQueryInput(BaseModel):
    """Captures the caller inputs used to generate candidate queries."""

    image: ImageMetadata
    images: list[ImageMetadata] = Field(default_factory=list)
    textual_query: str | None = None
    context_input: ContextInput | None = None


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
    screen_context: ScreenContext | None = None
    processing_time_seconds: float
    metadata: dict[str, Any] = Field(default_factory=dict)
