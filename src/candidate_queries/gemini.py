"""Gemini implementation of the candidate-query provider interface."""

from __future__ import annotations

from contextlib import ExitStack
from importlib import metadata
import os
from pathlib import Path
from time import perf_counter

from dotenv import load_dotenv
from PIL import Image
from pydantic import ValidationError

from .base import (
    CandidateQueryConfigurationError,
    CandidateQueryDependencyError,
    CandidateQueryGenerator,
    CandidateQueryProviderError,
    build_user_prompt,
)
from .config import CandidateQuerySettings
from .context import CONTEXT_SYSTEM_INSTRUCTION, PreparedContextInput, prepare_context_input
from .models import (
    CandidateQueryInput,
    CandidateQueryResult,
    ContextQueryResponse,
    ContextQueryResponseWithReasoning,
    FocusBox,
    GeneratorMetadata,
    QueryResponse,
    QueryResponseWithReasoning,
    ScreenContext,
)

class GeminiCandidateQueryGenerator(CandidateQueryGenerator):
    """Generate documentation-retrieval queries with the Gemini multimodal API."""

    provider_name = "gemini"

    def __init__(self, settings: CandidateQuerySettings) -> None:
        """Store provider settings selected from the project's configuration file."""
        self.settings = settings

    def generate(
        self,
        image_path: str | Path,
        textual_query: str | None = None,
        output_trace: bool | None = None,
        visual_context: str | None = None,
        input_description: str | None = None,
        additional_image_paths: list[str | Path] | None = None,
        context_image_path: str | Path | None = None,
        focus_bbox: FocusBox | None = None,
        context_artifact_directory: str | Path | None = None,
        prepared_context_input: PreparedContextInput | None = None,
    ) -> CandidateQueryResult:
        """Send ordered screenshot inputs and optional user intent to Gemini as JSON queries."""
        image_metadata = self.read_image_metadata(image_path)
        if prepared_context_input is not None and context_image_path is None:
            raise CandidateQueryConfigurationError(
                "prepared_context_input requires context_image_path provenance."
            )
        context_input = prepared_context_input or prepare_context_input(
            context_image_path,
            focus_bbox,
            context_artifact_directory,
        )
        additional_metadata = [
            self.read_image_metadata(path) for path in (additional_image_paths or [])
        ]
        normalized_textual_query = self._normalize_textual_query(textual_query)
        api_key = self._load_api_key()
        client, types = self._create_client(api_key)
        effective_input_description = self._effective_input_description(
            input_description,
            context_input is not None,
        )
        prompt = build_user_prompt(
            normalized_textual_query,
            visual_context,
            effective_input_description,
        )
        should_output_reasoning = self._should_output_reasoning(output_trace)
        response_schema = self._response_schema(should_output_reasoning, context_input is not None)
        effective_system_instruction = self._effective_system_instruction(context_input is not None)

        started_at = perf_counter()
        try:
            context_metadata = (
                [context_input.context_input.submitted_image] if context_input is not None else []
            )
            image_metadata_list = [image_metadata, *context_metadata, *additional_metadata]
            with ExitStack() as image_stack:
                images = [
                    image_stack.enter_context(Image.open(metadata.path)).copy()
                    for metadata in image_metadata_list
                ]
                response = client.models.generate_content(
                    model=self.settings.model,
                    contents=[prompt, *images],
                    config=types.GenerateContentConfig(
                        system_instruction=effective_system_instruction,
                        response_mime_type="application/json",
                        response_schema=response_schema,
                        temperature=self.settings.generation.temperature,
                        max_output_tokens=self.settings.generation.max_output_tokens,
                        thinking_config=types.ThinkingConfig(
                            thinking_level=self.settings.generation.thinking_level
                        ),
                    ),
                )
            query_response = self._parse_response(response, response_schema)
        except CandidateQueryProviderError:
            raise
        except Exception as error:
            raise CandidateQueryProviderError(
                f"Gemini failed to generate candidate queries: {error}"
            ) from error
        elapsed = perf_counter() - started_at
        queries = self._normalize_queries(query_response.retrieval_queries)
        if not queries:
            raise CandidateQueryProviderError("Gemini returned no usable retrieval queries.")

        return CandidateQueryResult(
            input=CandidateQueryInput(
                image=image_metadata,
                images=image_metadata_list,
                textual_query=normalized_textual_query,
                context_input=(context_input.context_input if context_input is not None else None),
            ),
            generator=GeneratorMetadata(
                provider=self.provider_name,
                model=self.settings.model,
                sdk_version=self._sdk_version(),
            ),
            retrieval_queries=queries,
            reasoning_summary=self._reasoning_summary(query_response),
            screen_context=self._screen_context(query_response) if context_input is not None else None,
            processing_time_seconds=round(elapsed, 6),
            metadata={
                "configured_max_candidates": self.settings.max_candidates,
                "thinking_level": self.settings.generation.thinking_level,
                "reasoning_summary_enabled": should_output_reasoning,
                **(
                    {"effective_system_instruction": effective_system_instruction}
                    if context_input is not None
                    else {}
                ),
            },
        )

    def _load_api_key(self) -> str:
        """Load the configured Gemini key from the repository's dotenv context."""
        load_dotenv()
        environment_variable = self.settings.api_key_environment_variable
        api_key = os.getenv(environment_variable)
        if not api_key:
            raise CandidateQueryConfigurationError(
                f"Missing {environment_variable}. Set it in .env or the environment."
            )
        return api_key

    @staticmethod
    def _create_client(api_key: str):
        """Create the Google Gen AI client only when the provider is selected."""
        try:
            from google import genai
            from google.genai import types
        except ImportError as error:
            raise CandidateQueryDependencyError(
                "Gemini support requires google-genai. Install dependencies with: "
                "pip install -r requirements.txt"
            ) from error
        return genai.Client(api_key=api_key), types

    @staticmethod
    def _normalize_textual_query(textual_query: str | None) -> str | None:
        """Convert blank optional CLI text into an absent user intent."""
        if textual_query is None:
            return None
        normalized = textual_query.strip()
        return normalized or None

    @staticmethod
    def _parse_response(
        response: object, response_schema: type[QueryResponse]
    ) -> QueryResponse:
        """Validate Gemini's parsed or textual structured response with Pydantic."""
        parsed = getattr(response, "parsed", None)
        try:
            if isinstance(parsed, response_schema):
                return parsed
            if parsed is not None:
                return response_schema.model_validate(parsed)
            response_text = getattr(response, "text", None)
            if not response_text:
                raise ValueError("response did not contain parsed JSON or text")
            return response_schema.model_validate_json(response_text)
        except (ValidationError, ValueError) as error:
            raise CandidateQueryProviderError(
                f"Gemini returned an invalid structured response: {error}"
            ) from error

    def _should_output_reasoning(self, output_trace: bool | None) -> bool:
        """Use the CLI preference when supplied, otherwise retain the YAML setting."""
        if output_trace is not None:
            return output_trace
        return self.settings.generation.output_reasoning_summary

    @staticmethod
    def _reasoning_summary(query_response: QueryResponse) -> str | None:
        """Extract the requested short rationale without exposing an internal trace."""
        summary = getattr(query_response, "reasoning_summary", None)
        return summary.strip() if isinstance(summary, str) else None

    @staticmethod
    def _screen_context(query_response: QueryResponse) -> ScreenContext:
        """Normalize the context-only response fields into the public result record."""
        return ScreenContext(
            selected_layer_type=getattr(query_response, "selected_layer_type", None),
            referenced_panel=getattr(query_response, "referenced_panel", None),
            visible_evidence_summary=getattr(query_response, "visible_evidence_summary", None),
        )

    @staticmethod
    def _response_schema(
        should_output_reasoning: bool, context_enabled: bool
    ) -> type[QueryResponse]:
        """Select the exact structured schema required by this request shape."""
        if context_enabled:
            return ContextQueryResponseWithReasoning if should_output_reasoning else ContextQueryResponse
        return QueryResponseWithReasoning if should_output_reasoning else QueryResponse

    def _effective_system_instruction(self, context_enabled: bool) -> str:
        """Append context roles only when the caller supplied a full-screen screenshot."""
        if not context_enabled:
            return self.settings.system_instruction
        return f"{self.settings.system_instruction}\n\n{CONTEXT_SYSTEM_INSTRUCTION}"

    @staticmethod
    def _effective_input_description(
        input_description: str | None, context_enabled: bool
    ) -> str | None:
        """Supply explicit image roles for standalone context calls without changing crop-only prompts."""
        if input_description or not context_enabled:
            return input_description
        return (
            "Images are ordered: the first image is the focused screenshot and primary target "
            "for retrieval queries. The second image is a full-screen context screenshot and "
            "may clarify surrounding state or panels, but is not a separate target."
        )

    def _normalize_queries(self, raw_queries: list[str]) -> list[str]:
        """Remove blank or duplicate queries and enforce the configured maximum."""
        queries: list[str] = []
        seen: set[str] = set()
        for raw_query in raw_queries:
            query = raw_query.strip()
            deduplication_key = query.casefold()
            if not query or deduplication_key in seen:
                continue
            seen.add(deduplication_key)
            queries.append(query)
            if len(queries) == self.settings.max_candidates:
                break
        return queries

    @staticmethod
    def _sdk_version() -> str | None:
        """Return the installed Google Gen AI SDK version when it is discoverable."""
        try:
            return metadata.version("google-genai")
        except metadata.PackageNotFoundError:
            return None
