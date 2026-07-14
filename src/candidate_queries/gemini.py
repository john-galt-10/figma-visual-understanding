"""Gemini implementation of the candidate-query provider interface."""

from __future__ import annotations

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
)
from .config import CandidateQuerySettings
from .models import (
    CandidateQueryInput,
    CandidateQueryResult,
    GeneratorMetadata,
    QueryResponse,
)


class GeminiCandidateQueryGenerator(CandidateQueryGenerator):
    """Generate documentation-retrieval queries with the Gemini multimodal API."""

    provider_name = "gemini"

    def __init__(self, settings: CandidateQuerySettings) -> None:
        """Store provider settings selected from the project's configuration file."""
        self.settings = settings

    def generate(
        self, image_path: str | Path, textual_query: str | None = None
    ) -> CandidateQueryResult:
        """Send a screenshot and optional user intent to Gemini and normalize its JSON."""
        image_metadata = self.read_image_metadata(image_path)
        normalized_textual_query = self._normalize_textual_query(textual_query)
        api_key = self._load_api_key()
        client, types = self._create_client(api_key)
        prompt = self._build_prompt(normalized_textual_query)

        started_at = perf_counter()
        try:
            with Image.open(image_metadata.path) as source_image:
                image = source_image.copy()
                response = client.models.generate_content(
                    model=self.settings.model,
                    contents=[prompt, image],
                    config=types.GenerateContentConfig(
                        system_instruction=self.settings.system_instruction,
                        response_mime_type="application/json",
                        response_schema=QueryResponse,
                        temperature=self.settings.generation.temperature,
                        max_output_tokens=self.settings.generation.max_output_tokens,
                    ),
                )
            query_response = self._parse_response(response)
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
                image=image_metadata, textual_query=normalized_textual_query
            ),
            generator=GeneratorMetadata(
                provider=self.provider_name,
                model=self.settings.model,
                sdk_version=self._sdk_version(),
            ),
            retrieval_queries=queries,
            processing_time_seconds=round(elapsed, 6),
            metadata={"configured_max_candidates": self.settings.max_candidates},
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
    def _build_prompt(textual_query: str | None) -> str:
        """Describe whether Gemini should refine a user intent or infer one from the image."""
        if textual_query:
            return f"User question: {textual_query}\nGenerate retrieval-query formulations."
        return (
            "No user question was supplied. Inspect the screenshot and generate "
            "feature-identification retrieval-query formulations."
        )

    @staticmethod
    def _parse_response(response: object) -> QueryResponse:
        """Validate Gemini's parsed or textual structured response with Pydantic."""
        parsed = getattr(response, "parsed", None)
        try:
            if isinstance(parsed, QueryResponse):
                return parsed
            if parsed is not None:
                return QueryResponse.model_validate(parsed)
            response_text = getattr(response, "text", None)
            if not response_text:
                raise ValueError("response did not contain parsed JSON or text")
            return QueryResponse.model_validate_json(response_text)
        except (ValidationError, ValueError) as error:
            raise CandidateQueryProviderError(
                f"Gemini returned an invalid structured response: {error}"
            ) from error

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
