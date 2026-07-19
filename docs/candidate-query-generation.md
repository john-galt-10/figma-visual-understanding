# Candidate Query Generation

The `src/candidate_queries` package turns a Figma screenshot, and optionally a user question, into a small list of queries for the companion textual RAG system. It is intentionally separate from OCR: a configured vision-language provider receives the screenshot directly and returns a Pydantic-validated result that is easy to inspect or persist.

## User intent categories

Use the following minimal categories when annotating the intent of a user question:

- `identify_and_define`: What is this element, and what is its basic purpose?
- `how_to`: How to perform an action, configure something, find a control, or use a shortcut.
- `explain_behavior`: Why a specific state, effect, or UI behavior occurs—for example, why a control is disabled.
- `troubleshoot`: How to fix unexpected or incorrect behavior.
- `compare`: Difference between two elements, features, modes, or approaches.

## Setup

Install the project dependencies in the `figma-navigator` Python 3.11 environment:

```bash
pip install -r requirements.txt
```

Set the API credential environment variable named by `candidate_queries.api_key_environment_variable` in the selected configuration file. For example:

```text
YOUR_PROVIDER_API_KEY=your-key
```

The `.env` file is ignored by Git. The generator loads it with `python-dotenv`; environment variables already set by the shell also work.

## Run the example

```bash
python scripts/candidate_query_example.py --image-path screenshots_example/aaaa0000.png
python scripts/candidate_query_example.py --image-path screenshots_example/aaaa0000.png --text-query "What does this control do?"
python scripts/candidate_query_example.py --image-path screenshots_example/aaaa0000.png --output-trace
```

`--image-path` is required and must point to a supported raster screenshot. `--text-query` is optional. With text, the model refines that intent using visible Figma context. Without text, it identifies the relevant visible feature and generates questions that help explain it, such as a button, slider, panel, or menu item. `--output-trace` adds a short `reasoning_summary` to this run's JSON and overrides the configuration value.

The command writes an indented JSON object to standard output. It contains the validated retrieval queries, input image metadata, selected provider/model, elapsed time, and the configured query limit. Errors are written to standard error and return exit code `1`.

## Configuration and provider contract

`query_gen_config.yaml` controls the active provider and model, candidate limit, prompt, API-key environment variable, temperature, output-token limit, `thinking_level`, and `output_reasoning_summary`. `thinking_level: minimal` preserves the configured provider's low-latency behavior where supported. Set `output_reasoning_summary: true` to include a brief, structured explanation of the query choices in every result; it is not a raw internal reasoning trace.

`CandidateQueryGenerator.generate(image_path, textual_query=None, output_trace=None)` is the stable provider contract. Passing `True` for `output_trace` enables the summary for that call; `None` uses the YAML setting. Provider adapters return `CandidateQueryResult`, a Pydantic model. An adapter supplies the provider's structured response contract, validates the response locally, removes blank/duplicate entries, and enforces `max_candidates`.

To add a provider, implement `CandidateQueryGenerator` and register it in `create_candidate_query_generator`. Callers and the CLI continue to consume the normalized Pydantic result without provider-specific changes.
