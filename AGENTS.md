## Project Purpose

This repository contains the visual-understanding layer for a personal assistant focused on explaining Figma's interface and behavior.

The broader assistant project combines two systems:

figma-visual-understanding: converts screenshots or image-like UI context from Figma into structured textual signals
figma-rag: retrieves official Figma documentation and generates grounded explanations

This repository is responsible only for the visual context and image-to-text translation layer. It should produce useful, inspectable context that downstream systems can use for retrieval, answer generation, or further reasoning.

The project is also intended as a learning project for multimodal understanding, RAG-adjacent query generation, and agentic patterns. Prioritize clarity, inspectability, and experiment quality over production scale.

## Repository structure

Use the existing repository layout and do not introduce new top-level folders unless necessary.

- `scripts/`: entrypoint scripts and one-off pipeline commands
- `src/`: reusable Python modules and pipeline logic
- `docs/`: design notes, architecture, and planning documents

When implementing new functionality:
- put reusable logic in `src/`
- keep CLI or task entrypoints in `scripts/`
- keep generated metadata close to the dataset it describes
- avoid creating additional top-level directories unless explicitly needed

You can find general info about the project in `README.md`.

## Technology stack

Use a Python-first stack for the entire RAG system.

Core technologies:

- Python 3.11.15 (use figma-navigator conda environment) 
- simple local storage using files and JSONL
- image-processing libraries only when they add clear value
- OCR utilities when useful for extracting visible UI text
- external vision-language model APIs or local VLMs when explicitly part of an experiment

Guidelines:

- prefer plain Python over framework-heavy stacks
- keep dependencies minimal and easy to understand
- keep capture/loading, OCR, VLM inference, layout analysis, normalization, query generation, and evaluation as separate modules
- design components so OCR, VLMs, visual heuristics, and query-generation strategies can be swapped without restructuring the project

## General indications

- Don't create tests if not explicitly asked. Not everything needs tests.
- Everytime you implement a script, present in the chat how to run it and the meaning of the CLI parameters. Also write a very short summary of the functioning (e.g., input, output, underlying mechanism).
- Comment the code you write. I want a docstring for each class and function, if not obvious or self-explanatory.
- Whenever adding new features or making significant changes, prompt me whether we should update the "Milestone recap" section in the README.md of the repo.
- For every major feature, add a documentation .md file in docs/