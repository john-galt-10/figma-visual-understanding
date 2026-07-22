# Figma Visual Understanding

This repository contains the visual-understanding layer for a personal assistant focused on explaining Figma's interface and behavior.

The broader project aims to help users understand UI elements in Figma by combining visual context from the Figma UI with grounded documentation retrieval from the companion `figma-rag` system. This repository is dedicated specifically to translating screenshots and other image-like UI context into structured text signals that downstream retrieval and answer-generation systems can use.

## Project goal

The goal of this project is to build an inspectable visual understanding layer that can identify likely Figma UI elements, controls, states, and surrounding context from screenshots of the Figma interface.

The long-term target workflow is:

1. capture UI context from Figma, such as a screenshot, pointer location, selection state, or cropped region
2. process the visual input with OCR, vision-language models, layout analysis, or other computer vision methods
3. produce a structured description of the visible Figma UI and the element or feature likely being referenced
4. pass that description to `figma-rag` as image-derived query/context
5. retrieve relevant official Figma documentation and generate a grounded explanation

This repository focuses on steps 1 through 3: visual input processing and image-to-text translation.

## Scope of this repository

This repo is responsible for:

* accepting screenshots or cropped UI regions from Figma
* experimenting with OCR, vision-language models, and layout-aware visual parsing
* detecting visible text, controls, panels, menus, toolbars, and interface states
* generating structured textual descriptions of screenshot content
* producing candidate feature labels or retrieval queries for `figma-rag`
* evaluating visual understanding outputs against manually reviewed examples
* comparing different visual-understanding approaches over the same screenshot set

This repo does **not** initially focus on:

* collecting or indexing official Figma documentation
* chunking, embedding, or retrieving documentation
* grounded answer generation from retrieved documentation
* building the Figma plugin UI
* production deployment

Those retrieval and answer-generation responsibilities live in the companion `figma-rag` repository.

## Relationship to figma-rag

`figma-rag` is the text/documentation layer. It answers questions using official Figma documentation.

`figma-visual-understanding` is the visual context layer. It converts screenshots of the Figma UI into useful text and metadata for retrieval.

The intended interface between the two systems is a structured visual context object, for example:

```json
{
  "screenshot_id": "example-001",
  "visible_text": ["Design", "Prototype", "Inspect", "Auto layout"],
  "ui_regions": [
    {
      "label": "right_sidebar",
      "description": "Design panel showing Auto layout controls"
    }
  ],
  "target_hint": {
    "text": "Auto layout",
    "region": "right_sidebar",
    "confidence": 0.82
  },
  "retrieval_queries": [
    "Figma Auto layout controls",
    "What does Auto layout do in Figma?"
  ]
}
```

The exact schema is expected to evolve as experiments clarify what the RAG layer needs most.

## Development approach

This project is primarily an exploratory project. The main priorities are:

* understanding which visual signals are useful for Figma UI assistance
* comparing simple baselines against stronger multimodal models
* keeping experiment inputs, outputs, and metrics inspectable
* designing a clear handoff format for `figma-rag`
* building small, replaceable pipeline stages rather than one opaque system

The implementation philosophy is:

* start with simple screenshot fixtures and explicit scripts
* prefer readable Python modules and plain data artifacts
* separate capture, OCR, VLM inference, normalization, and evaluation
* make intermediate outputs easy to inspect
* optimize for iteration speed before optimizing for performance

## Milestone recap

* **2026-07-10: Repository initialization**  
  Created the initial README and defined the repository's role as the visual-understanding layer for the broader Figma assistant system.
* **2026-07-13: OCR baseline**
  Added a replaceable OCR interface, PaddleOCR and EasyOCR implementations, and a JSON inspection script for screenshot scans.
* **2026-07-14: Icon candidate generation**
  Added configurable classical-CV region proposals for likely icons, including inspectable overlays and matcher-ready candidate metadata.
* **2026-07-15: Icon library and matching**
  Added icon-template library generation with shared normalization, plus configurable Chamfer matching and an optional soft-template tie breaker for ranking single-icon crops.
* **2026-07-15: Icon matching evaluation**
  Added labeled candidate-crop evaluation with configurable score thresholding, threshold mining, precision, recall, MRR, and end-to-end and detection accuracy reporting.
* **2026-07-16: Unified visual-signal pipeline**
  Added one configurable OCR, icon-matching, and candidate-query pipeline that provides ordered, inspectable visual evidence to the VLM, supports token-free signal inspection, and writes a unified JSON artifact.
* **2026-07-17: Configurable VLM input modes**
  Added vanilla, segmented, and hybrid screenshot inputs so VLM experiments can ground accepted icon detections in retained numbered overlays.
* **2026-07-19: Human query-evaluation support**
  Added timestamped candidate-query artifacts with configuration snapshots and a resumable human-review UI for target correctness, intent preservation, grounding, and standalone quality.

## Repository intent

This repository is intentionally narrow in scope: it is the visual context and image-to-text translation layer of the larger Figma UI assistant project.

Its role is to provide reliable, inspectable context from screenshots before handing off to the documentation-grounded retrieval and answer-generation system in `figma-rag`.
