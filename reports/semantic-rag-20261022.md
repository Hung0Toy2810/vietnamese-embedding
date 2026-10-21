# Semantic RAG implementation verification — October 22, 2026

## Scope

The document-specific law parser, page-based parent construction, law-specific
prompts/tools and mandatory built-in law index have been replaced by a generic
semantic ingestion and retrieval pipeline. All authored code, comments, prompts
and current Markdown documentation are English. Vietnamese source content and
localization remain data. Historical original reports are preserved as unchanged
text records, with clearly identified English summaries at their former paths.

## What determines the graph

Language units are derived from sentence punctuation and encoder-safe source
windows after canonicalizing whitespace. Fonts, heading levels, page boundaries
and file extensions are not semantic graph features. Source offsets, parser item
IDs and pages are retained only for provenance.

The 77M encoder proposes nearest-neighbor pairs. Qwen generates a document-specific
plan and verifies shortlisted pairs by meaning. Each accepted relation needs quotes
that map to both source units. Relation labels are open-ended rather than a legal
ontology. Sequence edges are excluded from community grouping. NetworkX semantic
communities form parents; source-preserving child windows fit the real encoder.

Exact-quote validation checks evidence existence, not whether a claimed relation
is logically true. Candidate shortlisting can miss dependencies. No claim of a
complete knowledge graph or measured enterprise retrieval quality is made.

The layout-based S2 alternative was not integrated because the final requirement
specifically excludes layout as the basis for semantic graph construction.

## Automated checks

- 40 RAG unit/integration tests passed in the isolated Python 3.11 environment.
- Six encoder training/preprocessing/gradient/resume/evaluation tests passed.
- Pyright standard: zero errors and zero warnings.
- Ruff F checks and JavaScript syntax checks passed.
- JSDOM checks passed for the document list, removed header, non-PDF preview,
  extraction-review warning visibility, citations, Markdown sanitization and
  multiple-format upload support. JSDOM is not a visual browser test.
- The six original protected artifacts matched their SHA256 values. Historical
  report originals moved to `reports/originals/`; their bytes did not change.
- Actual CPU embeddings from the original encoder source and the reformatted,
  English-comment source matched exactly for two inputs. No architecture change
  or checkpoint/tokenizer rewrite occurred.

Tests specifically cover line-wrap invariance, same-page unrelated text,
nonadjacent exceptions, open-ended relation labels, invented quotes/identifiers,
long sentences, complete source coverage, strict source/passages, translated
citation labels, real LangGraph tool loops with deterministic models, context
budgets, version isolation, interrupted ingestion, warning review, persisted jobs
and optional web URL restrictions. A processing budget failure is distinguished
from absence of supporting source evidence.

## Actual document conversion

Docling converted small PDF, DOCX, PPTX, XLSX and HTML fixtures. UTF-8 text and
Markdown were read without rewriting their content. See
[format checks](semantic-rag-20261022/format-checks.json).

A raster-only English PDF was OCR'd back into the expected sentence, retaining a
page/source item. That first OCR check took approximately 284 seconds, including
initialization/model acquisition. It is not a steady-state throughput benchmark
or proof of Vietnamese OCR quality. After explicitly installing ONNX Runtime in the clean isolated environment, the same raster-only PDF passed again in 11.70 seconds with cached models. See [initial OCR trace](semantic-rag-20261022/ocr-check.json) and [clean-runtime rerun](semantic-rag-20261022/ocr-check-final-runtime.json).

## Actual encoder, semantic graph, reranker and Qwen checks

A synthetic Vietnamese store-policy document interleaves returns, refunds and
machine-safety instructions. The final language-unit revision produced 10 nodes,
5 semantic edges, 7 semantic parents and 10 children. The persisted [plan](semantic-rag-20261022/plan.json)
and [graph](semantic-rag-20261022/graph.json) include a relationship between distant
returns/refund statements. Safety units were not merged merely because they were
adjacent in the source.

Building that graph took 663.85 seconds on the tested local setup. An earlier
coarser-window trial took 333.44 seconds. These are ingestion timings on a very
small fixture, not document-size scaling measurements. Local reasoning-model
calls are the main observed latency cost; no large-corpus throughput or memory
benchmark has been completed.

| Check | Seconds | Recorded outcome |
|---|---:|---|
| Broad returns question | 245.72 | Answered deadline, conditions and exceptions; six validated evidence items |
| Corporate tax rate absent from the document | 28.93 | Refused with zero evidence; did not invent a rate |
| Custom-made product returns, through HTTP API | 65.61 | Answered with a source-backed exclusion and version/source-position metadata |

The first broad-question trial needed structured-output repairs. An early API
trial exhausted its model rounds because validation feedback was too generic.
That failure is retained in [the original trace](semantic-rag-20261022/api-validation-failure-before-fix.json).
The final validator explains which citation/source/passage is invalid; translated
citation labels are canonicalized without changing numbers or loosening quote
checks. The same API question then completed in two model rounds. See
[CLI answers](semantic-rag-20261022/live-answers.json) and [API trace](semantic-rag-20261022/api-smoke.json).

All live inference turns ran to natural completion. No active generation was
terminated early. These are development smoke checks on synthetic content,
not a held-out evaluation or an answer-accuracy percentage.

## HTTP and interface checks

The application root, catalog, semantic-graph endpoint and non-PDF preview
returned successful responses. Preview HTML has an inert-content CSP. The API
returns evidence with a pinned document version and source positions.

No controllable browser was available in this session, so actual browser layout,
PDF plugin behavior, resizing and pointer interaction were not visually verified.
The final local server uses the persistent project workspace. A clearly named
synthetic store-policy document has a working semantic index for demonstration.
Existing source PDFs were registered as unindexed documents without activating
legacy indexes; users must build their new semantic indexes. Original source
files and earlier indexes remain untouched.

## Deployment boundaries

This is a local single-process workspace with durable jobs, restartable artifact
stages, validated semantic plans and version-scoped vector retrieval. It is not
an authenticated shared enterprise deployment. Tenant authorization, retention/
physical-deletion policies, distributed workers, Qdrant server deployment,
operational backups and a held-out multi-document QA benchmark remain deployment
work. No new CV metrics should be inferred from these smoke checks.

The October 20 encoder evaluation harness now owns its frozen legacy overlapping
window helper. It no longer imports the application's changing chunker; its
reported protocol and metric files remain separate from the semantic pipeline.
