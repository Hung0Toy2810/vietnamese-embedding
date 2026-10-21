# Vietnamese Retrieval Encoder and Semantic Document RAG

A 77,083,392-parameter Vietnamese text encoder implemented in PyTorch, with offline retrieval training and a local document question-answering application.

The RAG application builds its document graph from **language meaning**. Font sizes, headings, page boundaries and file extensions do not create semantic relationships. Source layout is retained only for extraction and citations.

## Encoder architecture

- Bidirectional Transformer: 8 layers, hidden size 768, 12 attention heads.
- RoPE, RMSNorm and SwiGLU; intermediate size 1792.
- Masked mean pooling, 768-dimensional projection and L2 normalization.
- Vocabulary size 32,000; maximum encoded input length 256 tokens.

The architecture, packaged checkpoint and tokenizer are unchanged by the RAG redesign. This is a Vietnamese encoder; support for many file formats does not imply demonstrated retrieval quality in every language.

## Restore model artifacts

```bash
gh release download checkpoint-step-190399 \
  --repo Hung0Toy2810/vietnamese-embedding \
  --pattern 'vietnamese-encoder-artifacts.tar.gz*'
shasum -a 256 -c vietnamese-encoder-artifacts.tar.gz.sha256
tar -xzf vietnamese-encoder-artifacts.tar.gz
```

The archive contains `checkpoints/best-step-190399.pt`, the production tokenizer and its metadata, and the saved evaluation tensors. Training data and user uploads are not included.

The packaged checkpoint has no optimizer state. It supports inference or initialization of a new training run, not exact continuation of its historical run.

## Install

Use a dedicated Python 3.11 virtual environment. Install the appropriate PyTorch build for your machine if the default wheel is unsuitable.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt -r legal_rag/requirements.txt
ollama pull qwen3.5:9b
```

Docling may download document-analysis/OCR models on first use. The reranker also downloads model weights on first use. Qwen runs through the local Ollama service.

## Run the document application

```bash
python -m legal_rag.webapp
```

Open http://127.0.0.1:8765. Upload a PDF, DOCX, PPTX, XLSX, HTML, Markdown or UTF-8 text file. Documents enter a durable, sequential ingestion queue. Review extraction warnings before activating a version that needs review.

The package retains the name `legal_rag` for command compatibility, but the runtime no longer requires a law PDF or a law-specific index.

```text
Read source → preserve locations → language units → embeddings
→ document-specific chunk plan → source-linked semantic hypotheses
→ semantic subject groups → parent/child chunks → versioned retrieval index
```

For detailed behavior, limitations, configuration and source layout, see [the RAG guide](legal_rag/README.md).

## Train the encoder

Preprocess once, then train independently of downloading and preprocessing:

```bash
python prepare.py --sources examples/sources.json
python train.py --data data/retrieval --output runs/retrieval --device cuda
python evaluate.py --data data/retrieval --checkpoint runs/retrieval/latest.pt --device cuda
python predict_embeddings.py "How can I return an item?" "Items can be returned within seven days."
```

The source manifest pins source revisions and selects at most 160,000 / 20,000 / 20,000 raw rows from its example files. It does not download all records from those datasets. Files are downloaded completely before preprocessing. You may also supply `--input /path/to/pairs.jsonl` or Parquet.

Example record:

```json
{"query":"What is the return deadline?","positive":"Items may be returned within seven days.","hard_negatives":["The store opens at eight."]}
```

The retail examples in `examples/` are synthetic command demonstrations. Training data may remain Vietnamese.

Preprocessing normalizes Unicode, validates QA answer offsets, rejects invalid or oversized pairs, groups known positives, and separates connected query/positive/context groups across train and validation. This prevents exact content overlap across those groups, not every possible semantic near-duplicate. The output is a SQLite corpus with token IDs and metadata; an existing output directory is not overwritten.

Training uses multi-positive InfoNCE, AdamW, learning rate 1e-4, 500 warmup updates, gradient clipping 1.0, effective batch size 64 and encoder microbatch size 16 by default. Gradient caching is used. The default initialization is the packaged checkpoint; `--from-scratch` initializes randomly. CUDA allocation is budgeted conservatively, but driver allocations and other processes are outside that allocator budget.

New runs write to their output directory, never to `checkpoints/`. Use `--resume` with unchanged data/settings to resume a new-format training run. Ctrl+C/SIGTERM finishes the current update and saves; power loss/SIGKILL can only recover the latest saved state. The simplified training code does not reproduce the original historical training procedure exactly.

## Evaluation scope

A fixed Zalo Legal subset evaluated on October 20, 2026 contains **788 queries and 5,000 documents**. Encoder-only **Document Recall@100 = 98.73%**, **MRR@10 = 0.7674**. Each document receives the maximum cosine score of its child windows.

This is not a full-corpus MTEB result, child-level recall, a benchmark of the new semantic graph pipeline, or RAG answer accuracy. Complete historical training text is unavailable, so absence of training overlap cannot be established. See [the reproducible evaluation protocol](reports/cv-evaluation-20261020/README.md).

The historical development set of 1,000 queries / 3,763 passages produced Hit@1 75.20%, Hit@10 93.50% and MRR@10 0.8211 for the packaged checkpoint. That set informed checkpoint selection and is not an untouched final test.

## Source map

| Files | Responsibility |
|---|---|
| `model.py`, `config.py` | Encoder architecture and configuration |
| `prepare.py`, `text.py`, `sources.py` | Offline data preparation |
| `data.py`, `losses.py`, `grad_cache.py` | Training data, objective and gradient caching |
| `train.py`, `checkpoint.py` | Training and checkpoint persistence |
| `evaluate.py`, `predict_embeddings.py` | Encoder evaluation and inference |
| `legal_rag/` | Semantic ingestion, retrieval, agent, API and browser interface |
| `reports/` | Evaluation reports and clearly identified historical records |

## Checks

```bash
python tests/test_pipeline.py
python -m unittest discover -s legal_rag/tests -v
python -m pyright
```

Historical reports describe the versions and experiments from their recorded dates. English summaries link to byte-preserved original records under `reports/originals/`. They are not validation claims for current code. Protected source artifact hashes are recorded in `PACKAGE_MANIFEST.json`.
