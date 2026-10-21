# CV metrics verification - 20 October 2026

## Vietnamese retrieval encoder: fresh evaluation

Checkpoint step 190399, 77,083,392 parameters. SHA256 `70866d2abe4d881c8310dd4fd042df862c8cd09c5b85cf8d62e24f93e43e2a37` was unchanged after inference. Production tokenizer was retained. No training or model selection was performed in this evaluation.

Source: [GreenNode/Zalo Legal Retrieval](https://huggingface.co/datasets/GreenNode/zalo-ai-legal-text-retrieval-vn), revision `12d76d4d04b94ceada970fcfbe7fec20bfa97389`. Test qrels contain 788 distinct labelled query IDs. Duplicate query-ID rows with identical text were collapsed; all labelled query IDs were retained.

Fixed before model scoring: all 447 relevant articles plus SHA256(seed 42, document ID) selected distractors, for 5,000 articles total from the full 61,425-article corpus. No sampling based on model quality. Exact IDs and input hashes are in `protocol.json`.

Articles are encoded using title + text, split into 224-token windows with 40-token overlap; 11,193 children. No long article is silently truncated. Article score is the maximum cosine of its child embeddings. Query and candidate vectors are L2 normalized. No reranker or LLM participates.

| Metric | Value |
|---|---:|
| Document Recall@100 | 0.987310 |
| Document Recall@10 | 0.932741 |
| MRR@10 | 0.767366 |
| nDCG@10 | 0.807810 |
| Hit@1 | 0.661168 |

778/788 queries have a labelled relevant article in the first 100 document candidates. Multiple positive labels are accounted for by Recall, distinct from Hit. All-query Recall@100 was recomputed separately; plain Python max aggregation and ranking matched seven sampled queries using the same batched float32 cosine matrix. Small rounding differences can reorder near-tied scores if matrix multiplication batch shapes change.

Inference ran on MPS in 157.8 seconds. This is a fixed 5,000-document evaluation subset, **not** an official full-corpus MTEB result. Max over chunks differs from single-vector document encoding. Complete historical training text is unavailable, so absence of training overlap cannot be proven. These are encoder-only document retrieval metrics, not end-to-end RAG answer accuracy and not exactly the web pipeline's top-100-child recall. The original development tracking set was not used to generate these new values.

Reproduce from the project root:

```bash
python3 reports/cv-evaluation-20261020/evaluate_77m.py
```

`encoder-result.json` holds metrics. `evaluation-tensors.pt` holds CPU vectors, owners, rankings and labels for numerical checks; it is not a model checkpoint.

## Human Activity Recognition: all six configurations

Metrics recomputed with scikit-learn from original saved prediction CSVs, not newly trained models. All six have the same 9,420 unique window IDs and labels from participants S13-S15. Seed 42; eight classes. Splits are by participant. E6 uses causal EMA; the others do not. Test results were inspected during development and should not be called an untouched final test.

| Configuration | Accuracy | Macro F1 |
|---|---:|---:|
| Random Forest, PPG+ACC | 0.730892 | 0.758709 |
| 1D CNN, PPG+ACC | 0.750955 | 0.779853 |
| 2D CNN, PPG+ACC | 0.737367 | 0.754620 |
| 2D CNN, ACC only | 0.713482 | 0.736349 |
| 2D CNN, PPG only | 0.447877 | 0.458166 |
| Multikernel 1D CNN + EMA, PPG+ACC | 0.814650 | 0.853085 |

`har-result.json` includes source hashes and confusion matrices. Historical ESP32 mean time for the multikernel CNN + EMA is 88.349124 ms over the same 1,000 replayed test windows (50 warm-ups); it excludes USB transfer, sensor acquisition and the eight-second collection interval. This is not a new board benchmark.

## Emergency sound: comparable primary-label scores

The original AST report uses multi-label threshold F1, whereas CNN uses one primary label. The CV instead uses **primary-label confusion-matrix metrics** for both. Matrices were read visually from the retained figures and transcribed in `audio-result.json`; both have 447 samples and matching class supports after accounting for the different class order.

| Model | Accuracy | Macro F1 | Evidence |
|---|---:|---:|---|
| CNN | 0.642058 | 0.658834 | `cnn/evaluation_report/confusion_matrix.png` |
| AST | 0.838926 | 0.846256 | `ast/plots_and_metrics/confusion_matrix.png` |

CNN was additionally rerun on all 447 original local mel images using the retained checkpoint and deterministic evaluation transforms. The new confusion matrix exactly matches the saved matrix (`cnn-retest.json`). The test split is also used by the retained training code to select checkpoints, so these are development results.

AST could not be rerun: its checkpoint and original test audio are absent from the local package. Its saved primary-label matrix gives 375/447 correct (83.89%), while the text report says 83.67%. The text report's 0.8469 F1 is a different, multi-label metric and is not used as a comparable primary-label F1. CV wording explicitly attributes AST numbers to the saved matrix; it does not claim fresh AST inference, a controlled comparison on identical verified audio IDs, or independent held-out performance.

## CV content decisions

- Add the encoder + legal RAG project, linked to the public vietnamese-embedding repository.
- Describe all six HAR configurations with seed-42 test metrics and name the model for the ESP32 timing.
- Include both AST/CNN primary-label metrics with the saved-matrix/model-selection qualifications.
- Removed Neural Atmosphere Operator entirely from the CV at the user's request; three projects remain.
- Keep existing education, contact details and skill levels, including GPA 7.31/10, without inventing certificates, work experience or proficiency.
- Rebuild the PDF from clean source; do not paint over or embed pages of the old CV.

## PDF verification

Final artifact: `Interview/CV_Luong_Nguyen_Thanh_Hung.pdf`, rebuilt as a new one-page PDF using Times typography and the existing education/skills layout. Three verified repository links plus email/phone/profile links are clickable. Latest full-page render was visually reviewed. Text extraction contains exactly one name and all expected metric values; no old template identity or preliminary weather metrics remain. Checkpoint/tokenizer/history-report hashes remain unchanged. The GitHub publication includes the local legal_rag source; the checkpoint, tokenizer and evaluation tensors are distributed through the checkpoint-step-190399 release.

Latest revision: Neural Atmosphere Operator and its link removed; the remaining text is unchanged after whitespace normalization. One-page rendering was visually rechecked.
