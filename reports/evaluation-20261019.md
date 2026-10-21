# Historical saturation evaluation: step 249,740

Snapshot: 2026-10-19T11:13:56.443276+07:00. Models were scored on the same 1,000-query / 3,763-passage development set with fixed labels and tie-breaking.

| Model | MRR@10 | Hit@1 | Hit@5 | Hit@10 |
|---|---:|---:|---:|---:|
| Step 108,000 | 0.8089 | 73.80% | 90.00% | 92.90% |
| Step 190,399 | 0.8211 | 75.20% | 91.00% | 93.50% |
| Step 249,740 | 0.8099 | 74.30% | 89.90% | 92.30% |
| Before fine-tuning | 0.6961 | 61.20% | 81.10% | 85.90% |
| Character TF-IDF | 0.7671 | 68.70% | 86.80% | 90.80% |
| BKAI reference | 0.9055 | 87.10% | 95.40% | 96.90% |

The last checkpoint did not clearly improve on step 108,000 and trailed step 190,399. This set informed checkpoint selection. It is not an untouched final test, and none of these numbers evaluates semantic graph retrieval or answer quality.

This is an English historical summary, not a full translation. The complete original record is preserved unchanged as [plain text](originals/evaluation-20261019.txt).
