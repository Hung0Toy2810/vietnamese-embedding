# Historical encoder evaluation: step 190,399

Snapshot: 2026-10-18T15:34:23.006080+07:00. All compared models used the same retained set of 1,010 queries and 3,791 passages.

| Model | MRR@10 | Hit@1 | Hit@5 | Hit@10 |
|---|---:|---:|---:|---:|
| Before fine-tuning | 0.6971 | 61.29% | 81.19% | 85.94% |
| Step 108,000 | 0.8089 | 73.76% | 90.10% | 92.87% |
| Step 190,399 | 0.8208 | 75.15% | 91.09% | 93.56% |
| Character TF-IDF | 0.7684 | 68.91% | 86.73% | 90.69% |
| BKAI reference | 0.9059 | 87.13% | 95.45% | 96.83% |

The paired bootstrap 95% interval for the MRR improvement over step 108,000 was [-0.0023, 0.0259], including zero. These historical development results do not establish a general improvement or evaluate the current RAG pipeline.

This is an English historical summary, not a full translation. The complete original record is preserved unchanged as [plain text](originals/evaluation-20261018.txt).
