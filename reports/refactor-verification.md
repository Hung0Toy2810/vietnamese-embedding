# Historical training-code refactor verification — October 19, 2026

The training source was reduced from 57 Python files to 13 main modules. The new workflow preprocesses once and trains with multi-positive InfoNCE; it does not reproduce the original training run exactly.

The packaged 308,357,883-byte checkpoint remained unchanged with SHA256 `70866d2abe4d881c8310dd4fd042df862c8cd09c5b85cf8d62e24f93e43e2a37`. The recorded real CPU embeddings matched exactly before/after that refactor. Optimizer state is absent from the original checkpoint; new runs save their own resume state elsewhere.

Six CPU tests passed, covering QA spans, connected splits, multiple positives, cached-gradient equivalence, train/resume equivalence, evaluation ranking and signal-safe saving. Pyright standard and Ruff F checks passed at that recorded revision. No full retraining, CUDA/A4500 profiling or retrieval-quality benchmark was performed by those checks.

This is an English historical summary, not a full translation. The complete original record is preserved unchanged as [plain text](originals/refactor-verification.txt).
