# Historical initial law RAG verification — October 20, 2026

This report describes the former Gemma-based prototype, not the current semantic application. It extracted 408 articles from a specific consolidated criminal-code PDF and a Gazette page substitution, then created 923 child embeddings of dimension 768.

Four RAG tests and six training-pipeline tests passed. Targeted generation checks exposed interpretation errors in OR conditions and over-selection of related offenses. The prototype subsequently used source selection and verbatim quotations to reduce unchecked free-form interpretation.

This was a smoke test, not a legal QA benchmark. Footnotes and transitional provisions were not indexed. Source versions were fixed, and complete legal coverage was not established. The document-specific parser described here has been replaced in the current runtime.

This is an English historical summary, not a full translation. The complete original record is preserved unchanged as [plain text](originals/rag-verification.txt).
