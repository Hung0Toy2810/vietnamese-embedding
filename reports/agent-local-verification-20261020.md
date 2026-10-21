# Historical local agent verification — October 20, 2026

The former law-specific application used LangChain 1.4.6, LangGraph 1.2.16, langchain-ollama 1.1.0 and Qwen3.5:9B. At the recorded stage, 32 API/agent/grounding/stream tests passed and Pyright checked 29 files without errors.

Live targeted checks included Articles 22/23 with a source map, a general theft-penalty question covering Article 173 clauses 1–5 (188.5 seconds), contextual follow-up questions, and an out-of-scope refusal. A later focused agent check completed in 110.5 seconds with supplied real legal sources; it did not evaluate retrieval quality.

A CLI regression exposed the model confusing the document number 135 with the numbered source ID. The validator rejected it; after clarifying the prompt, the recorded rerun completed in 123.6 seconds with a valid citation. No quote validation was loosened.

The map represented source references rather than graph-based chunking. The old question classifiers and coverage checks relied on law-specific wording. They are historical behavior, not the current semantic graph design. Browser interaction was not visually verified in that experiment.

This is an English historical summary, not a full translation. The complete original record is preserved unchanged as [plain text](originals/agent-local-verification-20261020.txt).
