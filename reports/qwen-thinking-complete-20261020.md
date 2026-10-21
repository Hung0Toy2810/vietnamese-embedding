# Historical complete Qwen reasoning trials — October 20, 2026

Three targeted questions used the same five legal articles as the preceding comparison. Settings: reasoning enabled, streaming enabled, temperature 0, context 16,384 tokens, num_predict=-1, no inference read timeout.

| Case | Seconds | Generated tokens, including reasoning | Observation |
|---|---:|---:|---|
| Theft involving at least two million VND | 214.46 | 4,274 | Distinguished threshold branches; did not enumerate all higher penalty ranges. |
| First theft involving three million VND | 105.45 | 2,264 | Did not import the conditions of the below-threshold branch. |
| Below two million VND and exceptions | 229.44 | 4,872 | Preserved the main exception logic in the conclusion. |

All ended naturally with done_reason=stop and nonempty final answers. Two needed more than 4,096 generated tokens; early interruption would not fairly assess them. These three examples do not establish general legal answer accuracy.

A recorded CLI smoke test for legitimate defense retrieved Article 22 and produced a cited explanation. Raw answers/configuration remain in [the JSON record](qwen-thinking-complete-20261020.json). No model checkpoint was modified.

This is an English historical summary, not a full translation. The complete original record is preserved unchanged as [plain text](originals/qwen-thinking-complete-20261020.txt).
