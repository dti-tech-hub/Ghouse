| mode | runs | groundedness | citation cov. | recall | abstention % | +latency p50 | llm calls |
|---|---|---|---|---|---|---|---|
| `mode1_normal_rag` | 361 | 0.9945 | 0.5967 | 0.9945 | 0.0 | 0.0 | 1.0 |
| `mode2_agentic_no_recovery` | 361 | 0.9945 | 0.5956 | 0.9945 | 0.0 | 0.0 | 1.0028 |
| `mode3_agentic_with_recovery` | 364 | 0.8516 | 0.4671 | 0.7895 | 34.07 | 82.9575 | 1.8819 |

| quantity | value |
|---|---|
| paired runs | 364 |
| mean groundedness delta | +0.1319 |
| improved | 52 |
| worsened | 0 |
| unchanged | 312 |
| improved % | 14.29 |
