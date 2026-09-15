# Case study template (fill only with platform-recorded numbers)

**Customer / team:** ___ **Agent:** ___ **Environment:** ___ (tools: ___, tasks: ___, verifier: state-based / judge, judge–human agreement: ___%)

| metric (held-out, k=__) | baseline v__ | final v__ | Δ | 95% CI | p |
|---|---|---|---|---|---|
| task_success | | | | | |
| pass^k | | | | | |
| tool_selection_f1 | | | | | |
| context_preservation | | | | | |
| p95 latency (ms) | | | | | |
| tokens / task | | | | | |

**Learning mechanisms used:** router / exemplars / prompt opt / SFT / DPO / GRPO — iterations: __, rollouts collected: __.
**Live A/B:** n=__ per arm, outcome=__, Δ=__, CI=__, p=__. **Gate decisions:** __ promoted / __ blocked (eval run ids: __).
**Scale:** rollouts/day __, spans/day __, workers __, `saphire bench` attached: yes/no.
**Links:** eval run ids ___, training run ids ___, experiment id ___, audit export ___.
