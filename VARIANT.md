# VARIANT: taste-2 — semantic-heavy

**Thesis under test:** does a real LLM judge beat deterministic rules?
Where taste-1 scores with strings and centroids, taste-2 scores with a
frontier model judging output against retrieved layers — and carries richer
standard docs (rule + examples + counterexamples) instead of atoms.

## Permutation

| Axis | Choice | Divergence from control |
|---|---|---|
| A schema | A2 rich docs (rule + example + counterexample) | documents, not atoms |
| B retrieval | B2 hybrid RRF (BM25 + vector fusion) | two-index recall |
| C embedding | C2 Qwen3-Embedding-0.6B | heavier, better recall |
| D scorer | **D2 LLM-judge** (frontier model vs taste) | the axis under test |
| E interview | E4 lazy (only when layers conflict/insufficient) | less owner friction |
| F hone | F2 critique-revise (generate→critique→revise) | cheaper than BoN |
| G storage | G2 LanceDB | versioned, columnar |
| H memory | H5 confidence reinforcement (confirm→boost, contradict→decay) | explicit conf math |
| I injection | I2 few-shot exemplars ("write like THIS") | style via examples |
| J pipeline | J2 dynamic decomposition | agent chooses segments |

## The question it answers

If owner picks prefer judge-scored output, D2 earns its token cost.
If the control keeps up, rules win on cost and complexity — useful either way.
