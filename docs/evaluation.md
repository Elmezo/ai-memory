# Evaluation

## Why evaluation exists

The evaluation harness measures whether the new memory interaction architecture
improves memory retrieval and decision quality compared with the existing
baseline. It is policy-agnostic and does not train or invoke any learned model.

## Dataset format

JSONL, one JSON object per line:

```json
{
  "id": "q001",
  "query": "...",
  "tenant_id": "...",
  "user_id": "...",
  "agent_id": "...",
  "project_id": "...",
  "relevant_memory_ids": ["mem_001"],
  "expected_answer": "...",
  "metadata": {}
}
```

Supports multiple relevant memories, no relevant memories, tenant/user/agent
isolation scenarios, and duplicate memories.

## Baselines

- `search`: existing normal memory search.
- `interaction`: current AiO-inspired interaction controller.

Future baselines (LLM/RL) can be registered in `evaluation/baselines.py`.

## Metrics

### Retrieval

- Recall@K = |retrieved[:K] ∩ relevant| / |relevant| (None if relevant empty)
- Precision@K = |retrieved[:K] ∩ relevant| / K
- MRR = 1 / rank of first relevant retrieved (0 if none)
- NDCG@K = DCG@K / IDCG@K

### Decisions

- ADD precision = |ADD ∩ relevant| / |ADD| (1.0 if no ADD)
- ADD recall = |ADD ∩ relevant| / |relevant| (None if relevant empty)
- redundant ADD rate = duplicate ADD / total ADD
- SKIP accuracy = correctly skipped irrelevant / total SKIP
- useful SEARCH rate = searches with results and not redundant / total
- redundant SEARCH rate = repeated queries / total searches
- premature STOP = STOP while a relevant memory is missing from ADD
- unnecessary continuation = more than one search with no relevant ADD on last
- correct STOP = not premature and not unnecessary

### Trajectory

Counts of SEARCH/ADD/SKIP/STOP, total actions, unique added memories,
duplicate searches, duplicate additions.

### Performance

Total latency, ES query count, reranker calls, candidates processed. Policy
execution time and per-stage latency are not separately measured; reported as
unavailable rather than fabricated.

### Process reward

Heuristic process rewards from `RewardEvaluator`: per-action, total, average.
These are not learned rewards.

## Running

```powershell
cd memory-api
python -m evaluation.run --dataset evaluation/dataset.jsonl --memories evaluation/memories.jsonl --reset-index --baseline interaction --output evaluation/results
```

## Output

`run.json` and `trajectories.jsonl` in the output directory.

## Canonical / equivalent memories

Datasets may include `equivalent_memory_ids`, mapping a duplicate memory ID to
its canonical ID. Retrieval and decision metrics canonicalize IDs before
comparison, so duplicate suppression is not penalized as a retrieval failure.

## ADD relevance gating

Score semantics:
- Primary score: reranker score when available.
- Fallback: no gating when reranker score is absent; Elasticsearch score is
  recorded but not thresholded.
- Malformed or non-finite reranker scores are treated as absent.
- Threshold 0.0 disables gating.

`MEMORY_ADD_RELEVANCE_THRESHOLD` (default `0.0`) gates ADD decisions. With
`0.0`, all non-duplicate candidates are added. Higher values require a
minimum reranker or Elasticsearch score before ADD.

## Current evaluation scope

- Current policy is rule-based.
- Process rewards are heuristic.
- No RL is trained.
- No DPO is performed.
- No PA-DAPO is performed.
- Evaluation infrastructure implemented; benchmark results pending for
  production-scale data. Synthetic CI results are not production benchmarks.
