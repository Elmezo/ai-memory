# Evaluation Error Analysis

## Summary

Current measured difference on the synthetic CI dataset:

Existing Search Recall@5 = 1.0
Interaction Recall@5 = 0.9375

The interaction system retrieves one fewer relevant memory ID on one query
because it intentionally suppresses a content-identical duplicate.

## Per-query Results

| Query | Baseline Recall@5 | Interaction Recall@5 | Failure |
|---|---:|---:|---|
| database port | 1.0 | 0.5 | DUPLICATE_SUPPRESSION_FAILURE |
| dark theme preference | 1.0 | 1.0 | None |
| PostgreSQL production decision | 1.0 | 1.0 | None |
| API rate limit | 1.0 | 1.0 | None |
| Redis cache TTL | 1.0 | 1.0 | None |
| deployment approval | 1.0 | 1.0 | None |
| backup schedule | 1.0 | 1.0 | None |
| quantum computing | N/A | N/A | None |
| database password | N/A | N/A | None |
| staging database port | 1.0 | 1.0 | None |

Baseline retrieved IDs include both `mem_001` and `mem_006` for `database port`.
Interaction retained `mem_001` and skipped `mem_006` as `redundant_content`.

## Failure Categories

| Category | Count | Percentage |
|---|---:|---:|
| Retrieval failure | 0 | 0% |
| ADD decision failure | 0 | 0% |
| STOP failure | 0 | 0% |
| Duplicate suppression | 1 | 100% of recall failures |
| Policy query failure | 0 | 0% |
| Other | 0 | 0% |

Only one query has `baseline_recall > interaction_recall`: `database port`.

## ADD Precision Analysis

ADD precision = 0.75. False-positive ADDs observed:

- `PostgreSQL production decision`: added `mem_001` (production database port)
  in addition to relevant `mem_004`. It was added because it was novel and
  non-redundant; the rule-based policy has no relevance threshold
  (`MEMORY_ADD_RELEVANCE_THRESHOLD=0.0`) and adds every non-redundant candidate.
- `database password`: added `mem_002` and `mem_001` although ground truth is
  empty. These are loosely related database memories.
- `staging database port`: added `mem_001` in addition to relevant `mem_002`.

Root cause for ADD precision is the absence of an effective relevance gate:
the policy adds all novel candidates returned by retrieval.

## STOP Analysis

Incorrect STOP count by recall metric: 0 direct STOP failures.

q001 is marked premature by decision metrics because `mem_006` is not in
working memory, but `mem_006` was already retrieved and explicitly skipped as
redundant content. Another search would not recover it unless duplicate
suppression were disabled.

Other STOPs use `sufficient_evidence` after one search. q008 correctly uses
`no_new_evidence`.

## Search Analysis

The rule-based policy does not synthesize new queries; it searches the
original user query once. Redundant search rate = 0.0. Useful search rate =
0.9 (q008 has no results). Search query quality is not the dominant recall
limitation in this dataset.

## Multi-view Analysis

`query_anchors`, `summary`, and `event_phrase` are not populated in the
synthetic memories. No recall failure in this dataset can be attributed to
unused multi-view fields. There is no evidence from this dataset that
query-anchor retrieval would change the results.

## Ground-Truth Normalization Update

After Phase 2A, `q001` uses:

```json
"relevant_memory_ids": ["mem_001"],
"equivalent_memory_ids": {"mem_006": "mem_001"}
```

Thus interaction Recall@5 becomes 1.0 while duplicate suppression remains
enabled.

## Missing-Reranker-Score Fix

Earlier positive-threshold sweeps collapsed Recall@5 because candidates with
missing reranker scores were thresholded using unrelated Elasticsearch scores.
The policy now bypasses gating when the reranker score is absent, malformed,
or non-finite. Positive thresholds therefore no longer destroy Recall@5 on
the available datasets, but they remain dataset-dependent.

## Root Cause

The dominant cause of the Recall@5 difference is duplicate-content
suppression, not retrieval failure, ADD failure, or premature STOP.
The interaction system correctly avoids storing a duplicate memory, but the
evaluation ground truth counts the duplicate ID as separately relevant.

## Recommended Changes

1. Decide whether duplicate memories should be separate ground-truth IDs or
   collapsed into one canonical memory before changing policy behavior.
2. If higher ADD precision is desired, add and evaluate a relevance gate
   for ADD decisions; do not change thresholds without measurement.
3. Keep STOP behavior unchanged until duplicate-ground-truth semantics are
   clarified.

No policy, retrieval, or reward changes were made in this analysis task.
