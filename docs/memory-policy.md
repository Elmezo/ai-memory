# Memory Policy Architecture

## Current architecture

The AI Memory Server stores memories in Elasticsearch with hybrid retrieval
(BM25 + dense vectors fused via RRF and a local cross-encoder reranker).
Historically the flow was:

    Store -> Retrieve -> Return

## New architecture

The system now supports an explicit evidence-state-based memory interaction
layer:

    Query
      ↓
    Evidence State
      ↓
    Memory Policy (SEARCH / STOP)
      ↓
    Multi-view retrieval + reranker
      ↓
    Candidates
      ↓
    Candidate evaluation (ADD / SKIP)
      ↓
    Working memory
      ↓
    STOP

Memory operations are explicit, traceable decisions.

## Evidence state

`EvidenceState` tracks:

- query
- working_memory
- retrieved_candidates
- search_history
- action_history
- provenance
- step, max_steps
- tenant_id, user_id, agent_id, project_id

## Action model

`MemoryActionType` defines:

- SEARCH
- ADD
- SKIP
- STOP

`MemoryAction` records action_type, step, query, memory_id, reason,
confidence, affected_memory_ids, result_count, latency, metadata, timestamp.

## Policy interface

`MemoryPolicy` exposes:

- `decide_search(state)`
- `evaluate_candidate(state, candidate)`
- `should_stop(state)`

The default implementation is `RuleBasedMemoryPolicy`. Future
`LLMMemoryPolicy` or `RLMemoryPolicy` implementations can replace it without
changing storage, retrieval, APIs, MCP, authentication, or tenant isolation.

## Reward interface

`RewardEvaluator` provides advisory, deterministic process-level scoring via
`ProcessReward`:

- action_quality
- evidence_relevance
- evidence_coverage
- memory_faithfulness
- redundancy_penalty
- stopping_quality
- overall_score

The reward system is observational and does not alter production behavior
unless explicitly enabled.

## Provenance

Optional source-traceability fields were added non-destructively:

- source_type
- source_id
- source_reference
- source_span
- created_from
- provenance
- parent_memory_id

Missing provenance is handled explicitly and never fabricated.

## Multi-view memory representation

Optional fields support multiple retrieval views:

- content (raw)
- summary
- event_phrase
- query_anchors (stored; not yet used by retrieval)

Retrieval currently combines `content`, `summary`, and `event_phrase` when
`MEMORY_ENABLE_MULTI_VIEW=true`. `query_anchors` is persisted for future use
but is not yet searched. The system continues to work when only `content`
exists. Multi-view fields are optional and are usually empty for older
documents; no values are fabricated.

## Retrieval flow

    Query
      ↓
    multiple retrieval views
      ↓
    candidate fusion (RRF)
      ↓
    existing reranker
      ↓
    final candidates

Tenant, user, and agent filters are applied before results reach the policy.

## Security considerations

Authorization remains deterministic and enforced below the policy layer.
Tenant/user/agent isolation is preserved in every retrieval and interaction.
The policy never decides authorization.

## Backward compatibility

All existing APIs, MCP tools, fields, and tests remain supported.
New functionality is additive. Elasticsearch mapping changes are optional
and non-destructive.

## Future RL roadmap

Not implemented yet:

- learned memory policy
- reward model training
- DPO
- RL / PA-DAPO
- benchmark reproduction

The current goal is to establish infrastructure, interfaces, observability,
and evaluation before introducing expensive policy training.

## Why we are not implementing RL yet

Training a memory policy requires stable infrastructure, reliable
observability, and evaluation signals. This change establishes those
foundations first, keeping the system production-oriented and avoiding
unverified claims.
