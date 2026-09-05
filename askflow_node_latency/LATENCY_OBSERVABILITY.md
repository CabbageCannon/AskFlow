# Node Latency Observability

AskFlow Eval records two distinct runtime concepts:

1. End-to-end wall-clock latency (`latency_seconds`)
2. Inclusive graph-node invocation duration (`node_latency` / `node_timings`)

They are intentionally non-additive. Parent subgraph nodes contain child work,
and concurrent researchers/tools can overlap. Therefore the sum of node durations
may exceed end-to-end latency.

`aggregate_recorded_node_seconds` means aggregate inclusive recorded node work,
not critical-path time. Do not compute `node_total / e2e_latency` and describe it
as a true wall-clock percentage.

Each invocation keeps `start_offset_seconds` and `end_offset_seconds` so a later
critical-path/overlap analysis remains possible without changing Agent behavior.

Cross-task summary semantics:

- `mean_total_seconds_per_task`: tasks where a node did not run count as zero.
- `median_total_seconds_per_task`: same zero-inclusive semantics.
- `mean_invocation_seconds`: only real invocations.
- `p50_invocation_seconds`: only real invocations.
- `p95_invocation_seconds`: only real invocations.
- `total_calls`: total real invocations.
