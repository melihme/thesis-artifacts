# Wiki reference results

Use these aggregate results from the original 81-configuration Wiki experiment to compare with your run. They contain no WikiNER records, demonstration text, raw responses, or per-example predictions.

- `matrix_results_81.csv`: one completed aggregate row for each model, device, strategy, and seed configuration
- `wiki_aggregates.csv`: 37 groups recalculated from `matrix_results_81.csv`
- `table_quality_latency.csv`: primary GPU encoder and LLM rows for thesis tables
- `table_output_reliability.csv`: request and schema-validation counts for the reported LLM configurations
- `figure_f1_latency_points.csv`: aggregate quality and latency points for plotting
- `latency_components_aggregate.csv`: mean, median, p90, p95, p99, minimum, and maximum component timings
- `entity_type_metrics_aggregate.csv`: exact-span counts and metrics by entity type
- `bert_cpu_gpu_speedups.csv`: matched CPU/GPU latency and throughput ratios
- `protocol_manifest_public.json`: experiment protocol, source hashes, and encoder artifact checksums
- `environment_summary.json`: software and A100 details from the source Wiki run
- `cohort_manifest.json`: definition of the 81 configurations included in reporting
- `matrix_status_public.json`: source-matrix and reported-cohort completion counts

The reference results cover all 81 configurations, including the seven Trendyol configurations.
