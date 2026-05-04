# ADR-036: Pipeline Orchestration — EventBridge + ECS Fargate

**Date:** 2026-05-04
**Status:** Accepted

## Context

The ingestion pipeline (fetch → chunk → embed → upsert) needs to run on a schedule (every
48–72 hours) to keep the corpus current with new PubMed publications. A scheduling mechanism
is required for the production AWS deployment.

Apache Airflow was considered. Airflow provides a DAG UI, dependency graph visualization,
retry logic, and is widely used in data engineering. However, it requires running a webserver,
scheduler, worker, and metadata PostgreSQL instance — approximately $50–100/month in AWS
infrastructure to run a 4-task DAG twice a week.

## Decision

Use **AWS EventBridge cron rule → ECS Fargate run-task** for pipeline scheduling.

- EventBridge fires a cron rule every 48–72 hours
- Rule triggers a one-off ECS Fargate task running the pipeline container
- Pipeline exits when complete — no persistent worker process
- CloudWatch Logs captures stdout/stderr per run automatically
- Cost: ~$0/month for EventBridge + compute cost of the task run (~$0.10/run on 0.5 vCPU)

For step-level visibility and retry logic, **AWS Step Functions** can be layered on top:
- Each pipeline step (ingest, chunk, embed, upsert) becomes a Step Functions state
- Failed steps retry without restarting the full pipeline
- Step Functions console provides an Airflow-equivalent DAG view with zero ops overhead

Airflow is the right tool when managing 10+ DAGs across multiple data products with a
dedicated data platform team. pubmed_rag does not meet that threshold.

## Consequences

- No persistent scheduler process — infrastructure cost is near-zero when pipeline is idle
- CloudWatch Logs is the primary monitoring interface for pipeline runs
- Adding Step Functions is a natural upgrade path if per-step retry becomes important
- Pipeline container and API container share the same Docker image — no separate build needed
- Airflow remains an option if the project grows into a multi-pipeline data platform
