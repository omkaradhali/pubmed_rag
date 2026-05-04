# ADR-038: LLM Production Default — Claude Sonnet

**Date:** 2026-05-04
**Status:** Accepted

## Context

The project supports a pluggable LLM provider via `LLM_PROVIDER` env var, with Ollama as the
OSS default (free, local, no API key). For production deployment, a hosted LLM is needed.
The primary candidates were Claude Haiku and Claude Sonnet from Anthropic.

## Decision

Production default: **Claude Sonnet 4.6** (`claude-sonnet-4-6`).

Cost per query at production scale (top_k=5, ~1,750 input tokens, ~300 output tokens):
- Sonnet: ~$0.010/query
- Haiku: ~$0.003/query

Monthly projection at 100 queries/day:
- Sonnet: ~$31/month
- Haiku: ~$8/month

At expected production scale ($31/month), Sonnet's answer quality and citation reliability
justify the cost differential over Haiku. The gap narrows at very high traffic (500+
queries/day → $157/month Sonnet vs $42/month Haiku), at which point a tiered approach
(Haiku for routine queries, Sonnet as opt-in) becomes worth evaluating.

Prompt caching (`cache_control` on the system prompt) reduces input costs by ~90% on the
cached portion. The system prompt (~150 tokens) is a natural cache candidate.

Full LLM provider ladder:
```
LLM_PROVIDER=ollama    # OSS default — free, local, no API key
LLM_PROVIDER=haiku     # dev/testing — cheap, fast
LLM_PROVIDER=sonnet    # production default
LLM_PROVIDER=openai    # BYOK alternative for OpenAI users
```

## Consequences

- `ANTHROPIC_API_KEY` required for `haiku` and `sonnet` providers
- Prompt caching should be wired into `generate.py` for the sonnet provider to reduce costs
- Haiku remains the recommended provider for development iteration — fast and cheap enough
  for testing prompt changes without burning Sonnet budget
- Cost scales linearly with queries — no infrastructure spike risk
