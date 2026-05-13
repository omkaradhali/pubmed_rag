# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | ✓         |

## Reporting a Vulnerability

If you discover a security vulnerability in pubmed-rag, please report it **privately** by emailing **omkar.a2989@gmail.com** with the subject line:

```
[pubmed-rag] Security Vulnerability
```

Please include:
- A description of the vulnerability and its potential impact
- Steps to reproduce the issue
- Any proof-of-concept code (if applicable)

You can expect an acknowledgement within **48 hours** and a resolution timeline within **14 days** for confirmed vulnerabilities.

**Please do not open a public GitHub issue for security vulnerabilities.**

---

## Security Considerations

**API keys**
Never commit `.env` files. `.gitignore` excludes `.env` and `detect-secrets` runs as a pre-commit hook to prevent accidental key exposure.

**CORS**
The default `CORS_ORIGINS` setting (`http://localhost:3000,http://localhost:5173`) restricts browser access to localhost. Set `CORS_ORIGINS` to your deployment domain before exposing the API publicly. Do not use `*` in production.

**NCBI rate limits**
Respect NCBI's E-utilities rate limits (3 req/sec without an API key, 10 req/sec with one). Exceeding these limits violates the NCBI terms of service.

**Data sensitivity**
pubmed-rag processes publicly available PubMed abstracts. Do not ingest or store non-public clinical data through this pipeline without appropriate data governance controls.
