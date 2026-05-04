FROM python:3.12-slim

# libgomp1 is required by PyTorch at runtime for OpenMP parallelism
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# WORKDIR must be the project root. api/ is not an installed package — it is imported
# as 'from api.config import ...' and requires the working directory to be /app.
WORKDIR /app

# uv is used instead of pip to respect [tool.uv.sources] in pyproject.toml.
# 'uv sync' (unlike 'uv pip install') reads [tool.uv.sources] and installs
# CPU-only torch (~181 MB) instead of the default CUDA build (~506 MB).
RUN pip install --no-cache-dir uv

# Copy dependency files and LICENSE before installing.
# LICENSE is required by hatchling to build the pubmed-rag wheel.
COPY pyproject.toml uv.lock LICENSE README.md ./
COPY src/ ./src/

# Install from the lock file into a venv.
# --frozen: use exact lockfile versions (no re-resolution).
# --no-dev: skip pytest, ruff, pre-commit — not needed at runtime.
RUN uv sync --frozen --no-dev

# Pre-download the default embedding model into the image so the container starts instantly.
# Adds ~90 MB to the image but eliminates a HuggingFace network call at runtime.
# When EMBEDDING_PROVIDER=openai or medcpt, this model is unused but still cached harmlessly.
RUN .venv/bin/python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Copy application code last — this layer rebuilds on code changes, not dep changes.
COPY api/ ./api/

# data/ is excluded via .dockerignore and bind-mounted at runtime.
# Create the directory so the path exists if the volume mount is not used.
RUN mkdir -p data/chroma_db

# Add venv binaries to PATH so uvicorn, python, etc. resolve to the venv.
ENV PATH="/app/.venv/bin:$PATH"

# Skip the HuggingFace Hub version check at startup.
# The model is pre-cached in the image — the check only adds network latency.
# Remove this if you want sentence-transformers to auto-update the model on startup.
ENV HF_HUB_OFFLINE=1

EXPOSE 8001

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8001"]
