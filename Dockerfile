# frontend
FROM python:3.12-slim

# install uv
COPY --from=ghcr.io/astral-sh/uv:0.13.1 /uv /uvx /bin/

# print log directly(not stacking in buffer)
ENV PYTHONUNBUFFERED=1
# not to write *.pyc
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY pyproject.toml uv.lock ./

# hf-pull
FROM base AS hf-pull

RUN uv sync --frozen --no-dev --no-cache --group hf

CMD ["uv", "run", "--no-sync" "python" "-c" "from huggingface_hub import snapshot_download; \
    snapshot_download('BAAI/bge-reranker-v2-m3')"]


# frontend
FROM base AS frontend

RUN uv sync --frozen --no-dev --no-cache --group frontend

COPY frontend.py ./frontend.py
COPY style.css ./style.css
COPY progress.css ./progress.css

EXPOSE 7860

CMD ["uv", "run", "--no-sync", "python", "frontend.py"]
