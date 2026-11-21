# Build
# add argument
ARG PYTHON_VERSION=3.12
# python version
FROM python:${PYTHON_VERSION}-slim

# install uv
COPY --from=ghcr.io/astral-sh/uv:0.13.1 /uv /uvx /bin/

# Excute
# print log directly(not stacking in buffer)
ENV PYTHONUNBUFFERED=1
# not to write *.pyc
ENV PYTHONDONTWRITEBYTECODE=1

# working directory
WORKDIR /app

# copy the application code into the container
COPY pyproject.toml uv.lock ./

# install application dependencies
RUN uv sync --frozen --no-dev --no-cache

# copy directoriies
COPY backend/ ./backend/
COPY engines/ ./engines/

# expose a port
EXPOSE 8000

# specify the command
CMD ["uv", "run", "--no-sync", "uvicorn", "backend.main:app", "--port", "8000", "--host", "0.0.0.0"]
