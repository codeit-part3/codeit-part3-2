# Build
# add argument
ARG PYTHON_VERSION=3.12
# python version
FROM python:${PYTHON_VERSION}-slim

# Excute
# print log directly(not stacking in buffer)
ENV PYTHONUNBUFFERED=1
ENV PORT_NUM=8000
# install uv
RUN pip install --no-cache-dir uv>=0.13.1

# working directory
WORKDIR /app

# copy the application code into the container
COPY pyproject.toml uv.lock ./

# install application dependencies
RUN uv sync --frozen --no-dev

# copy directoriies
COPY backend/ ./backend/
COPY engines/ ./engines/
COPY indexes/ ./indexes/

# expose a port
EXPOSE ${PORT_NUM}

# specify the command
CMD exec uv run uvicorn backend.main:app --port ${PORT_NUM} --host 0.0.0.0
