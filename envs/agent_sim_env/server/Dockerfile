ARG BASE_IMAGE=openenv-base:latest
FROM ${BASE_IMAGE} AS builder
WORKDIR /app/env
COPY . /app/env
RUN --mount=type=cache,target=/root/.cache/uv uv sync --no-install-project --no-editable

FROM ${BASE_IMAGE}
WORKDIR /app
COPY --from=builder /app/env /app/env
ENV PYTHONPATH="/app/env:$PYTHONPATH"
CMD ["sh", "-c", "cd /app/env && uvicorn server.app:app --host 0.0.0.0 --port 8000"]
