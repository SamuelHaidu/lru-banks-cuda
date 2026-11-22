# The pod image for CPU-only pods (remote-pod up --cpu N): data work without a GPU (download, clean, train
# tokenizers, compress datasets, then remote-pod dataset add). Ubuntu + Python + uv + zstd + git + the agent.
FROM ubuntu:24.04
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates zstd git curl python3 python3-venv \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.9 /uv /uvx /usr/local/bin/
COPY tools/remote-pod/dist/remote-pod /usr/local/bin/remote-pod
ENV REMOTE_POD_WORKDIR=/work
WORKDIR /work
EXPOSE 8080
ENTRYPOINT ["remote-pod", "agent"]
