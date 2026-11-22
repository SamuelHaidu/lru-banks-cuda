# The pod image for training: the agent (entrypoint), lbc (the engine), uv (Python on demand), zstd.
# Build: remote-pod build && remote-pod image build train   (context: the repository root, see .dockerignore)
# The base is an argument so the agent can be tested on a plain image: --build-arg BASE=ubuntu:24.04 --target agent
ARG BASE=nvidia/cuda:12.8.1-runtime-ubuntu24.04
FROM ${BASE} AS agent
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates zstd \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.9 /uv /uvx /usr/local/bin/
COPY tools/remote-pod/dist/remote-pod /usr/local/bin/remote-pod
ENV REMOTE_POD_WORKDIR=/work
WORKDIR /work
EXPOSE 8080
ENTRYPOINT ["remote-pod", "agent"]

FROM agent AS train
COPY engine/build/dist/lbc /usr/local/bin/lbc
