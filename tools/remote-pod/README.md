# remote-pod

Work on a RunPod GPU as if it were this machine. **The manual is the tool itself:** `remote-pod --help` (an overview
and a quick start) and `remote-pod <command> --help` (what it does, its flags, real examples).

```bash
go build -C tools/remote-pod -o ~/.local/bin/remote-pod .     # install
remote-pod config                                             # what to put in ~/.config/remote-pod/config.toml
remote-pod up --gpu "RTX 4090"
remote-pod sh nvidia-smi
```

## How it works

```
your machine                                  the pod (docker/train.Dockerfile or dev.Dockerfile)
remote-pod (CLI) ── HTTPS + websocket ──▶      remote-pod agent: exec, files, jobs, logs, idle stop,
                    https://<pod>-8080.proxy.runpod.net        dataset fetch, checkpoint upload
      └── R2 (S3 API): datasets/, runs/, jobs/ ◀──┘
```

- **One binary**: the CLI here, `remote-pod agent` in the pod (the image's entrypoint).
- **No SSH**: the agent is reached through RunPod's HTTPS proxy, so community pods without a public IP work. Every
  request carries the pod's random token (made by `up` / `train`, kept in `.remote-pod/state.json`, gitignored).
- **Pods**: RunPod's GraphQL API (`internal/runpod`).
- **Datasets and checkpoints**: R2 or any S3-compatible store (`internal/store`); datasets are one tar.zst per
  dataset, streamed both ways.

| Package | What |
|---|---|
| `main.go`, `cmd_*.go` | the commands (cobra) and their help |
| `internal/agent` | the server in the pod: exec (process groups, pty), files, jobs and logs, the train runner |
| `internal/client` | the CLI's side of the protocol (`internal/proto` documents it) |
| `internal/syncx` | tar streams (with path-traversal checks) and the hash manifests of `sync` |
| `internal/store` | R2 / S3 |
| `internal/runpod` | the RunPod API |
| `internal/ui` | colours, tables, plain output when piped, `--json` |

## Tests

```bash
go test -race ./...                                     # unit + agent/client integration + an in-memory S3
remote-pod image build agent --base ubuntu:24.04 --tag remote-pod:agent-test
docker run -d --rm --name rp-agent -p 127.0.0.1:18080:8080 -e REMOTE_POD_TOKEN=localtest remote-pod:agent-test
bash tools/remote-pod/scripts/e2e_local.sh              # the CLI against that container: sh, cp, sync, jobs, logs
```
