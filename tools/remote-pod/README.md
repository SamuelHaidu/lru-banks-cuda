# remote-pod

Work on a rented GPU (RunPod or vast.ai) as if it were this machine. **The manual is the tool itself:** `remote-pod --help` (an overview
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
- **Pods**: RunPod's GraphQL API (`internal/runpod`) or vast.ai's REST API (`internal/vast`), behind one set of
  commands (`cmd_provider.go`): `--provider vast` on `up`, `gpus`, `train`, or `provider = "vast"` in the settings;
  `status`, `stop`, `down` follow the pod's own provider; `balance` shows both.
- **vast.ai**: no proxy, so the agent's port is published on the host's public IP. `up` makes a self-signed
  certificate for each pod (`internal/pin`): the key goes to the pod with the token, the CLI pins the certificate
  (state.json) and talks TLS 1.3 to that certificate only. vast.ai hosts cannot pull our private images, so the pod
  runs a public CUDA image (`--image cuda`); its onstart script restores the container's environment from
  `/proc/1/environ` and runs the bootstrap, which fetches the agent (and `lbc`, when `engine/build/dist/lbc` exists)
  from R2 through presigned URLs. The agent stops its own instance when idle with the key vast.ai gives every
  container (`CONTAINER_ID`, `CONTAINER_API_KEY`).
- **Datasets and checkpoints**: R2 or any S3-compatible store (`internal/store`); datasets are one tar.zst per
  dataset, streamed both ways.
- **No key on a pod** (`internal/grant`): the store key stays here. A pod gets presigned URLs, one object and one
  operation each, signed here when its policy and limits allow (`remote-pod signer`, or while `train` / `logs -f`
  follow): its inputs, uploads under its own prefixes, sized, paced by its run's plan; a few are pre-issued at
  creation for when no signer runs. Outside HTTPS trusts only embedded roots (`internal/tlsroots`). The threat model
  and the tests that back it: `security/README.md` (`go test ./security/`).

| Package | What |
|---|---|
| `main.go`, `cmd_*.go` | the commands (cobra) and their help |
| `internal/agent` | the server in the pod: exec (process groups, pty), files, jobs and logs, the train runner |
| `internal/client` | the CLI's side of the protocol (`internal/proto` documents it) |
| `internal/syncx` | tar streams (with path-traversal checks) and the hash manifests of `sync` |
| `internal/store` | R2 / S3 |
| `internal/runpod` | the RunPod API |
| `internal/vast` | the vast.ai API (offers, instances, credit) |
| `internal/grant` | the pod's store access without a key: policy, limits, upload plans, signer, journal |
| `internal/tlsroots` | the embedded public roots every outside HTTPS connection trusts |
| `internal/sshkey` | the throwaway SSH key of one connection (`remote-pod ssh`) |
| `internal/boot` | the boot script on public images (sha256-checked agent and lbc) |
| `security/` | the security tests (threat model in its README) |
| `internal/pin` | the per-pod TLS certificate (vast.ai) |
| `internal/ui` | colours, tables, plain output when piped, `--json` |

## Tests

```bash
go test -race ./...                                     # unit + agent/client integration + an in-memory S3
remote-pod image build agent --base ubuntu:24.04 --tag remote-pod:agent-test
docker run -d --rm --name rp-agent -p 127.0.0.1:18080:8080 -e REMOTE_POD_TOKEN=localtest remote-pod:agent-test
bash tools/remote-pod/scripts/e2e_local.sh              # the CLI against that container: sh, cp, sync, jobs, logs
VAST_LIVE=1 go test ./internal/vast/ -run TestLive -v   # vast.ai, read-only: credit, offers, instances
```

The vast.ai boot path runs locally too: `remote-pod bootstrap-env FILE`, the pod's TLS certificate and token appended to
FILE (`REMOTE_POD_TLS_CERT` / `_KEY` as base64 PEM), `docker run --env-file FILE ... sleep infinity`, then the onstart
command (`vastOnstart` in bootstrap.go) through `docker exec ... env -i bash -c` so it must restore the environment the
way it does on vast.ai.
