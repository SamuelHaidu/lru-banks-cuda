# Security tests

`go test ./security/` (from `tools/remote-pod`). One more, against the real R2 bucket, runs with `SECURITY_LIVE=1`.

## Threat model

A rented GPU's host is a stranger with root on the machine (vast.ai is a marketplace; RunPod's community cloud too).
It can read and change everything inside our container: its environment, files, memory, GPU memory, and anything sent
in the clear. It can also impersonate the container (it holds the pod's TLS key and token, by definition). So:

- **Nothing on a pod may unlock more than that pod's own work.** No store key, no provider account key, no deploy
  key. The store key stays on this machine; a pod gets presigned URLs, one object and one operation each, signed on
  this machine only when the pod's policy allows it (package `internal/grant`, `remote-pod signer`).
- **Nothing a pod sends back may reach outside what was asked.** Downloads to this machine (`pull`, `cp :`, dataset
  archives) never write outside their target directory.
- **Nothing on the wire is readable by others.** The CLI talks to the agent over RunPod's HTTPS proxy, or (vast.ai,
  where the port is public) TLS 1.3 pinned to a certificate made for that pod; every request carries the pod's token.
  Every other HTTPS connection (the store, RunPod, vast.ai, the pod's own uploads) trusts only Mozilla's roots
  embedded in the binary (`internal/tlsroots`), never the operating system's store: a CA injected there (a corporate
  TLS inspector, malware, a host planting one in the container) cannot intercept.
- **Even allowed requests follow a protocol.** An upload declares its size and the URL is signed for exactly that
  length; per-file, per-day and object-count budgets; rates per minute, per hour and per object; and for a training
  run, an upload plan derived from its config (how many checkpoints, how often at the fastest a GPU could train). A
  pod that keeps being refused is suspended. Every decision is journaled on this machine
  (`.remote-pod/grants-<tag>.jsonl`), which also keeps the budgets across signer restarts.
- **Nothing identifies you to a host.** No SSH key in a provider account (they copy those into every machine);
  `remote-pod ssh` makes a nameless ed25519 pair for each connection, attaches it to that one instance, offers it
  alone, then detaches it and deletes the private half. No code reads `~/.ssh`.
- **Secrets are unpredictable.** Every token, tag, key seed and certificate serial comes from `internal/entropy`: the
  CPU's thermal-noise generator (RDSEED, health-checked), the TPM (`/dev/hwrng`, when readable) and the kernel's
  CSPRNG, hashed with SHA-512, hardware read before the kernel's bytes; unpredictable if any one source is. Keys are
  derived from those seeds (Ed25519), never from a random source a library might ignore.
- **What a pod runs is what we built.** The agent and `lbc` are downloaded through presigned URLs and refused unless
  their sha256 matches the one fixed when the pod was created.

The worst a hostile host can then do: fake its own results, overwrite its own run's files within its plan, read the
public datasets it was given, burn the credit of the instance it rents to us.

What this does not do, by design: protect anything on the pod from its host. The host controls the kernel and the
hardware (RAM, GPU memory, the disk); pinning and checks inside the pod only make other attackers' lives harder. The
defence against the host is that the pod holds nothing worth taking. Against an adversary on this machine (the one
place the store key, the provider keys and the signer live) the protection is this machine's own: disk encryption,
the settings at mode 0600, no secrets pasted into a pod's shell.

## What each test shows

| File | Shows |
|---|---|
| `policy_test.go` | The policy grants only a pod's own inputs (exact keys) and uploads under its own prefixes; ~30 hostile requests (another run, prefix confusion, `..`, `%2e%2e`, backslash, unicode look-alikes, NUL, list, delete, ...) are refused, and none of them reaches the presigner. A fuzz test checks the same property on random keys. The rate limit holds. |
| `abuse_test.go` | The protocol beyond the policy: size declared and capped, the 24-hour byte budget, the object cap, one object hammered, rates, suspension after repeated refusals (and recovery), budgets rebuilt from the journal after a restart, a run's plan refusing checkpoints rewritten faster than any GPU trains or more of them than the run writes, and an honest 95-epoch run at our real pace never refused. |
| `relay_test.go` | The whole path: a pod process asks the agent on loopback, the agent asks the CLI over the authenticated websocket, the CLI checks the policy and signs. Refused requests never get a URL; without a signer the pod gets 503 (it waits), not a key; non-loopback callers and callers without the token are refused. A pod with no key still uploads and downloads through grants. |
| `podenv_test.go` | The real CLI binary, run against fake RunPod / vast.ai / S3 servers: for `up` and `train` on both providers, the create request a provider receives (the pod's environment, onstart, every pre-issued grant, base64 decoded) contains no store secret key and no provider account key, and every pre-issued URL is for a key the pod's policy allows. |
| `paths_test.go` | Store keys and tar entries crafted to escape (`../`, absolute, backslash, symlink then a file through it, hard links) never write outside the target directory. |
| `boot_test.go` | The boot script runs a downloaded agent only when its sha256 matches; a tampered agent or `lbc` is deleted and nothing runs. |
| `transport_test.go` | Every agent route refuses a missing or wrong token; the pinned client refuses an interceptor (even one with a certificate for the pinned name from a CA this machine trusts), another pod's certificate, and an unpinned client refuses the pod's self-signed one. With a "corporate inspection" CA injected into the system store (a child process with `SSL_CERT_FILE`), a client on the system store is intercepted, ours is not. The token is never sent over plain http except to this machine. A static check finds no code turning verification off or using the system's trust. |
| `sshkey_test.go` | Every connection's key is new, ed25519, with no comment, 0600 in a 0700 directory, deleted afterwards; `ssh -G` (the client's own reading) shows only that key offered, no agent, nothing forwarded, no password prompt, the pod's own known_hosts, `~/.ssh/config` ignored; no code reads the user's SSH keys. |
| `entropy_test.go` | The CPU's hardware generator is really used where present; 1 MB of output passes monobit, byte chi-square, no repeated block and incompressibility checks; 200,000 tokens never collide; no code imports `crypto/rand` or `math/rand` outside the mixer or calls a `GenerateKey`. (`internal/entropy/health_test.go`: a stuck, all-ones, all-zeros or never-ready CPU source is dropped.) |
| `live_r2_test.go` | (`SECURITY_LIVE=1`) Against real R2, through the embedded roots: a sized upload URL takes exactly its size (not longer, not shorter), for its key only; it cannot read, delete or write another key; a download URL cannot write or read another key; an expired URL does nothing. |
