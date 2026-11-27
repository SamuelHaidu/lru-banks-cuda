// Package proto is the CLI <-> agent protocol (HTTP + websockets, bearer token on every request).
//
//	GET  /v1/health                      {"ok": true, "version": ..., "workdir": ...}
//	WS   /v1/exec                        run a command (below)
//	POST /v1/files?dest=PATH&mode=M      body: a tar stream (see files.go); mode "cp" or "merge" (sync)
//	GET  /v1/files?path=PATH             a tar stream of a file or directory
//	POST /v1/manifest?root=PATH          {"files": {"rel/path": "sha256"}} of what is under root
//	POST /v1/jobs                        start a detached command: ExecStart -> Job
//	GET  /v1/jobs                        [Job]
//	GET  /v1/jobs/{id}                   Job
//	WS   /v1/jobs/{id}/log?offset=N      the job's output from byte N, followed until it ends
//	WS   /v1/signer                      the CLI signs store grants for the pod (below; see package grant)
//
// Signer websocket: the CLI connects and waits; the agent sends SignRequest (text, JSON) whenever something on the
// pod needs a presigned URL, the CLI answers SignReply with the same ID (a grant, or the policy's refusal). The pod
// itself never holds a store key. Inside the pod the agent serves POST /v1/grant {"op", "key"} on 127.0.0.1 only
// (REMOTE_POD_GRANT_URL), answering with the grant or an error.
//
// Exec websocket: the client sends an ExecStart (text, JSON), then binary frames [ChanStdin]+bytes and text Control
// messages (stdin EOF, a signal, a terminal resize). The agent sends binary frames [ChanStdout|ChanStderr]+bytes and,
// last, a text Control {"type": "exit", "code": N}. Job logs: binary [ChanStdout]+bytes, then {"type": "end", "code": N}.
package proto

import "time"

const Version = "0.1.0"

const (
	ChanStdin  byte = 0
	ChanStdout byte = 1
	ChanStderr byte = 2
)

type ExecStart struct {
	Command string            `json:"command"` // run with bash -lc
	Cwd     string            `json:"cwd,omitempty"`
	Env     map[string]string `json:"env,omitempty"`
	TTY     bool              `json:"tty,omitempty"`
	Rows    uint16            `json:"rows,omitempty"`
	Cols    uint16            `json:"cols,omitempty"`
	Name    string            `json:"name,omitempty"` // jobs: a name for the job (default: its id)
}

type Control struct {
	Type   string `json:"type"`             // stdin_eof, signal, resize, exit, end, error
	Signal string `json:"signal,omitempty"` // INT, TERM, KILL
	Rows   uint16 `json:"rows,omitempty"`
	Cols   uint16 `json:"cols,omitempty"`
	Code   int    `json:"code,omitempty"`
	Error  string `json:"error,omitempty"`
}

type Job struct {
	ID       string    `json:"id"`
	Name     string    `json:"name"`
	Command  string    `json:"command"`
	Cwd      string    `json:"cwd"`
	Started  time.Time `json:"started"`
	Ended    time.Time `json:"ended,omitzero"`
	Running  bool      `json:"running"`
	ExitCode int       `json:"exit_code"`
	LogBytes int64     `json:"log_bytes"`
}

type Health struct {
	OK      bool   `json:"ok"`
	Version string `json:"version"`
	Workdir string `json:"workdir"`
	Jobs    int    `json:"jobs_running"`
}

// SignRequest asks the CLI for a grant (Op "get" or "put", one object Key).
type SignRequest struct {
	ID   uint64 `json:"id"`
	Op   string `json:"op"`
	Key  string `json:"key"`
	Size int64  `json:"size,omitempty"` // an upload's exact size (the URL is signed for it)
}

// SignReply answers a SignRequest: URL valid until Expires, or Error.
type SignReply struct {
	ID      uint64    `json:"id"`
	URL     string    `json:"url,omitempty"`
	Expires time.Time `json:"expires,omitzero"`
	Error   string    `json:"error,omitempty"`
}
