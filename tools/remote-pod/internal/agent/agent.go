// Package agent is the server inside the pod (`remote-pod agent`): it runs commands (streamed, with signals and a
// terminal), moves files, keeps detached jobs and their logs, and stops the pod when it sits idle. Protocol: proto.
package agent

import (
	"crypto/subtle"
	"crypto/tls"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/gorilla/websocket"

	"github.com/SamuelHaidu/rnn-research/tools/remote-pod/internal/grant"
	"github.com/SamuelHaidu/rnn-research/tools/remote-pod/internal/proto"
	"github.com/SamuelHaidu/rnn-research/tools/remote-pod/internal/syncx"
)

type Options struct {
	Addr     string        // ":8080"
	Token    string        // required: every request carries "Authorization: Bearer <token>"
	Workdir  string        // relative paths and the default cwd
	IdleStop time.Duration // > 0: stop the pod after this long with no request and nothing running
	OnIdle   func()        // what "stop the pod" does (set by main: the provider's API from inside the pod)
	TLS      *tls.Config   // non-nil: serve HTTPS with it (a pod reached on its public IP: vast.ai, see pin)
	// GrantAddr: where processes on the pod ask for store grants (POST /v1/grant); loopback only, e.g.
	// "127.0.0.1:8090" ("" = none). Grants are the pre-issued ones (REMOTE_POD_GRANTS), the rest come from the CLI.
	GrantAddr string
	Grants    []grant.Grant
}

type Server struct {
	opt      Options
	jobs     *Jobs
	active   atomic.Int64 // execs in progress
	lastSeen atomic.Int64 // unix nanos of the last request
	upgrader websocket.Upgrader
	relay    *Relay
}

func New(opt Options) (*Server, error) {
	if opt.Token == "" {
		return nil, errors.New("agent: no token (REMOTE_POD_TOKEN)")
	}
	if opt.Workdir == "" {
		opt.Workdir = "/work"
	}
	if err := os.MkdirAll(opt.Workdir, 0o755); err != nil {
		return nil, err
	}
	s := &Server{opt: opt, jobs: NewJobs(filepath.Join(opt.Workdir, ".remote-pod", "jobs")),
		upgrader: websocket.Upgrader{ReadBufferSize: 64 << 10, WriteBufferSize: 64 << 10}, relay: NewRelay(opt.Grants)}
	s.lastSeen.Store(time.Now().UnixNano())
	return s, nil
}

func (s *Server) Jobs() *Jobs { return s.jobs }

// Handler is the HTTP API (all routes behind the token).
func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /v1/health", s.health)
	mux.HandleFunc("GET /v1/exec", s.exec)
	mux.HandleFunc("POST /v1/files", s.upload)
	mux.HandleFunc("GET /v1/files", s.download)
	mux.HandleFunc("POST /v1/manifest", s.manifest)
	mux.HandleFunc("POST /v1/delete", s.delete)
	mux.HandleFunc("POST /v1/jobs", s.startJob)
	mux.HandleFunc("GET /v1/jobs", s.listJobs)
	mux.HandleFunc("GET /v1/jobs/{id}", s.getJob)
	mux.HandleFunc("POST /v1/jobs/{id}/signal", s.signalJob)
	mux.HandleFunc("GET /v1/jobs/{id}/log", s.jobLog)
	mux.HandleFunc("GET /v1/signer", s.signerWS)
	return s.auth(mux)
}

func (s *Server) auth(next http.Handler) http.Handler {
	want := []byte("Bearer " + s.opt.Token)
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if subtle.ConstantTimeCompare([]byte(r.Header.Get("Authorization")), want) != 1 {
			http.Error(w, "unauthorized", http.StatusUnauthorized)
			return
		}
		s.lastSeen.Store(time.Now().UnixNano())
		next.ServeHTTP(w, r)
	})
}

// Serve listens and runs the idle guard.
func (s *Server) Serve() error {
	if s.opt.IdleStop > 0 {
		go s.idleGuard()
	}
	log.Printf("remote-pod agent %s on %s, workdir %s, tls %v", proto.Version, s.opt.Addr, s.opt.Workdir, s.opt.TLS != nil)
	if s.opt.GrantAddr != "" {
		host, _, err := net.SplitHostPort(s.opt.GrantAddr)
		if ip := net.ParseIP(host); err != nil || ip == nil || !ip.IsLoopback() {
			return fmt.Errorf("agent: the grant endpoint must listen on a loopback address, not %q", s.opt.GrantAddr)
		}
		ln, err := net.Listen("tcp", s.opt.GrantAddr)
		if err != nil {
			return err
		}
		go func() { log.Printf("grant endpoint: %v", http.Serve(ln, s.GrantHandler())) }()
	}
	if s.opt.TLS != nil {
		srv := &http.Server{Addr: s.opt.Addr, Handler: s.Handler(), TLSConfig: s.opt.TLS}
		return srv.ListenAndServeTLS("", "")
	}
	return http.ListenAndServe(s.opt.Addr, s.Handler())
}

func (s *Server) idleGuard() {
	for range time.Tick(30 * time.Second) {
		idle := time.Since(time.Unix(0, s.lastSeen.Load()))
		if s.active.Load() == 0 && s.jobs.Running() == 0 && idle > s.opt.IdleStop {
			log.Printf("idle for %s with nothing running: stopping the pod", idle.Round(time.Second))
			if s.opt.OnIdle != nil {
				s.opt.OnIdle()
			}
			return
		}
	}
}

// Resolve maps a request path to the filesystem: relative to the workdir, "~" to $HOME.
func (s *Server) Resolve(p string) string {
	if p == "" {
		return s.opt.Workdir
	}
	if p == "~" || strings.HasPrefix(p, "~/") {
		home, _ := os.UserHomeDir()
		return filepath.Join(home, strings.TrimPrefix(p, "~"))
	}
	if !filepath.IsAbs(p) {
		return filepath.Join(s.opt.Workdir, p)
	}
	return filepath.Clean(p)
}

func writeJSON(w http.ResponseWriter, code int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	_ = json.NewEncoder(w).Encode(v)
}

func fail(w http.ResponseWriter, code int, err error) {
	writeJSON(w, code, map[string]string{"error": err.Error()})
}

func (s *Server) health(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, 200, proto.Health{OK: true, Version: proto.Version, Workdir: s.opt.Workdir, Jobs: s.jobs.Running()})
}

// upload: a tar stream (syncx) to dest. mode=cp (default): into dest if it is a directory (or ends with "/"), else
// as dest; base names the source. mode=merge: paths relative to dest (sync).
func (s *Server) upload(w http.ResponseWriter, r *http.Request) {
	q := r.URL.Query()
	dest := s.Resolve(q.Get("dest"))
	target := dest
	if q.Get("mode") != "merge" {
		fi, err := os.Stat(dest)
		isDir := err == nil && fi.IsDir()
		target = syncx.CpTarget(dest, q.Get("base"), isDir || strings.HasSuffix(q.Get("dest"), "/"))
	}
	files, n, err := syncx.Extract(r.Body, target)
	if err != nil {
		fail(w, 400, err)
		return
	}
	writeJSON(w, 200, map[string]any{"target": target, "files": files, "bytes": n})
}

func (s *Server) download(w http.ResponseWriter, r *http.Request) {
	p := s.Resolve(r.URL.Query().Get("path"))
	fi, err := os.Stat(p)
	if err != nil {
		fail(w, 404, err)
		return
	}
	w.Header().Set("Content-Type", "application/x-tar")
	w.Header().Set("X-Base", filepath.Base(p))
	w.Header().Set("X-Dir", strconv.FormatBool(fi.IsDir()))
	if err := syncx.WriteTar(w, p, nil); err != nil {
		log.Printf("download %s: %v", p, err)
	}
}

func (s *Server) manifest(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Root     string   `json:"root"`
		Excludes []string `json:"excludes"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		fail(w, 400, err)
		return
	}
	m, err := syncx.Manifest(s.Resolve(req.Root), req.Excludes)
	if err != nil {
		fail(w, 500, err)
		return
	}
	writeJSON(w, 200, map[string]any{"files": m})
}

func (s *Server) delete(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Root  string   `json:"root"`
		Paths []string `json:"paths"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		fail(w, 400, err)
		return
	}
	root, n := s.Resolve(req.Root), 0
	for _, p := range req.Paths {
		clean := filepath.Clean(filepath.FromSlash(p))
		if filepath.IsAbs(clean) || strings.HasPrefix(clean, "..") {
			fail(w, 400, fmt.Errorf("unsafe path %q", p))
			return
		}
		if err := os.Remove(filepath.Join(root, clean)); err == nil {
			n++
		}
	}
	writeJSON(w, 200, map[string]int{"deleted": n})
}

// wsWriter serializes writes to a websocket (frames from several goroutines).
type wsWriter struct {
	mu sync.Mutex
	c  *websocket.Conn
}

func (w *wsWriter) binary(ch byte, b []byte) error {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.c.WriteMessage(websocket.BinaryMessage, append([]byte{ch}, b...))
}

func (w *wsWriter) control(c proto.Control) error {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.c.WriteJSON(c)
}

// pump copies a reader to the websocket as frames of one channel.
func pump(r io.Reader, w *wsWriter, ch byte) {
	buf := make([]byte, 32<<10)
	for {
		n, err := r.Read(buf)
		if n > 0 {
			if w.binary(ch, buf[:n]) != nil {
				return
			}
		}
		if err != nil {
			return
		}
	}
}
