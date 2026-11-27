package agent

// The pod's way to the object store without a key (package grant): presigned URLs, pre-issued at creation
// (REMOTE_POD_GRANTS) or signed on demand by the CLI over the /v1/signer websocket. Processes on the pod ask the
// agent on 127.0.0.1 (REMOTE_POD_GRANT_URL); the agent itself uses the same Relay for datasets, checkpoints and logs.

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"sync"
	"sync/atomic"
	"time"

	"github.com/gorilla/websocket"
	"github.com/klauspost/compress/zstd"

	"github.com/SamuelHaidu/rnn-research/tools/remote-pod/internal/grant"
	"github.com/SamuelHaidu/rnn-research/tools/remote-pod/internal/proto"
	"github.com/SamuelHaidu/rnn-research/tools/remote-pod/internal/store"
	"github.com/SamuelHaidu/rnn-research/tools/remote-pod/internal/syncx"
	"github.com/SamuelHaidu/rnn-research/tools/remote-pod/internal/tlsroots"
)

// ErrNoSigner: nothing pre-issued covers the request and no CLI is connected to sign it (the caller retries later).
var ErrNoSigner = errors.New("no signer connected (run remote-pod signer, or follow the run / logs)")

// Relay hands out grants: a pre-issued one while it is valid for at least another minute, else the connected CLI's.
type Relay struct {
	mu      sync.Mutex
	cached  map[grant.Request]grant.Grant
	conn    *websocket.Conn
	wmu     sync.Mutex // one writer on conn
	pending map[uint64]chan proto.SignReply
	next    atomic.Uint64
	Timeout time.Duration // how long to wait for the CLI's answer (default 30 s)
}

func NewRelay(preissued []grant.Grant) *Relay {
	r := &Relay{cached: map[grant.Request]grant.Grant{}, pending: map[uint64]chan proto.SignReply{}}
	for _, g := range preissued {
		r.cached[grant.Request{Op: g.Op, Key: g.Key, Size: g.Size}] = g
	}
	return r
}

// Grant returns a URL for the request.
func (r *Relay) Grant(ctx context.Context, req grant.Request) (grant.Grant, error) {
	if err := grant.CleanKey(req.Key); err != nil {
		return grant.Grant{}, err
	}
	r.mu.Lock()
	if g, ok := r.cached[req]; ok && time.Until(g.Expires) > time.Minute {
		r.mu.Unlock()
		return g, nil
	}
	// a grant pre-issued at creation for this key (no size: the CLI chose the key itself, S3 caps a request at 5 GB)
	if g, ok := r.cached[grant.Request{Op: req.Op, Key: req.Key}]; ok && req.Op == grant.Put && g.Size == 0 &&
		time.Until(g.Expires) > time.Minute {
		r.mu.Unlock()
		return g, nil
	}
	conn := r.conn
	if conn == nil {
		r.mu.Unlock()
		return grant.Grant{}, ErrNoSigner
	}
	id := r.next.Add(1)
	ch := make(chan proto.SignReply, 1)
	r.pending[id] = ch
	r.mu.Unlock()
	defer func() {
		r.mu.Lock()
		delete(r.pending, id)
		r.mu.Unlock()
	}()
	r.wmu.Lock()
	err := conn.WriteJSON(proto.SignRequest{ID: id, Op: req.Op, Key: req.Key, Size: req.Size})
	r.wmu.Unlock()
	if err != nil {
		return grant.Grant{}, ErrNoSigner
	}
	timeout := r.Timeout
	if timeout <= 0 {
		timeout = 30 * time.Second
	}
	select {
	case rep := <-ch:
		if rep.Error != "" {
			return grant.Grant{}, errors.New(rep.Error)
		}
		g := grant.Grant{Op: req.Op, Key: req.Key, Size: req.Size, URL: rep.URL, Expires: rep.Expires}
		r.mu.Lock()
		r.cached[req] = g
		r.mu.Unlock()
		return g, nil
	case <-time.After(timeout):
		return grant.Grant{}, ErrNoSigner
	case <-ctx.Done():
		return grant.Grant{}, ctx.Err()
	}
}

// Connected reports whether a CLI is signing.
func (r *Relay) Connected() bool {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.conn != nil
}

// serve takes a CLI's signer connection (the newest replaces an older one) and routes its replies until it closes.
func (r *Relay) serve(conn *websocket.Conn) {
	r.mu.Lock()
	old := r.conn
	r.conn = conn
	r.mu.Unlock()
	if old != nil {
		_ = old.Close()
	}
	defer func() {
		r.mu.Lock()
		if r.conn == conn {
			r.conn = nil
		}
		r.mu.Unlock()
		_ = conn.Close()
	}()
	for {
		var rep proto.SignReply
		if err := conn.ReadJSON(&rep); err != nil {
			return
		}
		r.mu.Lock()
		ch := r.pending[rep.ID]
		r.mu.Unlock()
		if ch != nil {
			ch <- rep
		}
	}
}

// signerWS is WS /v1/signer (behind the token like every route).
func (s *Server) signerWS(w http.ResponseWriter, req *http.Request) {
	conn, err := s.upgrader.Upgrade(w, req, nil)
	if err != nil {
		return
	}
	s.relay.serve(conn)
}

// GrantHandler is the pod-local endpoint: POST /v1/grant {"op", "key"} -> grant.Grant. It answers loopback clients
// only (it also listens on 127.0.0.1 only): the grants it relays are still bounded by the CLI's policy.
func (s *Server) GrantHandler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/grant", func(w http.ResponseWriter, req *http.Request) {
		host, _, err := net.SplitHostPort(req.RemoteAddr)
		if ip := net.ParseIP(host); err != nil || ip == nil || !ip.IsLoopback() {
			http.Error(w, `{"error": "loopback only"}`, http.StatusForbidden)
			return
		}
		var gr grant.Request
		if err := json.NewDecoder(io.LimitReader(req.Body, 4096)).Decode(&gr); err != nil {
			writeJSON(w, http.StatusBadRequest, map[string]string{"error": "body: " + err.Error()})
			return
		}
		g, err := s.relay.Grant(req.Context(), gr)
		switch {
		case errors.Is(err, ErrNoSigner):
			writeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": err.Error()})
		case err != nil:
			writeJSON(w, http.StatusForbidden, map[string]string{"error": err.Error()})
		default:
			writeJSON(w, http.StatusOK, g)
		}
	})
	return mux
}

// Relay is the server's grant relay (pre-issued grants come from Options.Grants).
func (s *Server) Relay() *Relay { return s.relay }

// Objects is what the agent does with the store: put and get files, get a dataset archive.
type Objects interface {
	PutFile(ctx context.Context, key, path string) error
	GetFile(ctx context.Context, key, path string) error
	GetDir(ctx context.Context, key, dir string, progress func(int64)) (int, int64, error)
}

// GrantObjects does it with presigned URLs from a Relay: no key on the pod.
type GrantObjects struct {
	Relay *Relay
	HTTP  *http.Client
}

func (g GrantObjects) client() *http.Client {
	if g.HTTP != nil {
		return g.HTTP
	}
	return tlsroots.Client(0) // the pod's uploads too: a CA planted in the container image is not trusted
}

func (g GrantObjects) PutFile(ctx context.Context, key, path string) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()
	fi, err := f.Stat()
	if err != nil {
		return err
	}
	// the URL is signed for exactly this size: the request body below is limited to it as well
	gr, err := g.Relay.Grant(ctx, grant.Request{Op: grant.Put, Key: key, Size: fi.Size()})
	if err != nil {
		return err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPut, gr.URL, io.LimitReader(f, fi.Size()))
	if err != nil {
		return err
	}
	req.ContentLength = fi.Size()
	resp, err := g.client().Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 300 {
		b, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		return fmt.Errorf("upload %s: HTTP %d: %s", key, resp.StatusCode, b)
	}
	return nil
}

func (g GrantObjects) open(ctx context.Context, key string) (io.ReadCloser, error) {
	gr, err := g.Relay.Grant(ctx, grant.Request{Op: grant.Get, Key: key})
	if err != nil {
		return nil, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, gr.URL, nil)
	if err != nil {
		return nil, err
	}
	resp, err := g.client().Do(req)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode >= 300 {
		resp.Body.Close()
		return nil, fmt.Errorf("download %s: HTTP %d", key, resp.StatusCode)
	}
	return resp.Body, nil
}

func (g GrantObjects) GetFile(ctx context.Context, key, path string) error {
	body, err := g.open(ctx, key)
	if err != nil {
		return err
	}
	defer body.Close()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	tmp := path + ".part"
	f, err := os.Create(tmp)
	if err != nil {
		return err
	}
	if _, err := io.Copy(f, body); err != nil {
		f.Close()
		return err
	}
	if err := f.Close(); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}

func (g GrantObjects) GetDir(ctx context.Context, key, dir string, progress func(int64)) (int, int64, error) {
	body, err := g.open(ctx, key)
	if err != nil {
		return 0, 0, err
	}
	defer body.Close()
	zr, err := zstd.NewReader(body)
	if err != nil {
		return 0, 0, err
	}
	defer zr.Close()
	return syncx.Extract(zr, dir)
}

// StoreObjects does it with a store key (a local agent, tests): never what remote-pod gives a pod.
type StoreObjects struct{ St *store.Store }

func (s StoreObjects) PutFile(ctx context.Context, key, path string) error {
	return s.St.PutFile(ctx, key, path)
}
func (s StoreObjects) GetFile(ctx context.Context, key, path string) error {
	return s.St.GetFile(ctx, key, path)
}
func (s StoreObjects) GetDir(ctx context.Context, key, dir string, progress func(int64)) (int, int64, error) {
	return s.St.GetDir(ctx, key, dir, progress)
}
