package security

// The grant path end to end: a process on the pod asks the agent (loopback), the agent asks the CLI (the
// authenticated websocket the CLI opened), the CLI checks the pod's policy and signs. Real agent, real client.

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/johannesboyne/gofakes3"
	"github.com/johannesboyne/gofakes3/backend/s3mem"

	"github.com/SamuelHaidu/rnn-research/tools/remote-pod/internal/agent"
	"github.com/SamuelHaidu/rnn-research/tools/remote-pod/internal/client"
	"github.com/SamuelHaidu/rnn-research/tools/remote-pod/internal/grant"
	"github.com/SamuelHaidu/rnn-research/tools/remote-pod/internal/store"
)

type rig struct {
	srv    *agent.Server
	local  *httptest.Server // the pod-local grant endpoint
	fp     *fakePresigner
	cancel context.CancelFunc
}

// newRig starts an agent and, unless noSigner, a CLI signer with the test policy connected to it.
func newRig(t *testing.T, ps grant.Presigner, noSigner bool) *rig {
	t.Helper()
	srv, err := agent.New(agent.Options{Token: "the-pods-token", Workdir: t.TempDir()})
	if err != nil {
		t.Fatal(err)
	}
	ts := httptest.NewServer(srv.Handler())
	t.Cleanup(ts.Close)
	local := httptest.NewServer(srv.GrantHandler()) // httptest listens on 127.0.0.1: a loopback caller
	t.Cleanup(local.Close)
	r := &rig{srv: srv, local: local}
	if fp, ok := ps.(*fakePresigner); ok {
		r.fp = fp
	}
	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)
	r.cancel = cancel
	if !noSigner {
		s := &grant.Signer{Policy: testPolicy(t), Store: ps, PerMinute: 1000}
		go client.New(ts.URL, "the-pods-token").ServeSigner(ctx, s.Sign, nil)
		waitFor(t, srv.Relay().Connected)
	}
	return r
}

func waitFor(t *testing.T, cond func() bool) {
	t.Helper()
	for i := 0; i < 200; i++ {
		if cond() {
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatal("timed out")
}

// ask is what a pod process does: POST /v1/grant on the agent's loopback endpoint.
func (r *rig) ask(t *testing.T, op, key string) (int, map[string]any) {
	t.Helper()
	req := map[string]any{"op": op, "key": key}
	if op == grant.Put {
		req["size"] = 1000
	}
	body, _ := json.Marshal(req)
	resp, err := http.Post(r.local.URL+"/v1/grant", "application/json", bytes.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	var out map[string]any
	_ = json.NewDecoder(resp.Body).Decode(&out)
	return resp.StatusCode, out
}

func TestPodGetsGrantsOnlyWithinItsPolicy(t *testing.T) {
	r := newRig(t, &fakePresigner{}, false)
	for _, a := range allowed {
		code, out := r.ask(t, a.Op, a.Key)
		if code != http.StatusOK || !strings.Contains(fmt.Sprint(out["url"]), a.Key) {
			t.Errorf("%s %s: HTTP %d %v", a.Op, a.Key, code, out)
		}
	}
	before := len(r.fp.signed())
	for name, h := range hostile {
		code, out := r.ask(t, h.Op, h.Key)
		if code == http.StatusOK || out["url"] != nil {
			t.Errorf("%s: HTTP %d %v", name, code, out)
		}
	}
	if after := r.fp.signed(); len(after) != before {
		t.Fatalf("hostile requests reached the presigner: %v", after[before:])
	}
}

// Without a signer the pod is told to wait (503), never handed anything else; pre-issued grants still work.
func TestWithoutSignerThePodWaits(t *testing.T) {
	r := newRig(t, &fakePresigner{}, true)
	if code, out := r.ask(t, grant.Put, "runs/my-run/model.lbc"); code != http.StatusServiceUnavailable || out["url"] != nil {
		t.Fatalf("no signer: HTTP %d %v", code, out)
	}
}

// Only loopback callers: the endpoint refuses a request from any other address (another machine, another tenant).
func TestGrantEndpointIsLoopbackOnly(t *testing.T) {
	r := newRig(t, &fakePresigner{}, false)
	h := r.srv.GrantHandler()
	for _, addr := range []string{"203.0.113.7:4444", "10.0.0.2:1", "[2001:db8::1]:80", "172.17.0.1:5555"} {
		req := httptest.NewRequest("POST", "/v1/grant", strings.NewReader(`{"op":"put","key":"runs/my-run/model.lbc"}`))
		req.RemoteAddr = addr
		w := httptest.NewRecorder()
		h.ServeHTTP(w, req)
		if w.Code != http.StatusForbidden {
			t.Errorf("from %s: HTTP %d, want 403", addr, w.Code)
		}
	}
	if err := agentWithGrantAddr(t, "0.0.0.0:0"); err == nil {
		t.Error("the agent accepted to serve grants on every interface")
	}
}

func agentWithGrantAddr(t *testing.T, addr string) error {
	srv, err := agent.New(agent.Options{Token: "x", Workdir: t.TempDir(), Addr: "127.0.0.1:0", GrantAddr: addr})
	if err != nil {
		return err
	}
	errc := make(chan error, 1)
	go func() { errc <- srv.Serve() }()
	select {
	case err := <-errc:
		return err
	case <-time.After(500 * time.Millisecond):
		return nil
	}
}

// The policy holds whatever the pod claims: the agent forwards requests, but the decision is the CLI's.
func TestAPodCannotWidenItsPolicy(t *testing.T) {
	r := newRig(t, &fakePresigner{}, false)
	// a pod could send anything to the signer websocket's peer; it only ever gets answers to what it asks, decided here
	for _, key := range []string{"runs/other-run/model.lbc", "datasets/fw/data.tar.zst", "bin/remote-pod-x"} {
		if code, _ := r.ask(t, grant.Put, key); code == http.StatusOK {
			t.Errorf("put %s granted", key)
		}
	}
}

// With no key anywhere on the pod, its uploads and downloads still work, through grants, against an S3 server.
func TestKeylessPodStillMovesItsFiles(t *testing.T) {
	backend := s3mem.New()
	if err := backend.CreateBucket("sec-test"); err != nil {
		t.Fatal(err)
	}
	s3 := httptest.NewServer(gofakes3.New(backend).Server())
	defer s3.Close()
	st, err := store.New(store.Config{Endpoint: s3.URL, Bucket: "sec-test", AccessKey: "AKTEST", SecretKey: "secret-on-this-machine"})
	if err != nil {
		t.Fatal(err)
	}
	ctx := context.Background()
	if err := st.PutBytes(ctx, "runs/base/model.lbc", []byte("base weights")); err != nil {
		t.Fatal(err)
	}
	r := newRig(t, st, false)
	objs := agent.GrantObjects{Relay: r.srv.Relay(), HTTP: s3.Client()}
	dir := t.TempDir()
	if err := objs.GetFile(ctx, "runs/base/model.lbc", filepath.Join(dir, "in.lbc")); err != nil {
		t.Fatalf("download through a grant: %v", err)
	}
	if b, _ := os.ReadFile(filepath.Join(dir, "in.lbc")); string(b) != "base weights" {
		t.Fatalf("downloaded %q", b)
	}
	_ = os.WriteFile(filepath.Join(dir, "out.lbc"), []byte("new weights"), 0o644)
	if err := objs.PutFile(ctx, "runs/my-run/model.lbc", filepath.Join(dir, "out.lbc")); err != nil {
		t.Fatalf("upload through a grant: %v", err)
	}
	if b, err := st.GetBytes(ctx, "runs/my-run/model.lbc"); err != nil || string(b) != "new weights" {
		t.Fatalf("in the store: %q %v", b, err)
	}
	if err := objs.PutFile(ctx, "runs/other-run/model.lbc", filepath.Join(dir, "out.lbc")); err == nil {
		t.Fatal("uploaded to another run")
	}
	if st.Exists(ctx, "runs/other-run/model.lbc") {
		t.Fatal("another run's file exists")
	}
}
