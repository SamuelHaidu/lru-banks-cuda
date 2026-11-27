package security

// The real CLI, run against fake RunPod, vast.ai and S3 servers: whatever a provider receives when a pod is created
// (and so whatever the pod's host can read: its environment, onstart, pre-issued grants) holds no secret of ours.

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/johannesboyne/gofakes3"
	"github.com/johannesboyne/gofakes3/backend/s3mem"

	"github.com/SamuelHaidu/rnn-research/tools/remote-pod/internal/grant"
	"github.com/SamuelHaidu/rnn-research/tools/remote-pod/internal/store"
)

// Distinct fake secrets: if any appears in what a provider receives, it leaked.
const (
	r2Secret   = "R2-SECRET-must-never-reach-a-pod-7f3a"
	r2KeyID    = "AKIDSECTEST0001"
	runpodKey  = "RUNPOD-ACCOUNT-KEY-must-never-reach-a-pod-91c2"
	vastKey    = "VAST-ACCOUNT-KEY-must-never-reach-a-pod-5d8e"
	testBucket = "sec-bucket"
)

var secrets = []string{r2Secret, runpodKey, vastKey}

var (
	buildOnce sync.Once
	cliPath   string
	buildErr  error
)

func buildCLI(t *testing.T) string {
	t.Helper()
	buildOnce.Do(func() {
		dir, err := os.MkdirTemp("", "rp-sec-bin")
		if err != nil {
			buildErr = err
			return
		}
		cliPath = filepath.Join(dir, "remote-pod")
		out, err := exec.Command("go", "build", "-o", cliPath, "..").CombinedOutput()
		if err != nil {
			buildErr = err
			cliPath = string(out)
		}
	})
	if buildErr != nil {
		t.Fatalf("building the CLI: %v\n%s", buildErr, cliPath)
	}
	return cliPath
}

// captured is what a fake provider received when asked to create a pod.
type captured struct {
	mu   sync.Mutex
	body []byte
	env  map[string]string
}

func (c *captured) set(body []byte, env map[string]string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.body, c.env = body, env
}

func (c *captured) get() ([]byte, map[string]string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.body, c.env
}

func fakeRunPod(t *testing.T, got *captured) *httptest.Server {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var q struct {
			Query     string `json:"query"`
			Variables struct {
				Input struct {
					Env []struct{ Key, Value string } `json:"env"`
				} `json:"input"`
			} `json:"variables"`
		}
		_ = json.Unmarshal(body, &q)
		switch {
		case strings.Contains(q.Query, "gpuTypes"):
			_, _ = w.Write([]byte(`{"data": {"gpuTypes": [{"id": "NVIDIA GeForce RTX 4090", "displayName": "RTX 4090",
				"memoryInGb": 24, "lowestPrice": {"uninterruptablePrice": 0.5, "stockStatus": "High"}}]}}`))
		case strings.Contains(q.Query, "podFindAndDeployOnDemand"):
			env := map[string]string{}
			for _, e := range q.Variables.Input.Env {
				env[e.Key] = e.Value
			}
			got.set(body, env)
			_, _ = w.Write([]byte(`{"data": {"podFindAndDeployOnDemand": {"id": "pod-sec-1"}}}`))
		default:
			_, _ = w.Write([]byte(`{"data": {}}`))
		}
	}))
	t.Cleanup(ts.Close)
	return ts
}

func fakeVast(t *testing.T, got *captured) *httptest.Server {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		switch {
		case r.URL.Path == "/api/v0/bundles/":
			_, _ = w.Write([]byte(`{"offers": [{"id": 901, "gpu_name": "RTX 4090", "num_gpus": 1, "dph_total": 0.4}]}`))
		case r.Method == "PUT" && r.URL.Path == "/api/v0/asks/901/":
			var b struct {
				Env map[string]string `json:"env"`
			}
			_ = json.Unmarshal(body, &b)
			got.set(body, b.Env)
			_, _ = w.Write([]byte(`{"success": true, "new_contract": 555}`))
		case r.URL.Path == "/api/v0/instances/555/":
			_, _ = w.Write([]byte(`{"instances": {"id": 555, "actual_status": "running", "intended_status": "running",
				"public_ipaddr": "127.0.0.1", "ports": {"8080/tcp": [{"HostIp": "0.0.0.0", "HostPort": "9"}]}}}`))
		default:
			http.NotFound(w, r)
		}
	}))
	t.Cleanup(ts.Close)
	return ts
}

// fakeS3 holds a dataset and a checkpoint a run can name as inputs.
func fakeS3(t *testing.T) string {
	backend := s3mem.New()
	if err := backend.CreateBucket(testBucket); err != nil {
		t.Fatal(err)
	}
	ts := httptest.NewServer(gofakes3.New(backend).Server())
	t.Cleanup(ts.Close)
	st, err := store.New(store.Config{Endpoint: ts.URL, Bucket: testBucket, AccessKey: r2KeyID, SecretKey: r2Secret})
	if err != nil {
		t.Fatal(err)
	}
	ctx := context.Background()
	if err := st.PutDatasetManifest(ctx, store.Dataset{Name: "fw", Key: store.DatasetKey("fw")}); err != nil {
		t.Fatal(err)
	}
	_ = st.PutBytes(ctx, store.DatasetKey("fw"), []byte("archive"))
	_ = st.PutBytes(ctx, "runs/base/model.lbc", []byte("weights"))
	return ts.URL
}

// project is a repository with the settings, a run config, and tools/remote-pod (the agent is built from it).
func project(t *testing.T, s3URL string) (dir, cfgDir string) {
	base := t.TempDir()
	dir, cfgDir = filepath.Join(base, "proj"), filepath.Join(base, "cfg")
	for _, d := range []string{filepath.Join(dir, ".git"), filepath.Join(dir, "tools"), filepath.Join(dir, "runs", "sec-run"), cfgDir} {
		if err := os.MkdirAll(d, 0o755); err != nil {
			t.Fatal(err)
		}
	}
	src, _ := filepath.Abs("..")
	if err := os.Symlink(src, filepath.Join(dir, "tools", "remote-pod")); err != nil {
		t.Fatal(err)
	}
	_ = os.WriteFile(filepath.Join(dir, "runs", "sec-run", "config.json"),
		[]byte(`{"data": {"dataset": "/data/fw"}, "training": {"init_from": "/work/runs/base/model.lbc"}}`), 0o644)
	cfg := `provider = "runpod"
[runpod]
api_key = "` + runpodKey + `"
[vast]
api_key = "` + vastKey + `"
[r2]
endpoint = "` + s3URL + `"
bucket = "` + testBucket + `"
access_key = "` + r2KeyID + `"
secret_key = "` + r2Secret + `"
`
	if err := os.WriteFile(filepath.Join(cfgDir, "config.toml"), []byte(cfg), 0o600); err != nil {
		t.Fatal(err)
	}
	return dir, cfgDir
}

// runUntilCreated runs the CLI until the fake provider got the create request, then stops it (it would wait for
// an agent that does not exist).
func runUntilCreated(t *testing.T, dir, cfgDir, rpURL, vastURL string, got *captured, args ...string) {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Minute)
	defer cancel()
	cmd := exec.CommandContext(ctx, buildCLI(t), args...)
	cmd.Dir = dir
	var env []string
	for _, kv := range os.Environ() {
		if k, _, _ := strings.Cut(kv, "="); strings.HasPrefix(k, "REMOTE_POD_") || k == "RUNPOD_API_KEY" || k == "VAST_API_KEY" {
			continue
		}
		env = append(env, kv)
	}
	cmd.Env = append(env, "REMOTE_POD_CONFIG_DIR="+cfgDir, "REMOTE_POD_RUNPOD_API="+rpURL, "REMOTE_POD_VAST_API="+vastURL)
	var out strings.Builder
	cmd.Stdout, cmd.Stderr = &out, &out
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	done := make(chan error, 1)
	go func() { done <- cmd.Wait() }()
	for {
		if b, _ := got.get(); b != nil {
			time.Sleep(300 * time.Millisecond) // let it save its state
			_ = cmd.Process.Kill()
			<-done
			return
		}
		select {
		case err := <-done:
			if b, _ := got.get(); b != nil { // --detach: it returns right after creating the pod
				return
			}
			t.Fatalf("remote-pod %v ended before creating a pod: %v\n%s", args, err, out.String())
		case <-time.After(100 * time.Millisecond):
		}
	}
}

var b64Run = regexp.MustCompile(`[A-Za-z0-9+/]{24,}={0,2}`)

// leaks looks for the secrets in s, in its URL-decoded form, and in every base64 run inside it (recursively).
func leaks(s string, depth int) []string {
	var found []string
	forms := []string{s}
	if u, err := url.QueryUnescape(s); err == nil {
		forms = append(forms, u)
	}
	for _, f := range forms {
		for _, sec := range secrets {
			if strings.Contains(f, sec) {
				found = append(found, sec)
			}
		}
	}
	if depth > 0 {
		for _, m := range b64Run.FindAllString(s, -1) {
			for _, enc := range []*base64.Encoding{base64.StdEncoding, base64.RawStdEncoding} {
				if d, err := enc.DecodeString(m); err == nil {
					found = append(found, leaks(string(d), depth-1)...)
				}
			}
		}
	}
	return found
}

func checkNoLeak(t *testing.T, what string, body []byte, env map[string]string) {
	t.Helper()
	if l := leaks(string(body), 3); len(l) > 0 {
		t.Errorf("%s: the provider received %v", what, l)
	}
	for k := range env {
		if strings.Contains(k, "S3_SECRET") || strings.Contains(k, "S3_ACCESS") || strings.Contains(k, "API_KEY") {
			t.Errorf("%s: the pod's environment has %s", what, k)
		}
	}
	if env["REMOTE_POD_TAG"] == "" {
		t.Errorf("%s: no pod tag", what)
	}
}

// checkGrants: every pre-issued grant is for a key the policy allows, with the right operation, at most 7 days.
func checkGrants(t *testing.T, what string, env map[string]string, wantGet, wantPut []string) {
	t.Helper()
	raw, err := base64.StdEncoding.DecodeString(env["REMOTE_POD_GRANTS"])
	if err != nil {
		t.Fatalf("%s: REMOTE_POD_GRANTS: %v", what, err)
	}
	var gs []grant.Grant
	if err := json.Unmarshal(raw, &gs); err != nil {
		t.Fatalf("%s: grants: %v", what, err)
	}
	allowed := map[string]bool{}
	for _, k := range wantGet {
		allowed["get "+k] = true
	}
	for _, k := range wantPut {
		allowed["put "+k] = true
	}
	for _, g := range gs {
		if !allowed[g.Op+" "+g.Key] {
			t.Errorf("%s: pre-issued %s %s, not in the pod's needs", what, g.Op, g.Key)
		}
		u, err := url.Parse(g.URL)
		if err != nil || !strings.HasSuffix(u.Path, "/"+g.Key) {
			t.Errorf("%s: grant for %s has URL path %q", what, g.Key, u.Path)
		}
		if exp := u.Query().Get("X-Amz-Expires"); exp == "" || len(exp) > 6 || exp > "604800" && len(exp) == 6 {
			t.Errorf("%s: grant for %s expires after %q s", what, g.Key, exp)
		}
		if strings.Contains(g.URL, r2Secret) {
			t.Errorf("%s: a grant URL carries the secret key", what)
		}
		delete(allowed, g.Op+" "+g.Key)
	}
	if len(allowed) > 0 {
		t.Errorf("%s: missing pre-issued grants %v", what, allowed)
	}
}

func TestNoSecretReachesAPod(t *testing.T) {
	if testing.Short() {
		t.Skip("builds and runs the CLI")
	}
	s3URL := fakeS3(t)
	trainGet := []string{"datasets/fw/data.tar.zst", "runs/base/model.lbc"}
	var trainPut []string
	for _, f := range []string{"model.lbc", "metrics.csv", "train.log", "config.json", "checkpoints/best.lbc"} {
		trainPut = append(trainPut, "runs/sec-run/"+f)
	}
	cases := []struct {
		name     string
		args     []string
		training bool
	}{
		{"runpod up (our image)", []string{"up", "--gpu", "RTX 4090"}, false},
		{"runpod up (public image, bootstrap)", []string{"up", "--gpu", "4090", "--image", "cuda"}, false},
		{"vast up", []string{"up", "--provider", "vast", "--gpu", "RTX 4090"}, false},
		{"runpod train", []string{"train", "runs/sec-run/config.json", "--gpu", "4090", "--detach"}, true},
		{"vast train", []string{"train", "runs/sec-run/config.json", "--provider", "vast", "--gpu", "RTX 4090", "--detach"}, true},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			dir, cfgDir := project(t, s3URL)
			var rp, va captured
			rpURL, vastURL := fakeRunPod(t, &rp).URL, fakeVast(t, &va).URL
			got := &rp
			if strings.Contains(c.name, "vast") {
				got = &va
			}
			runUntilCreated(t, dir, cfgDir, rpURL, vastURL, got, c.args...)
			body, env := got.get()
			checkNoLeak(t, c.name, body, env)
			if c.training {
				checkGrants(t, c.name, env, trainGet, trainPut)
				if l := leaks(env["REMOTE_POD_JOB"], 3); len(l) > 0 {
					t.Errorf("%s: the job spec carries %v", c.name, l)
				}
			}
			// the project state (this machine) holds the policy; it never holds the store key either
			st, err := os.ReadFile(filepath.Join(dir, ".remote-pod", "state.json"))
			if err != nil {
				t.Fatalf("no state: %v", err)
			}
			if l := leaks(string(st), 2); len(l) > 0 {
				t.Errorf("%s: state.json holds %v", c.name, l)
			}
			var s struct {
				Tag    string       `json:"tag"`
				Policy grant.Policy `json:"policy"`
			}
			_ = json.Unmarshal(st, &s)
			if s.Tag == "" || len(s.Policy.PutPrefixes) == 0 || s.Policy.PutPrefixes[0] != grant.JobLogPrefix(s.Tag) {
				t.Errorf("%s: state policy %+v", c.name, s)
			}
		})
	}
}
