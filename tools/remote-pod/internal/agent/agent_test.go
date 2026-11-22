package agent_test

import (
	"bytes"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/SamuelHaidu/rnn-research/tools/remote-pod/internal/agent"
	"github.com/SamuelHaidu/rnn-research/tools/remote-pod/internal/client"
	"github.com/SamuelHaidu/rnn-research/tools/remote-pod/internal/proto"
)

// start runs an agent on a test server and returns a client for it.
func start(t *testing.T) (*client.Client, string) {
	t.Helper()
	work := t.TempDir()
	srv, err := agent.New(agent.Options{Token: "secret", Workdir: work})
	if err != nil {
		t.Fatal(err)
	}
	ts := httptest.NewServer(srv.Handler())
	t.Cleanup(ts.Close)
	return client.New(ts.URL, "secret"), work
}

func run(t *testing.T, cl *client.Client, cmd string, stdin string) (int, string, string) {
	t.Helper()
	var out, errb bytes.Buffer
	io := client.ExecIO{Stdout: &out, Stderr: &errb}
	if stdin != "" {
		io.Stdin = strings.NewReader(stdin)
	}
	code, err := cl.Exec(proto.ExecStart{Command: cmd}, io)
	if err != nil {
		t.Fatal(err)
	}
	return code, out.String(), errb.String()
}

func TestExecOutputAndExitCode(t *testing.T) {
	cl, _ := start(t)
	code, out, errs := run(t, cl, "echo out; echo err >&2; exit 3", "")
	if code != 3 || out != "out\n" || errs != "err\n" {
		t.Fatalf("code %d out %q err %q", code, out, errs)
	}
}

func TestExecStdinAndWorkdir(t *testing.T) {
	cl, work := start(t)
	code, out, _ := run(t, cl, "tr a-z A-Z; pwd", "hello\n")
	if code != 0 || out != "HELLO\n"+work+"\n" {
		t.Fatalf("code %d out %q", code, out)
	}
}

func TestExecSignalReachesTheCommand(t *testing.T) {
	cl, _ := start(t)
	sigs := make(chan string, 1)
	var out bytes.Buffer
	go func() { time.Sleep(300 * time.Millisecond); sigs <- "INT" }()
	code, err := cl.Exec(proto.ExecStart{Command: "sleep 30"}, client.ExecIO{Stdout: &out, Stderr: &out, Signals: sigs})
	if err != nil {
		t.Fatal(err)
	}
	if code != 130 { // 128 + SIGINT
		t.Fatalf("exit %d, want 130", code)
	}
}

func TestBadTokenIsRefused(t *testing.T) {
	cl, _ := start(t)
	cl.Token = "wrong"
	if _, err := cl.Health(); err == nil || !strings.Contains(err.Error(), "refused the token") {
		t.Fatalf("err %v", err)
	}
}

func TestCopyRoundTrip(t *testing.T) {
	cl, work := start(t)
	src := t.TempDir()
	os.MkdirAll(filepath.Join(src, "d", "e"), 0o755)
	os.WriteFile(filepath.Join(src, "d", "a.txt"), []byte("A"), 0o644)
	os.WriteFile(filepath.Join(src, "d", "e", "b.bin"), bytes.Repeat([]byte{7}, 3<<20), 0o600)
	target, err := cl.Upload(filepath.Join(src, "d"), "dst/", nil, nil)
	if err != nil || target != filepath.Join(work, "dst", "d") {
		t.Fatalf("upload: %v %s", err, target)
	}
	back := t.TempDir()
	got, err := cl.Download("dst/d", back+"/", nil)
	if err != nil {
		t.Fatal(err)
	}
	b, _ := os.ReadFile(filepath.Join(got, "e", "b.bin"))
	a, _ := os.ReadFile(filepath.Join(got, "a.txt"))
	if len(b) != 3<<20 || string(a) != "A" {
		t.Fatalf("round trip: %d bytes, %q", len(b), a)
	}
	fi, _ := os.Stat(filepath.Join(got, "e", "b.bin"))
	if fi.Mode().Perm() != 0o600 {
		t.Fatalf("mode %v", fi.Mode().Perm())
	}
	// a single file to a new name
	if target, err := cl.Upload(filepath.Join(src, "d", "a.txt"), "renamed.txt", nil, nil); err != nil || filepath.Base(target) != "renamed.txt" {
		t.Fatalf("file: %v %s", err, target)
	}
}

func TestSyncSendsOnlyChanges(t *testing.T) {
	cl, work := start(t)
	src := t.TempDir()
	os.WriteFile(filepath.Join(src, "k1.cu"), []byte("one"), 0o644)
	os.WriteFile(filepath.Join(src, "k2.cu"), []byte("two"), 0o644)
	os.MkdirAll(filepath.Join(src, ".git"), 0o755)
	os.WriteFile(filepath.Join(src, ".git", "HEAD"), []byte("x"), 0o644)
	ex := []string{".git"}
	sent, _, _, err := cl.Sync(src, "engine", ex, false)
	if err != nil || len(sent) != 2 {
		t.Fatalf("first sync: %v %v", err, sent)
	}
	os.WriteFile(filepath.Join(src, "k2.cu"), []byte("two, edited"), 0o644)
	os.Remove(filepath.Join(src, "k1.cu"))
	sent, deleted, _, err := cl.Sync(src, "engine", ex, true)
	if err != nil || len(sent) != 1 || sent[0] != "k2.cu" || len(deleted) != 1 {
		t.Fatalf("second sync: %v sent %v deleted %v", err, sent, deleted)
	}
	b, _ := os.ReadFile(filepath.Join(work, "engine", "k2.cu"))
	if string(b) != "two, edited" {
		t.Fatalf("content %q", b)
	}
	if _, err := os.Stat(filepath.Join(work, "engine", ".git")); err == nil {
		t.Fatal(".git was sent")
	}
}

func TestDetachedJobLogResumes(t *testing.T) {
	cl, _ := start(t)
	j, err := cl.StartJob(proto.ExecStart{Command: "for i in 1 2 3; do echo line$i; sleep 0.2; done; exit 5", Name: "demo"})
	if err != nil {
		t.Fatal(err)
	}
	var first bytes.Buffer
	code, err := cl.FollowLog(j.ID, 0, true, &first, nil)
	if err != nil || code != 5 || first.String() != "line1\nline2\nline3\n" {
		t.Fatalf("follow: %v code %d %q", err, code, first.String())
	}
	var rest bytes.Buffer // a reconnect resumes at the byte it had reached
	if _, err := cl.FollowLog("demo", int64(len("line1\n")), false, &rest, nil); err != nil || rest.String() != "line2\nline3\n" {
		t.Fatalf("resume: %v %q", err, rest.String())
	}
	got, err := cl.Job("demo")
	if err != nil || got.Running || got.ExitCode != 5 {
		t.Fatalf("job %+v %v", got, err)
	}
}
