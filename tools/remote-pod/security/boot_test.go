package security

// The boot script (package boot) runs on a public image: it must refuse binaries whose sha256 is not the one fixed
// when the pod was created (a replaced object in the store, a tampered download), and run nothing then.

import (
	"crypto/sha256"
	"encoding/hex"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/SamuelHaidu/rnn-research/tools/remote-pod/internal/boot"
)

// fakeAgent is a "binary" that proves it ran by printing a marker.
const fakeAgent = "#!/bin/sh\necho AGENT-STARTED \"$@\"\n"
const fakeLbc = "#!/bin/sh\necho LBC\n"

func sum(s string) string {
	h := sha256.Sum256([]byte(s))
	return hex.EncodeToString(h[:])
}

func runBoot(t *testing.T, served map[string]string, env map[string]string) (string, error, string) {
	t.Helper()
	if _, err := exec.LookPath("bash"); err != nil {
		t.Skip("no bash")
	}
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(served[r.URL.Path]))
	}))
	defer ts.Close()
	bin := filepath.Join(t.TempDir(), "bin")
	cmd := exec.Command("bash", "-c", boot.Script)
	cmd.Env = []string{"PATH=" + os.Getenv("PATH"), "REMOTE_POD_BIN_DIR=" + bin, "REMOTE_POD_WORKDIR=" + t.TempDir(),
		"REMOTE_POD_AGENT_URL=" + ts.URL + "/agent"}
	for k, v := range env {
		cmd.Env = append(cmd.Env, k+"="+strings.ReplaceAll(v, "URL/", ts.URL+"/"))
	}
	out, err := cmd.CombinedOutput()
	return string(out), err, bin
}

func TestBootRunsTheAgentItWasGiven(t *testing.T) {
	out, err, bin := runBoot(t, map[string]string{"/agent": fakeAgent, "/lbc": fakeLbc},
		map[string]string{"REMOTE_POD_AGENT_SHA256": sum(fakeAgent), "REMOTE_POD_LBC_URL": "URL/lbc",
			"REMOTE_POD_LBC_SHA256": sum(fakeLbc)})
	if err != nil || !strings.Contains(out, "AGENT-STARTED agent") {
		t.Fatalf("the genuine agent did not start: %v\n%s", err, out)
	}
	if _, err := os.Stat(filepath.Join(bin, "lbc")); err != nil {
		t.Fatal("lbc not installed")
	}
}

func TestBootRefusesATamperedAgent(t *testing.T) {
	evil := "#!/bin/sh\necho AGENT-STARTED owned\n"
	out, err, bin := runBoot(t, map[string]string{"/agent": evil},
		map[string]string{"REMOTE_POD_AGENT_SHA256": sum(fakeAgent)}) // the hash fixed at creation: the genuine one
	if err == nil || strings.Contains(out, "AGENT-STARTED") {
		t.Fatalf("a tampered agent ran: %v\n%s", err, out)
	}
	if _, err := os.Stat(filepath.Join(bin, "remote-pod")); err == nil {
		t.Fatal("the tampered agent was left on disk")
	}
}

func TestBootRefusesATamperedLbc(t *testing.T) {
	out, err, bin := runBoot(t, map[string]string{"/agent": fakeAgent, "/lbc": "#!/bin/sh\necho evil\n"},
		map[string]string{"REMOTE_POD_AGENT_SHA256": sum(fakeAgent), "REMOTE_POD_LBC_URL": "URL/lbc",
			"REMOTE_POD_LBC_SHA256": sum(fakeLbc)})
	if err == nil || strings.Contains(out, "AGENT-STARTED") {
		t.Fatalf("booted with a tampered lbc: %v\n%s", err, out)
	}
	if _, err := os.Stat(filepath.Join(bin, "lbc")); err == nil {
		t.Fatal("the tampered lbc was left on disk")
	}
}

func TestBootRefusesWithoutAHash(t *testing.T) {
	out, err, _ := runBoot(t, map[string]string{"/agent": fakeAgent}, nil)
	if err == nil || strings.Contains(out, "AGENT-STARTED") {
		t.Fatalf("booted an agent with no hash to check: %v\n%s", err, out)
	}
}
