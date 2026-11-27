package security

// remote-pod ssh connects with a key made for that one connection (internal/sshkey): new every time, nameless,
// private on disk while it lives, deleted after; offered alone, with nothing forwarded. The ssh client's own view of
// the options (ssh -G) is the evidence, not our reading of them.

import (
	"go/ast"
	"go/parser"
	"go/token"
	"io/fs"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/SamuelHaidu/rnn-research/tools/remote-pod/internal/sshkey"
)

func TestEveryConnectionGetsANewNamelessKey(t *testing.T) {
	seen := map[string]bool{}
	for i := 0; i < 20; i++ {
		k, err := sshkey.New(t.TempDir())
		if err != nil {
			t.Fatal(err)
		}
		f := strings.Fields(k.Public)
		if len(f) != 2 || f[0] != "ssh-ed25519" {
			t.Fatalf("public key %q: want exactly `ssh-ed25519 <key>` (no comment that could name you)", k.Public)
		}
		if seen[f[1]] {
			t.Fatal("a key was made twice")
		}
		seen[f[1]] = true
		di, _ := os.Stat(k.Dir)
		fi, _ := os.Stat(k.Private)
		if di.Mode().Perm() != 0o700 || fi.Mode().Perm() != 0o600 {
			t.Fatalf("private key %v in dir %v: want 0600 in 0700", fi.Mode().Perm(), di.Mode().Perm())
		}
		b, _ := os.ReadFile(k.Private)
		if !strings.Contains(string(b), "OPENSSH PRIVATE KEY") {
			t.Fatal("not an OpenSSH private key")
		}
		if err := k.Close(); err != nil {
			t.Fatal(err)
		}
		if _, err := os.Stat(k.Dir); !os.IsNotExist(err) {
			t.Fatal("the private key survived its connection")
		}
	}
}

// What the OpenSSH client will actually do with our options: only our key, no agent, nothing forwarded, no
// password, the pod's own known_hosts, and ~/.ssh/config ignored.
func TestSSHOptionsAsTheClientSeesThem(t *testing.T) {
	if _, err := exec.LookPath("ssh"); err != nil {
		t.Skip("no ssh client")
	}
	k, err := sshkey.New(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer k.Close()
	kh := filepath.Join(t.TempDir(), "known_hosts-555")
	args := k.Args(kh, "root", "203.0.113.9", 40022)
	out, err := exec.Command("ssh", append([]string{"-G"}, args...)...).Output()
	if err != nil {
		t.Fatalf("ssh -G: %v", err)
	}
	got := map[string][]string{}
	for _, line := range strings.Split(string(out), "\n") {
		if f := strings.SplitN(line, " ", 2); len(f) == 2 {
			got[f[0]] = append(got[f[0]], f[1])
		}
	}
	want := map[string]string{
		"identitiesonly": "yes", "identityagent": "none", "forwardagent": "no", "forwardx11": "no",
		"forwardx11trusted": "no", "clearallforwardings": "yes", "passwordauthentication": "no",
		"kbdinteractiveauthentication": "no", "permitlocalcommand": "no", "stricthostkeychecking": "accept-new",
		"updatehostkeys": "false", "hostname": "203.0.113.9", "port": "40022", "user": "root",
	}
	for k2, v := range want {
		if len(got[k2]) == 0 || got[k2][0] != v {
			t.Errorf("ssh would use %s = %v, want %s", k2, got[k2], v)
		}
	}
	if ids := got["identityfile"]; len(ids) != 1 || ids[0] != k.Private {
		t.Errorf("ssh would offer %v, want only this connection's key %s", ids, k.Private)
	}
	if kf := got["userknownhostsfile"]; len(kf) == 0 || !strings.HasPrefix(kf[0], kh) {
		t.Errorf("known hosts %v, want the pod's own file", kf)
	}
	for _, opt := range []string{"localforward", "remoteforward", "dynamicforward", "proxycommand", "proxyjump"} {
		if v := got[opt]; len(v) > 0 && v[0] != "none" {
			t.Errorf("ssh would set %s %v", opt, v)
		}
	}
}

// No code reads the user's own SSH keys (an account key would let a host link pods to you): a static check.
func TestNoCodeReadsYourSSHKeys(t *testing.T) {
	err := filepath.WalkDir("..", func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() && (d.Name() == "security" || d.Name() == "dist") {
			return filepath.SkipDir
		}
		if d.IsDir() || !strings.HasSuffix(p, ".go") || strings.HasSuffix(p, "_test.go") {
			return nil
		}
		f, err := parser.ParseFile(token.NewFileSet(), p, nil, 0)
		if err != nil {
			return err
		}
		ast.Inspect(f, func(n ast.Node) bool {
			if lit, ok := n.(*ast.BasicLit); ok && lit.Kind == token.STRING {
				s := lit.Value
				if strings.Contains(s, ".ssh/") || strings.Contains(s, "id_ed25519") || strings.Contains(s, "id_rsa") ||
					strings.Contains(s, "authorized_keys") {
					t.Errorf("%s: %s", p, s)
				}
			}
			return true
		})
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
}
