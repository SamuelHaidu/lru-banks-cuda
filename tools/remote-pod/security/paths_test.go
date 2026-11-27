package security

// What a pod sends back never lands outside where it was asked to: store keys turned into local paths (pull), and
// tar streams (cp from the pod, dataset archives) are both checked.

import (
	"archive/tar"
	"bytes"
	"os"
	"path/filepath"
	"testing"

	"github.com/SamuelHaidu/rnn-research/tools/remote-pod/internal/syncx"
)

func TestStoreKeysCannotEscapeTheRunDirectory(t *testing.T) {
	for rel, ok := range map[string]bool{
		"metrics.csv": true, "checkpoints/best.lbc": true, "a/b/c.txt": true, "0..": true,
		"": false, "..": false, "../x": false, "a/../../x": false, "/etc/passwd": false,
		"../../../../.bashrc": false, `a\..\..\x`: false, `..\x`: false, "a/../..": false,
	} {
		if got := syncx.SafeRelPath(rel); got != ok {
			t.Errorf("SafeRelPath(%q) = %v, want %v", rel, got, ok)
		}
	}
}

type entry struct {
	name, link string
	typ        byte
	body       string
}

func tarOf(t *testing.T, es []entry) *bytes.Buffer {
	t.Helper()
	var b bytes.Buffer
	tw := tar.NewWriter(&b)
	for _, e := range es {
		h := &tar.Header{Name: e.name, Typeflag: e.typ, Linkname: e.link, Mode: 0o644, Size: int64(len(e.body))}
		if e.typ != tar.TypeReg {
			h.Size = 0
		}
		if err := tw.WriteHeader(h); err != nil {
			t.Fatal(err)
		}
		if e.typ == tar.TypeReg {
			_, _ = tw.Write([]byte(e.body))
		}
	}
	_ = tw.Close()
	return &b
}

// Each hostile stream is extracted into target; afterwards nothing exists outside it (outside has a canary dir).
func TestTarStreamsCannotEscapeTheTarget(t *testing.T) {
	streams := map[string][]entry{
		"dot-dot":           {{name: "../escaped", typ: tar.TypeReg, body: "x"}},
		"deep dot-dot":      {{name: "a/../../escaped", typ: tar.TypeReg, body: "x"}},
		"absolute":          {{name: "/tmp/escaped-abs", typ: tar.TypeReg, body: "x"}},
		"symlink then file": {{name: "link", typ: tar.TypeSymlink, link: ".."}, {name: "link/escaped", typ: tar.TypeReg, body: "x"}},
		"symlink to /etc":   {{name: "etc", typ: tar.TypeSymlink, link: "/etc"}, {name: "etc/escaped", typ: tar.TypeReg, body: "x"}},
		"hard link out":     {{name: "hard", typ: tar.TypeLink, link: "../outside/canary"}, {name: "hard", typ: tar.TypeReg, body: "overwritten"}},
	}
	for name, es := range streams {
		base := t.TempDir()
		target, outside := filepath.Join(base, "target"), filepath.Join(base, "outside")
		_ = os.MkdirAll(target, 0o755)
		_ = os.MkdirAll(outside, 0o755)
		canary := filepath.Join(outside, "canary")
		_ = os.WriteFile(canary, []byte("intact"), 0o644)
		_, _, _ = syncx.Extract(tarOf(t, es), target) // refusing with an error is fine; escaping is not
		_ = filepath.Walk(base, func(p string, fi os.FileInfo, err error) error {
			if err == nil && p != base && !isWithin(target, p) && p != outside && p != canary {
				t.Errorf("%s: %s was written outside the target", name, p)
			}
			if err == nil && fi.Mode()&os.ModeSymlink != 0 {
				t.Errorf("%s: a symlink was created at %s", name, p)
			}
			return nil
		})
		if b, _ := os.ReadFile(canary); string(b) != "intact" {
			t.Errorf("%s: the file outside was changed to %q", name, b)
		}
		if _, err := os.Stat("/tmp/escaped-abs"); err == nil {
			_ = os.Remove("/tmp/escaped-abs")
			t.Errorf("%s: wrote an absolute path", name)
		}
		if _, err := os.Stat("/etc/escaped"); err == nil {
			t.Errorf("%s: wrote into /etc", name)
		}
	}
}

func isWithin(dir, p string) bool {
	rel, err := filepath.Rel(dir, p)
	return err == nil && rel != ".." && !filepath.IsAbs(rel) && (rel == "." || !startsWithDotDot(rel))
}

func startsWithDotDot(rel string) bool {
	return rel == ".." || len(rel) > 2 && rel[:3] == ".."+string(filepath.Separator)
}
