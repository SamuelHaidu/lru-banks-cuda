package syncx

import (
	"archive/tar"
	"bytes"
	"strings"
	"testing"
)

func TestExtractRefusesPathsOutsideTheTarget(t *testing.T) {
	for _, name := range []string{"../escape.txt", "/etc/passwd", "a/../../b"} {
		var buf bytes.Buffer
		tw := tar.NewWriter(&buf)
		_ = tw.WriteHeader(&tar.Header{Typeflag: tar.TypeReg, Name: name, Size: 1, Mode: 0o644})
		_, _ = tw.Write([]byte("x"))
		_ = tw.Close()
		if _, _, err := Extract(&buf, t.TempDir()); err == nil || !strings.Contains(err.Error(), "unsafe path") {
			t.Fatalf("%q: err %v", name, err)
		}
	}
}

func TestDiff(t *testing.T) {
	send, del := Diff(map[string]string{"a": "1", "b": "2"}, map[string]string{"a": "1", "b": "x", "c": "3"})
	if len(send) != 1 || send[0] != "b" || len(del) != 1 || del[0] != "c" {
		t.Fatalf("send %v del %v", send, del)
	}
}

func TestExcluded(t *testing.T) {
	if !Excluded("src/__pycache__/x.pyc", DefaultExcludes) || Excluded("src/model.cu", DefaultExcludes) {
		t.Fatal("excludes")
	}
}
