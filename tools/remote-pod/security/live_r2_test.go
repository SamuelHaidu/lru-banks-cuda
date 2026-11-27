package security

// Against the real store (R2), with the settings' key, under security-tests/<random>/ (deleted afterwards):
// SECURITY_LIVE=1 go test ./security/ -run Live -v
//
// What the policy decides is only as good as what a presigned URL can do. This shows R2 itself holds a URL to its
// one object, its one method, its exact size and its expiry, through the embedded roots (R2's real certificate).

import (
	"bytes"
	"context"
	"crypto/rand"
	"encoding/hex"
	"io"
	"net/http"
	"net/url"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/SamuelHaidu/rnn-research/tools/remote-pod/internal/config"
	"github.com/SamuelHaidu/rnn-research/tools/remote-pod/internal/store"
	"github.com/SamuelHaidu/rnn-research/tools/remote-pod/internal/tlsroots"
)

func liveStore(t *testing.T) *store.Store {
	if os.Getenv("SECURITY_LIVE") != "1" {
		t.Skip("SECURITY_LIVE=1 runs it against the real store")
	}
	cfg, err := config.Load()
	if err != nil || !cfg.HasR2() {
		t.Skip("no store configured")
	}
	st, err := store.New(store.Config{Endpoint: cfg.R2.Endpoint, Bucket: cfg.R2.Bucket, AccessKey: cfg.R2.AccessKey,
		SecretKey: cfg.R2.SecretKey, Region: cfg.R2.Region})
	if err != nil {
		t.Fatal(err)
	}
	return st
}

func do(t *testing.T, method, u string, body []byte) (int, string) {
	t.Helper()
	req, _ := http.NewRequest(method, u, bytes.NewReader(body))
	if body != nil {
		req.ContentLength = int64(len(body))
	}
	resp, err := tlsroots.Client(30 * time.Second).Do(req)
	if err != nil {
		t.Fatalf("%s: %v", method, err)
	}
	defer resp.Body.Close()
	b, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
	return resp.StatusCode, string(b)
}

func otherKey(u, from, to string) string {
	p, _ := url.Parse(u)
	p.Path = strings.Replace(p.Path, from, to, 1)
	return p.String()
}

func TestLiveR2GrantIsBoundToOneObjectMethodSizeAndTime(t *testing.T) {
	st := liveStore(t)
	ctx := context.Background()
	rb := make([]byte, 6)
	_, _ = rand.Read(rb)
	base := "security-tests/" + hex.EncodeToString(rb) + "/"
	keyA, keyB := base+"a.bin", base+"b.bin"
	t.Cleanup(func() {
		for _, k := range []string{keyA, keyB} {
			_ = st.Remove(context.Background(), k)
		}
	})
	data := []byte("exactly these 33 bytes, no more!!")

	putA, err := st.PresignPut(ctx, keyA, 2*time.Minute, int64(len(data)))
	if err != nil {
		t.Fatal(err)
	}
	if code, b := do(t, "PUT", putA, append(append([]byte{}, data...), []byte(" and 1 GB more")...)); code < 400 {
		t.Errorf("a sized URL took a longer body: HTTP %d %s", code, b)
	}
	if code, b := do(t, "PUT", putA, data[:10]); code < 400 {
		t.Errorf("a sized URL took a shorter body: HTTP %d %s", code, b)
	}
	if code, b := do(t, "PUT", putA, data); code != 200 {
		t.Fatalf("the declared upload failed: HTTP %d %s", code, b)
	}
	if code, _ := do(t, "PUT", otherKey(putA, "a.bin", "b.bin"), data); code < 400 {
		t.Error("the upload URL wrote another key")
	}
	if st.Exists(ctx, keyB) {
		t.Error("another key exists")
	}
	if code, _ := do(t, "GET", putA, nil); code < 400 {
		t.Error("the upload URL read the object")
	}
	if code, _ := do(t, "DELETE", putA, nil); code < 400 {
		t.Error("the upload URL deleted the object")
	}
	if !st.Exists(ctx, keyA) {
		t.Fatal("the object is gone")
	}

	getA, err := st.PresignGet(ctx, keyA, 2*time.Minute)
	if err != nil {
		t.Fatal(err)
	}
	if code, b := do(t, "GET", getA, nil); code != 200 || b != string(data) {
		t.Fatalf("the download URL: HTTP %d %q", code, b)
	}
	if code, _ := do(t, "PUT", getA, data); code < 400 {
		t.Error("the download URL wrote")
	}
	if code, _ := do(t, "GET", otherKey(getA, "a.bin", "anything-else"), nil); code < 400 {
		t.Error("the download URL read another key")
	}
	if code, _ := do(t, "GET", otherKey(getA, base+"a.bin", "datasets/"), nil); code < 400 {
		t.Error("the download URL listed or read elsewhere")
	}

	short, err := st.PresignPut(ctx, keyB, 2*time.Second, int64(len(data)))
	if err != nil {
		t.Fatal(err)
	}
	time.Sleep(5 * time.Second)
	if code, _ := do(t, "PUT", short, data); code < 400 {
		t.Error("an expired URL still wrote")
	}
	if st.Exists(ctx, keyB) {
		t.Error("an expired URL created the object")
	}
}
