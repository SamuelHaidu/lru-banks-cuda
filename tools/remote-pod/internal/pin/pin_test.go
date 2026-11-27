package pin

import (
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
)

// The client trusts the pinned certificate and refuses another pod's.
func TestPinned(t *testing.T) {
	cert, key, err := New()
	if err != nil {
		t.Fatal(err)
	}
	sc, err := ServerConfig(cert, key)
	if err != nil {
		t.Fatal(err)
	}
	srv := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprint(w, "ok")
	}))
	srv.TLS = sc
	srv.StartTLS()
	defer srv.Close()

	cc, err := ClientConfig(cert)
	if err != nil {
		t.Fatal(err)
	}
	c := &http.Client{Transport: &http.Transport{TLSClientConfig: cc}}
	resp, err := c.Get(srv.URL)
	if err != nil {
		t.Fatalf("pinned certificate refused: %v", err)
	}
	b, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	if string(b) != "ok" {
		t.Fatalf("got %q", b)
	}

	other, _, _ := New()
	oc, _ := ClientConfig(other)
	c2 := &http.Client{Transport: &http.Transport{TLSClientConfig: oc}}
	if _, err := c2.Get(srv.URL); err == nil {
		t.Fatal("another certificate was accepted")
	}
}
