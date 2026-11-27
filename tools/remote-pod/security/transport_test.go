package security

// Who can talk to the agent, and whom our connections trust: the token on every route, the pinned certificate of a
// vast.ai pod, and the embedded public roots everywhere else, which a CA injected by a TLS inspector (Netskope-style),
// malware or a hostile host does not join.

import (
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/tls"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/pem"
	"go/ast"
	"go/parser"
	"go/token"
	"io/fs"
	"math/big"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/SamuelHaidu/rnn-research/tools/remote-pod/internal/agent"
	"github.com/SamuelHaidu/rnn-research/tools/remote-pod/internal/client"
	"github.com/SamuelHaidu/rnn-research/tools/remote-pod/internal/pin"
	"github.com/SamuelHaidu/rnn-research/tools/remote-pod/internal/tlsroots"
)

// Every route, the signer's included, refuses a request without the token or with another pod's.
func TestEveryAgentRouteNeedsTheToken(t *testing.T) {
	srv, err := agent.New(agent.Options{Token: "the-pods-token", Workdir: t.TempDir()})
	if err != nil {
		t.Fatal(err)
	}
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()
	routes := [][2]string{{"GET", "/v1/health"}, {"GET", "/v1/exec"}, {"POST", "/v1/files?dest=x"},
		{"GET", "/v1/files?path=/etc/passwd"}, {"POST", "/v1/manifest"}, {"POST", "/v1/delete"},
		{"POST", "/v1/jobs"}, {"GET", "/v1/jobs"}, {"GET", "/v1/jobs/x"}, {"POST", "/v1/jobs/x/signal"},
		{"GET", "/v1/jobs/x/log"}, {"GET", "/v1/signer"}}
	for _, auth := range []string{"", "Bearer wrong-token", "Bearer the-pods-token-but-longer", "the-pods-token",
		"Basic the-pods-token"} {
		for _, rt := range routes {
			req, _ := http.NewRequest(rt[0], ts.URL+rt[1], nil)
			if auth != "" {
				req.Header.Set("Authorization", auth)
			}
			resp, err := http.DefaultClient.Do(req)
			if err != nil {
				t.Fatal(err)
			}
			resp.Body.Close()
			if resp.StatusCode != http.StatusUnauthorized {
				t.Errorf("%s %s with %q: HTTP %d, want 401", rt[0], rt[1], auth, resp.StatusCode)
			}
		}
	}
}

// A self-signed CA, the way a TLS inspector or an attacker makes one, and a server certificate it signs for host.
func makeCA(t *testing.T, name string) (*x509.Certificate, *ecdsa.PrivateKey, []byte) {
	t.Helper()
	key, _ := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	tmpl := &x509.Certificate{SerialNumber: big.NewInt(time.Now().UnixNano()), Subject: pkix.Name{CommonName: name},
		NotBefore: time.Now().Add(-time.Hour), NotAfter: time.Now().Add(time.Hour), IsCA: true,
		BasicConstraintsValid: true, KeyUsage: x509.KeyUsageCertSign | x509.KeyUsageDigitalSignature}
	der, err := x509.CreateCertificate(rand.Reader, tmpl, tmpl, &key.PublicKey, key)
	if err != nil {
		t.Fatal(err)
	}
	c, _ := x509.ParseCertificate(der)
	return c, key, pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: der})
}

func serverCert(t *testing.T, ca *x509.Certificate, caKey *ecdsa.PrivateKey, hosts ...string) tls.Certificate {
	t.Helper()
	key, _ := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	tmpl := &x509.Certificate{SerialNumber: big.NewInt(time.Now().UnixNano()), Subject: pkix.Name{CommonName: hosts[0]},
		DNSNames: hosts, NotBefore: time.Now().Add(-time.Hour), NotAfter: time.Now().Add(time.Hour),
		ExtKeyUsage: []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth}, KeyUsage: x509.KeyUsageDigitalSignature}
	der, err := x509.CreateCertificate(rand.Reader, tmpl, ca, &key.PublicKey, caKey)
	if err != nil {
		t.Fatal(err)
	}
	return tls.Certificate{Certificate: [][]byte{der}, PrivateKey: key}
}

// interceptor is a TLS server presenting a certificate an inspector's CA signed for the names we would ask for.
func interceptor(t *testing.T, ca *x509.Certificate, caKey *ecdsa.PrivateKey, seen *[]string) *httptest.Server {
	t.Helper()
	ts := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		*seen = append(*seen, r.Header.Get("Authorization")) // what an interceptor would steal
		_, _ = w.Write([]byte(`{"ok": true}`))
	}))
	ts.TLS = &tls.Config{Certificates: []tls.Certificate{serverCert(t, ca, caKey, pin.ServerName, "localhost",
		"api.runpod.io", "console.vast.ai", "example.r2.cloudflarestorage.com")}}
	ts.StartTLS()
	t.Cleanup(ts.Close)
	return ts
}

// A vast.ai pod's agent is pinned: an interceptor whose CA this machine (or the network) trusts still cannot pose as
// the agent, even with a certificate for the exact name the client checks; the token never leaves.
func TestPinnedAgentRefusesAnInterceptor(t *testing.T) {
	podCert, podKey, err := pin.New()
	if err != nil {
		t.Fatal(err)
	}
	corpCA, corpKey, _ := makeCA(t, "Corporate TLS Inspection CA")
	var stolen []string
	mitm := interceptor(t, corpCA, corpKey, &stolen)

	cl := client.New(mitm.URL, "the-pods-token")
	if err := cl.Pin(podCert); err != nil {
		t.Fatal(err)
	}
	if _, err := cl.Health(); err == nil {
		t.Fatal("the pinned client accepted the interceptor's certificate")
	}
	if len(stolen) != 0 {
		t.Fatalf("the interceptor received requests: %q", stolen)
	}

	// the real agent with the pod's certificate is accepted by the same client
	sc, err := pin.ServerConfig(podCert, podKey)
	if err != nil {
		t.Fatal(err)
	}
	srv, _ := agent.New(agent.Options{Token: "the-pods-token", Workdir: t.TempDir()})
	real := httptest.NewUnstartedServer(srv.Handler())
	real.TLS = sc
	real.StartTLS()
	defer real.Close()
	good := client.New(real.URL, "the-pods-token")
	if err := good.Pin(podCert); err != nil {
		t.Fatal(err)
	}
	if _, err := good.Health(); err != nil {
		t.Fatalf("the pinned client refused the real agent: %v", err)
	}

	// another pod's certificate (a host that kept one from an earlier rental) is refused too
	other, _, _ := pin.New()
	bad := client.New(real.URL, "the-pods-token")
	_ = bad.Pin(other)
	if _, err := bad.Health(); err == nil {
		t.Fatal("a client pinned to another certificate accepted this agent")
	}
	// and an unpinned client does not trust the pod's self-signed certificate at all
	if _, err := client.New(real.URL, "the-pods-token").Health(); err == nil {
		t.Fatal("an unpinned client accepted the pod's self-signed certificate")
	}
}

// The embedded roots refuse a certificate from an injected CA even when the operating system trusts that CA
// (SSL_CERT_FILE: the system store as a TLS inspector configures it). Run in a child process: the system store is
// read once per process.
func TestInjectedCAIsNotTrusted(t *testing.T) {
	if os.Getenv("SECURITY_CHILD") == "injected-ca" {
		injectedCAChild(t)
		return
	}
	dir := t.TempDir()
	corpCA, corpKey, caPEM := makeCA(t, "Corporate TLS Inspection CA")
	if err := os.WriteFile(filepath.Join(dir, "ca.pem"), caPEM, 0o600); err != nil {
		t.Fatal(err)
	}
	var stolen []string
	mitm := interceptor(t, corpCA, corpKey, &stolen)
	cmd := exec.Command(os.Args[0], "-test.run=^TestInjectedCAIsNotTrusted$", "-test.v")
	cmd.Env = append(os.Environ(), "SECURITY_CHILD=injected-ca", "SSL_CERT_FILE="+filepath.Join(dir, "ca.pem"),
		"SSL_CERT_DIR="+dir, "MITM_URL="+mitm.URL)
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("child: %v\n%s", err, out)
	}
	t.Logf("child:\n%s", out)
}

func injectedCAChild(t *testing.T) {
	u := strings.Replace(os.Getenv("MITM_URL"), "127.0.0.1", "localhost", 1)
	// the attack works against a client that trusts the system store (where the CA was injected)
	sys := &http.Client{Transport: &http.Transport{TLSClientConfig: &tls.Config{}}}
	if resp, err := sys.Get(u); err != nil {
		t.Fatalf("setup: the system store does not trust the injected CA, so this test proves nothing: %v", err)
	} else {
		resp.Body.Close()
		t.Log("a client trusting the system store was intercepted (expected: this is the attack)")
	}
	// ours does not: the embedded roots
	resp, err := tlsroots.Client(10 * time.Second).Get(u)
	if err == nil {
		resp.Body.Close()
		t.Fatal("tlsroots trusted a CA injected into the system store")
	}
	if !strings.Contains(tlsroots.Explain("localhost", err).Error(), "intercepting") {
		t.Errorf("the refusal does not say what happened: %v", tlsroots.Explain("localhost", err))
	}
	t.Logf("tlsroots refused: %v", err)
	// and so do the clients built on it: the agent client (RunPod's proxy path) ...
	if _, err := client.New(u, "the-pods-token").Health(); err == nil {
		t.Fatal("the agent client trusted the injected CA")
	}
	// ... and the pod's uploads through grants
	t.Setenv("REMOTE_POD_TLS_ROOTS", "")
}

// The token is only ever sent over https, or plain http to this machine (a local agent).
func TestTokenNeverGoesOutInClear(t *testing.T) {
	for url, ok := range map[string]bool{
		"https://1.2.3.4:40123": true, "https://abc-8080.proxy.runpod.net": true,
		"http://127.0.0.1:8080": true, "http://localhost:8080": true, "http://[::1]:8080": true,
		"http://1.2.3.4:40123": false, "http://abc-8080.proxy.runpod.net": false, "ws://1.2.3.4": false,
		"1.2.3.4:8080": false, "https://": false, "http://127.0.0.1.evil.com:8080": false, "ftp://x": false,
	} {
		if err := client.CheckBase(url); (err == nil) != ok {
			t.Errorf("CheckBase(%q) = %v, want ok=%v", url, err, ok)
		}
	}
}

// No code turns verification off or trusts the system store behind our back: a static check of every Go file.
func TestNoCodeWeakensTLS(t *testing.T) {
	root := ".."
	err := filepath.WalkDir(root, func(p string, d fs.DirEntry, err error) error {
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
			switch x := n.(type) {
			case *ast.KeyValueExpr:
				if id, ok := x.Key.(*ast.Ident); ok && (id.Name == "InsecureSkipVerify" || id.Name == "VerifyPeerCertificate") {
					t.Errorf("%s: sets %s", p, id.Name)
				}
			case *ast.SelectorExpr:
				if id, ok := x.X.(*ast.Ident); ok {
					switch id.Name + "." + x.Sel.Name {
					case "x509.SystemCertPool", "http.DefaultClient", "http.Get", "http.Post":
						t.Errorf("%s: uses %s.%s (the system's trust, not tlsroots)", p, id.Name, x.Sel.Name)
					}
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
