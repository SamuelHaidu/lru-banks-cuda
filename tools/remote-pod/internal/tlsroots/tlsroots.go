// Package tlsroots is the trust every outside HTTPS connection uses (the store, RunPod, vast.ai, RunPod's proxy, and
// the agent's own uploads on a pod): Mozilla's NSS root certificates, embedded in the binary at build time. The
// operating system's store is never consulted, so a CA added to it (a corporate TLS inspector such as Netskope or
// Zscaler, malware, or a host that plants one in a rented container) cannot intercept us: its certificates fail
// verification, and the connection is refused before any key, token or data is sent.
//
// The pod's agent itself (vast.ai) is not reached through this package: it is pinned to the one certificate made for
// it (package pin).
//
// REMOTE_POD_TLS_ROOTS=system trusts the operating system's store instead: only for a network where you knowingly
// accept TLS inspection.
package tlsroots

import (
	"crypto/tls"
	"crypto/x509"
	"errors"
	"fmt"
	"net/http"
	"os"
	"sync"
	"time"

	"golang.org/x/crypto/x509roots/fallback/bundle"
)

var (
	once sync.Once
	pool *x509.CertPool
)

// Pool is the embedded roots (nil when REMOTE_POD_TLS_ROOTS=system: crypto/tls then uses the system's).
func Pool() *x509.CertPool {
	if os.Getenv("REMOTE_POD_TLS_ROOTS") == "system" {
		return nil
	}
	once.Do(func() {
		pool = x509.NewCertPool()
		for r := range bundle.Roots() {
			c, err := x509.ParseCertificate(r.Certificate)
			if err != nil {
				panic(fmt.Sprintf("tlsroots: embedded root: %v", err))
			}
			if r.Constraint == nil {
				pool.AddCert(c)
			} else {
				pool.AddCertWithConstraint(c, r.Constraint)
			}
		}
	})
	return pool
}

// Config verifies servers against the embedded roots only (TLS 1.2 or newer).
func Config() *tls.Config {
	return &tls.Config{RootCAs: Pool(), MinVersion: tls.VersionTLS12}
}

// Transport is http.DefaultTransport's settings with Config.
func Transport() *http.Transport {
	t := http.DefaultTransport.(*http.Transport).Clone()
	t.TLSClientConfig = Config()
	return t
}

// Client is an HTTP client with Transport (timeout 0: none).
func Client(timeout time.Duration) *http.Client {
	return &http.Client{Transport: Transport(), Timeout: timeout}
}

// Explain turns a certificate from outside the trusted roots into a message that says what it likely means.
func Explain(host string, err error) error {
	var ua x509.UnknownAuthorityError
	var ci x509.CertificateInvalidError
	var hn x509.HostnameError
	if errors.As(err, &ua) || errors.As(err, &ci) || errors.As(err, &hn) {
		return fmt.Errorf("refusing %s: its certificate is not from a public CA (%v). Something is intercepting TLS "+
			"between here and the server (a corporate inspector, a proxy, an injected CA); nothing was sent. "+
			"REMOTE_POD_TLS_ROOTS=system accepts it, if you trust that network", host, err)
	}
	return err
}
