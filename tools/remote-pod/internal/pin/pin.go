// Package pin secures the agent's connection where no provider proxy terminates TLS (vast.ai publishes the agent's
// port on the host's public IP): `up` makes a self-signed certificate for each pod, the pod's agent serves TLS with
// it, and the CLI trusts that one certificate only (pinned: no CA, no hostname). The private key goes to the pod in
// its environment, like the token; the certificate stays in .remote-pod/state.json.
package pin

import (
	"crypto/ed25519"
	"crypto/tls"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/pem"
	"errors"
	"math/big"
	"time"

	"github.com/SamuelHaidu/rnn-research/tools/remote-pod/internal/entropy"
)

// ServerName is the certificate's only name; the CLI verifies against it (the pod's IP is not known in advance).
const ServerName = "remote-pod-agent"

// New makes an Ed25519 key (from a seed of internal/entropy: hardware and kernel randomness) and a self-signed
// certificate for ServerName, valid for a year; both PEM-encoded.
func New() (certPEM, keyPEM []byte, err error) {
	key := ed25519.NewKeyFromSeed(entropy.Bytes(ed25519.SeedSize))
	serial := new(big.Int).SetBytes(entropy.Bytes(15))
	tmpl := &x509.Certificate{
		SerialNumber: serial,
		Subject:      pkix.Name{CommonName: ServerName},
		DNSNames:     []string{ServerName},
		NotBefore:    time.Now().Add(-time.Hour),
		NotAfter:     time.Now().Add(365 * 24 * time.Hour),
		KeyUsage:     x509.KeyUsageDigitalSignature,
		ExtKeyUsage:  []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth},
	}
	der, err := x509.CreateCertificate(entropy.Reader, tmpl, tmpl, key.Public(), key) // Ed25519 signs deterministically
	if err != nil {
		return nil, nil, err
	}
	kd, err := x509.MarshalPKCS8PrivateKey(key)
	if err != nil {
		return nil, nil, err
	}
	return pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: der}),
		pem.EncodeToMemory(&pem.Block{Type: "PRIVATE KEY", Bytes: kd}), nil
}

// ClientConfig trusts the given certificate only.
func ClientConfig(certPEM []byte) (*tls.Config, error) {
	pool := x509.NewCertPool()
	if !pool.AppendCertsFromPEM(certPEM) {
		return nil, errors.New("pin: no certificate in the PEM")
	}
	return &tls.Config{RootCAs: pool, ServerName: ServerName, MinVersion: tls.VersionTLS13}, nil
}

// ServerConfig serves the given certificate and key.
func ServerConfig(certPEM, keyPEM []byte) (*tls.Config, error) {
	c, err := tls.X509KeyPair(certPEM, keyPEM)
	if err != nil {
		return nil, err
	}
	return &tls.Config{Certificates: []tls.Certificate{c}, MinVersion: tls.VersionTLS13}, nil
}
