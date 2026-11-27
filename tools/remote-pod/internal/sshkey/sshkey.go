// Package sshkey makes the throwaway SSH identity of one connection: a fresh ed25519 pair, no comment (nothing that
// names you), the private half in a private temporary directory deleted when the connection ends. A pod's host sees
// a public key it has never seen before and will never see again: it cannot link pods to each other or to you
// (an account key, or one also on GitHub, would do both).
package sshkey

import (
	"crypto/ed25519"
	"encoding/pem"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"golang.org/x/crypto/ssh"

	"github.com/SamuelHaidu/rnn-research/tools/remote-pod/internal/entropy"
)

// Key is one throwaway identity.
type Key struct {
	Dir     string // private directory (0700) holding the private key: Close deletes it
	Private string // path of the private key (0600)
	Public  string // "ssh-ed25519 AAAA..." with no comment
}

// New makes a key in a new private directory under base ("" = the system's temp dir).
func New(base string) (*Key, error) {
	// the seed is ours (internal/entropy: the CPU's and TPM's hardware generators and the kernel's, hashed): key
	// generation functions may ignore a caller's random source, a seed cannot be ignored
	priv := ed25519.NewKeyFromSeed(entropy.Bytes(ed25519.SeedSize))
	pub := priv.Public().(ed25519.PublicKey)
	sp, err := ssh.NewPublicKey(pub)
	if err != nil {
		return nil, err
	}
	blk, err := ssh.MarshalPrivateKey(priv, "") // empty comment
	if err != nil {
		return nil, err
	}
	dir, err := os.MkdirTemp(base, "rp-ssh-")
	if err != nil {
		return nil, err
	}
	if err := os.Chmod(dir, 0o700); err != nil {
		_ = os.RemoveAll(dir)
		return nil, err
	}
	k := &Key{Dir: dir, Private: filepath.Join(dir, "id"),
		Public: strings.TrimSpace(string(ssh.MarshalAuthorizedKey(sp)))}
	if err := os.WriteFile(k.Private, pem.EncodeToMemory(blk), 0o600); err != nil {
		_ = os.RemoveAll(dir)
		return nil, err
	}
	return k, nil
}

// Close deletes the private key (overwritten first) and its directory.
func (k *Key) Close() error {
	if fi, err := os.Stat(k.Private); err == nil {
		_ = os.WriteFile(k.Private, make([]byte, fi.Size()), 0o600)
	}
	return os.RemoveAll(k.Dir)
}

// Args are the ssh options of a connection with this key only: no other identity is offered (not from ~/.ssh, not
// from the agent), nothing is forwarded (agent, X11, ports), no password or keyboard prompt, the host's key is pinned
// per pod on first use (knownHosts: that pod's own file), and ~/.ssh/config cannot add anything (-F none).
func (k *Key) Args(knownHosts, user, host string, port int) []string {
	return []string{"-F", "none", "-i", k.Private, "-p", fmt.Sprint(port),
		"-o", "IdentitiesOnly=yes", "-o", "IdentityAgent=none",
		"-o", "ForwardAgent=no", "-o", "ForwardX11=no", "-o", "ForwardX11Trusted=no", "-o", "ClearAllForwardings=yes",
		"-o", "PasswordAuthentication=no", "-o", "KbdInteractiveAuthentication=no", "-o", "PubkeyAuthentication=yes",
		"-o", "PermitLocalCommand=no", "-o", "UpdateHostKeys=no", "-o", "HashKnownHosts=yes",
		"-o", "StrictHostKeyChecking=accept-new", "-o", "UserKnownHostsFile=" + knownHosts,
		"-o", "GlobalKnownHostsFile=/dev/null", "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=30",
		user + "@" + host}
}
