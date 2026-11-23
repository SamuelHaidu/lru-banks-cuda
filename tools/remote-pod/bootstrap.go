package main

// Starting the agent on a public image (nvidia/cuda, buildpack-deps, ...) instead of one of ours: nothing large is
// pushed from here. The agent binary (~15 MB) goes to R2 once per build; the pod downloads it at boot through a
// presigned URL (a signature that expires, no credentials) and runs it. Public images are cached on RunPod's hosts,
// so the pod also starts faster.

import (
	"context"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"github.com/spf13/cobra"

	"github.com/SamuelHaidu/rnn-research/tools/remote-pod/internal/config"
	"github.com/SamuelHaidu/rnn-research/tools/remote-pod/internal/store"
)

// bootstrapScript downloads the agent with whatever the image has (curl, wget, python3, else apt installs curl) and
// replaces itself with it. A restart runs it again, so the URL must still be valid then (bootstrapTTL).
const bootstrapScript = `set -e
fetch() {
  if command -v curl >/dev/null; then curl -fsSL "$1" -o "$2"
  elif command -v wget >/dev/null; then wget -qO "$2" "$1"
  elif command -v python3 >/dev/null; then python3 -c 'import sys, urllib.request; urllib.request.urlretrieve(sys.argv[1], sys.argv[2])' "$1" "$2"
  else apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq curl ca-certificates >/dev/null && curl -fsSL "$1" -o "$2"
  fi
}
fetch "$REMOTE_POD_AGENT_URL" /usr/local/bin/remote-pod
chmod +x /usr/local/bin/remote-pod
mkdir -p "$REMOTE_POD_WORKDIR"
cd "$REMOTE_POD_WORKDIR"
exec /usr/local/bin/remote-pod agent
`

const bootstrapTTL = 7 * 24 * time.Hour // R2's (S3's) longest presigned URL

// isOurImage reports whether ref is one of the configured images (their entrypoint is already the agent).
func isOurImage(cfg *config.Config, ref string) bool {
	return ref == cfg.Images.Train || ref == cfg.Images.Dev || ref == cfg.Images.CPU
}

// bootstrap builds the agent, uploads it to R2 (once per content) and returns the pod's command and extra env.
func bootstrap(ctx context.Context, cfg *config.Config) (dockerArgs string, env map[string]string, err error) {
	if !cfg.HasR2() {
		return "", nil, hint(fmt.Errorf("a public image needs R2 (the pod downloads the agent from it)"),
			"set [r2] in "+config.Path())
	}
	dir := filepath.Join(config.ProjectRoot(), "tools", "remote-pod")
	bin := filepath.Join(dir, "dist", "remote-pod")
	gb := exec.Command("go", "build", "-trimpath", "-ldflags", "-s -w", "-o", bin, ".")
	gb.Dir = dir
	gb.Env = append(os.Environ(), "CGO_ENABLED=0", "GOOS=linux", "GOARCH=amd64")
	gb.Stdout, gb.Stderr = os.Stderr, os.Stderr
	if err := gb.Run(); err != nil {
		return "", nil, fmt.Errorf("building the agent: %w", err)
	}
	sum, err := fileSHA256(bin)
	if err != nil {
		return "", nil, err
	}
	st, err := store.New(store.Config{Endpoint: cfg.R2.Endpoint, Bucket: cfg.R2.Bucket, AccessKey: cfg.R2.AccessKey,
		SecretKey: cfg.R2.SecretKey, Region: cfg.R2.Region})
	if err != nil {
		return "", nil, err
	}
	key := "bin/remote-pod-" + sum[:16]
	if !st.Exists(ctx, key) {
		if err := st.PutFile(ctx, key, bin); err != nil {
			return "", nil, fmt.Errorf("uploading the agent to R2: %w", err)
		}
	}
	url, err := st.Presign(ctx, key, bootstrapTTL)
	if err != nil {
		return "", nil, err
	}
	script := base64.StdEncoding.EncodeToString([]byte(bootstrapScript))
	env = map[string]string{"REMOTE_POD_AGENT_URL": url, "REMOTE_POD_WORKDIR": "/work", "REMOTE_POD_BOOTSTRAP": script}
	return `bash -c 'echo "$REMOTE_POD_BOOTSTRAP" | base64 -d | bash'`, env, nil
}

func fileSHA256(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", err
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

// bootstrapEnvCommand writes the bootstrap's env to a docker --env-file, to test a public image's boot locally:
//
//	remote-pod bootstrap-env /tmp/boot.env && docker run --env-file /tmp/boot.env -e REMOTE_POD_TOKEN=t -p 8080:8080 \
//	    ubuntu:24.04 bash -c 'echo "$REMOTE_POD_BOOTSTRAP" | base64 -d | bash'
func bootstrapEnvCommand() *cobra.Command {
	return &cobra.Command{
		Use:    "bootstrap-env FILE",
		Short:  "Write the public-image bootstrap's env to FILE (docker --env-file), for a local test",
		Hidden: true,
		Args:   cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			cfg, err := config.Load()
			if err != nil {
				return err
			}
			_, env, err := bootstrap(context.Background(), cfg)
			if err != nil {
				return err
			}
			var b strings.Builder
			for k, v := range env {
				fmt.Fprintf(&b, "%s=%s\n", k, v)
			}
			return os.WriteFile(args[0], []byte(b.String()), 0o600)
		},
	}
}
