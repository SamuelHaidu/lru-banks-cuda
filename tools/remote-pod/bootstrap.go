package main

// Starting the agent on a public image (nvidia/cuda, buildpack-deps, ...) instead of one of ours: nothing large is
// pushed from here. The agent binary (~15 MB) goes to R2 once per build; the pod downloads it at boot through a
// presigned URL (a signature that expires, no credentials) and runs it. Public images are cached on the hosts, so the
// pod also starts faster. When engine/build/dist/lbc exists (remote-pod build), it goes the same way and lands in
// /usr/local/bin/lbc, so train works on a public CUDA image too (vast.ai's default: its hosts cannot pull our private
// images).

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

	"github.com/SamuelHaidu/rnn-research/tools/remote-pod/internal/boot"
	"github.com/SamuelHaidu/rnn-research/tools/remote-pod/internal/config"
	"github.com/SamuelHaidu/rnn-research/tools/remote-pod/internal/store"
)

const bootstrapTTL = 7 * 24 * time.Hour // R2's (S3's) longest presigned URL

// isOurImage reports whether ref is one of the configured images (their entrypoint is already the agent).
func isOurImage(cfg *config.Config, ref string) bool {
	return ref == cfg.Images.Train || ref == cfg.Images.Dev || ref == cfg.Images.CPU
}

// uploadPresigned puts a local file in R2 under prefix + its content hash (once) and returns a presigned URL to it
// and the file's sha256 (the pod checks the download against it).
func uploadPresigned(ctx context.Context, st *store.Store, prefix, path string) (url, sum string, err error) {
	if sum, err = fileSHA256(path); err != nil {
		return "", "", err
	}
	key := prefix + sum[:16]
	if !st.Exists(ctx, key) {
		if err := st.PutFile(ctx, key, path); err != nil {
			return "", "", fmt.Errorf("uploading %s to R2: %w", filepath.Base(path), err)
		}
	}
	url, err = st.Presign(ctx, key, bootstrapTTL)
	return url, sum, err
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
	st, err := store.New(store.Config{Endpoint: cfg.R2.Endpoint, Bucket: cfg.R2.Bucket, AccessKey: cfg.R2.AccessKey,
		SecretKey: cfg.R2.SecretKey, Region: cfg.R2.Region})
	if err != nil {
		return "", nil, err
	}
	url, sum, err := uploadPresigned(ctx, st, "bin/remote-pod-", bin)
	if err != nil {
		return "", nil, err
	}
	script := base64.StdEncoding.EncodeToString([]byte(boot.Script))
	env = map[string]string{"REMOTE_POD_AGENT_URL": url, "REMOTE_POD_AGENT_SHA256": sum, "REMOTE_POD_WORKDIR": "/work",
		"REMOTE_POD_BOOTSTRAP": script}
	if lbc := filepath.Join(config.ProjectRoot(), "engine", "build", "dist", "lbc"); fileExists(lbc) {
		if env["REMOTE_POD_LBC_URL"], env["REMOTE_POD_LBC_SHA256"], err = uploadPresigned(ctx, st, "bin/lbc-", lbc); err != nil {
			return "", nil, err
		}
	}
	return `bash -c 'echo "$REMOTE_POD_BOOTSTRAP" | base64 -d | bash'`, env, nil
}

func fileExists(path string) bool {
	fi, err := os.Stat(path)
	return err == nil && !fi.IsDir()
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
