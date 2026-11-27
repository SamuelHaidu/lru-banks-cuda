package main

import (
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"time"

	"github.com/spf13/cobra"

	"github.com/SamuelHaidu/rnn-research/tools/remote-pod/internal/config"
	"github.com/SamuelHaidu/rnn-research/tools/remote-pod/internal/sshkey"
	"github.com/SamuelHaidu/rnn-research/tools/remote-pod/internal/ui"
)

func sshCommand() *cobra.Command {
	return &cobra.Command{
		Use:   "ssh [-- COMMAND...]",
		Short: "SSH into the pod with a key made for this one connection (vast.ai)",
		Long: `Open an SSH session (or run COMMAND) on the project's vast.ai instance. Every run makes a new ed25519 key pair
with no comment, attaches its public half to this instance only, connects offering that key and nothing else (no
key from ~/.ssh, no agent, nothing forwarded, no password), and when the session ends detaches it and deletes the
private half. The host never sees the same key twice, so it cannot link your pods to each other or to you. Keep no
SSH key in your vast.ai or RunPod account: providers copy account keys into every machine.

The host's own key is pinned per pod on first use (.remote-pod/ssh/known_hosts-<pod>). remote-pod sh does the same
work without SSH (the agent's TLS connection); RunPod pods here run no SSH server.`,
		Example: "  remote-pod ssh\n  remote-pod ssh -- nvidia-smi",
		GroupID: "work",
		RunE: func(cmd *cobra.Command, args []string) error {
			cfg, err := config.Load()
			if err != nil {
				return err
			}
			st, _ := config.LoadState()
			if st.PodID == "" {
				return hint(errors.New("no pod for this project"), "remote-pod up --provider vast --gpu 4090")
			}
			if st.ProviderName() != "vast" {
				return hint(errors.New("RunPod pods made by remote-pod run no SSH server"), "remote-pod sh (the same, over the agent)")
			}
			cl, err := cloudFor(cfg, "vast")
			if err != nil {
				return err
			}
			c := cl.(vastCloud).c
			id, err := vastID(st.PodID)
			if err != nil {
				return err
			}
			ctx := context.Background()
			in, err := c.Get(ctx, id)
			if err != nil {
				return err
			}
			port := in.HostPort(22)
			if in.PublicIP == "" || port == 0 {
				return hint(errors.New("the instance publishes no SSH port yet"), "remote-pod status")
			}
			key, err := sshkey.New("")
			if err != nil {
				return err
			}
			defer key.Close()
			keyID, err := c.AttachSSH(ctx, id, key.Public)
			if err != nil {
				return fmt.Errorf("attaching this connection's key: %w", err)
			}
			defer func() {
				if keyID == "" {
					ui.Note("vast.ai gave the key no id to detach: it stays on the instance (its private half is deleted)")
					return
				}
				if err := c.DetachSSH(context.Background(), id, keyID); err != nil {
					ui.Note("detaching the key: %v (its private half is deleted anyway)", err)
				}
			}()
			khDir := filepath.Join(config.ProjectRoot(), ".remote-pod", "ssh")
			if err := os.MkdirAll(khDir, 0o700); err != nil {
				return err
			}
			sshArgs := key.Args(filepath.Join(khDir, "known_hosts-"+st.PodID), "root", in.PublicIP, port)
			// the key reaches the instance's authorized_keys within seconds: wait for it with a no-op login
			deadline := time.Now().Add(90 * time.Second)
			for {
				probe := exec.Command("ssh", append(append([]string{"-o", "BatchMode=yes", "-o", "LogLevel=ERROR"}, sshArgs...), "true")...)
				if probe.Run() == nil {
					break
				}
				if time.Now().After(deadline) {
					return errors.New("the instance did not accept this connection's key within 90 s")
				}
				time.Sleep(3 * time.Second)
			}
			full := append(append([]string{}, sshArgs...), args...)
			if len(args) == 0 {
				full = append([]string{"-t"}, full...)
			}
			s := exec.Command("ssh", full...)
			s.Stdin, s.Stdout, s.Stderr = os.Stdin, os.Stdout, os.Stderr
			if err := s.Run(); err != nil {
				var ee *exec.ExitError
				if errors.As(err, &ee) {
					return exitError{ee.ExitCode()}
				}
				return err
			}
			return nil
		},
	}
}
