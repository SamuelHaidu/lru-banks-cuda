package main

// Pods never get the store's key (package grant). At creation the CLI gives a pod a random tag, a policy (kept in
// .remote-pod/state.json, on this machine only) and a few long-lived grants for what must work while no signer is
// connected; from then on the CLI signs the pod's requests over the agent's connection: `remote-pod signer`, and
// every command that follows the pod (train, logs -f).

import (
	"context"
	"encoding/hex"
	"errors"
	"fmt"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"time"

	"github.com/spf13/cobra"

	"github.com/SamuelHaidu/rnn-research/tools/remote-pod/internal/client"
	"github.com/SamuelHaidu/rnn-research/tools/remote-pod/internal/config"
	"github.com/SamuelHaidu/rnn-research/tools/remote-pod/internal/entropy"
	"github.com/SamuelHaidu/rnn-research/tools/remote-pod/internal/grant"
	"github.com/SamuelHaidu/rnn-research/tools/remote-pod/internal/store"
	"github.com/SamuelHaidu/rnn-research/tools/remote-pod/internal/ui"
)

// newTag is a pod's name in the store: random, so one pod's grants cannot reach another's logs.
func newTag() string {
	return hex.EncodeToString(entropy.Bytes(8))
}

// mainStore opens the store with its key (this machine only).
func mainStore(cfg *config.Config) (*store.Store, error) {
	if !cfg.HasR2() {
		return nil, hint(errors.New("no store configured: the signer needs the R2 key"), "set [r2] in "+config.Path())
	}
	return store.New(store.Config{Endpoint: cfg.R2.Endpoint, Bucket: cfg.R2.Bucket, AccessKey: cfg.R2.AccessKey,
		SecretKey: cfg.R2.SecretKey, Region: cfg.R2.Region})
}

// serveSigner signs the project pod's requests in the background until ctx ends (quietly: refusals are logged).
func serveSigner(ctx context.Context, cfg *config.Config, st *config.State, cl *client.Client, verbose bool) error {
	if len(st.Policy.PutPrefixes) == 0 && len(st.Policy.Get) == 0 {
		return nil // a pod made before grants: nothing to sign
	}
	ms, err := mainStore(cfg)
	if err != nil {
		return err
	}
	// every decision is journaled (the audit trail; the pod's budgets survive a restart of the signer)
	journal := &grant.FileJournal{Path: filepath.Join(config.ProjectRoot(), ".remote-pod", "grants-"+st.Tag+".jsonl")}
	s := &grant.Signer{Policy: st.Policy, Store: ms, Journal: journal}
	s.OnGrant = func(r grant.Request, err error) {
		switch {
		case errors.Is(err, grant.ErrSuspended):
		case err != nil:
			ui.Warning("refused %s %s: %v", r.Op, r.Key, err)
			if until := s.Suspended(); !until.IsZero() {
				ui.Warning("pod %s suspended until %s: it keeps asking for what it may not have. Is it compromised? "+
					"remote-pod down ends it; the journal is %s", st.PodID, until.Format(time.TimeOnly), journal.Path)
			}
		case verbose:
			ui.Note("granted %s %s (%d bytes)", r.Op, r.Key, r.Size)
		}
	}
	go cl.ServeSigner(ctx, s.Sign, func(connected bool, err error) {
		if verbose {
			switch {
			case connected:
				ui.Note("signer connected to the pod")
			case err != nil:
				ui.Note("signer: %v (retrying)", err)
			}
		}
	})
	return nil
}

func signerCommand() *cobra.Command {
	var allowPut, allowGet []string
	cmd := &cobra.Command{
		Use:   "signer",
		Short: "Sign the pod's store requests (it holds no key): keep running while the pod works",
		Long: `The pod has no store key. Whatever it uploads (checkpoints, logs) or downloads (datasets) goes through a
presigned URL for that one object, valid for 15 minutes, which this command signs on this machine when the pod asks
(over the agent's encrypted connection) and only when the pod's policy allows it: downloads of its own inputs,
uploads under its own prefixes (runs/<run>/, jobs/<tag>/). Never a listing, a delete, or another run's files.

train and logs -f sign while they follow; this command is for the rest of the time (a run left alone, scripts on the
pod using $REMOTE_POD_GRANT_URL, e.g. engine/scripts/r2.py). Without a signer the pod keeps working: its uploads wait
and retry, and what was pre-issued at creation (its inputs, its run's fixed-name files) works for 7 days.

--allow-put / --allow-get widen the policy (saved in .remote-pod/state.json, on this machine: the pod cannot).`,
		Example: `  remote-pod signer                                   sign for the pod until Ctrl-C
  remote-pod signer --allow-put runs/my-run/          a run started by hand on the pod may upload there
  remote-pod signer --allow-get datasets/finewiki/data.tar.zst`,
		GroupID: "work",
		RunE: func(cmd *cobra.Command, args []string) error {
			cfg, err := config.Load()
			if err != nil {
				return err
			}
			cl, st, err := agentClient()
			if err != nil {
				return err
			}
			if st.Tag == "" {
				return hint(errors.New("this pod was made before grants (it has no tag or policy)"), "remote-pod down, then up again")
			}
			if len(allowPut)+len(allowGet) > 0 {
				var puts []string
				for _, pre := range st.Policy.PutPrefixes {
					if pre != grant.JobLogPrefix(st.Tag) {
						puts = append(puts, pre)
					}
				}
				p, err := grant.PodPolicy(st.Tag, append(puts, allowPut...), append(append([]string{}, st.Policy.Get...), allowGet...))
				if err != nil {
					return err
				}
				st.Policy = p
				if err := config.SaveState(st); err != nil {
					return err
				}
			}
			ui.Note("policy: uploads under %s; downloads of %d key(s)", strings.Join(st.Policy.PutPrefixes, ", "), len(st.Policy.Get))
			ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
			defer stop()
			if err := serveSigner(ctx, cfg, st, cl, true); err != nil {
				return err
			}
			<-ctx.Done()
			fmt.Fprintln(os.Stderr)
			ui.Note("signer stopped at %s", time.Now().Format("15:04:05"))
			return nil
		},
	}
	cmd.Flags().StringArrayVar(&allowPut, "allow-put", nil, "also allow uploads under this prefix (repeatable)")
	cmd.Flags().StringArrayVar(&allowGet, "allow-get", nil, "also allow downloading this exact key (repeatable)")
	return cmd
}
