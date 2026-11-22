// remote-pod: work on a RunPod GPU as if it were this machine. See `remote-pod --help` (the help is the manual).
package main

import (
	"errors"
	"os"

	"github.com/spf13/cobra"

	"github.com/SamuelHaidu/rnn-research/tools/remote-pod/internal/proto"
	"github.com/SamuelHaidu/rnn-research/tools/remote-pod/internal/ui"
)

// exitError carries a remote exit code through cobra (sh returns the command's own code).
type exitError struct{ code int }

func (e exitError) Error() string { return "" }

// hinted is an error with the next command to try.
type hinted struct {
	err  error
	hint string
}

func (h hinted) Error() string { return h.err.Error() }
func hint(err error, next string) error {
	if err == nil {
		return nil
	}
	return hinted{err, next}
}

const overview = `Work on a RunPod GPU as if it were this machine: run commands, copy files, sync code,
train with datasets and checkpoints in R2, follow logs that survive disconnects.

Quick start
  remote-pod gpus                          what is in stock, cheapest first
  remote-pod up --gpu "RTX 4090"           a pod with our image (community first, then secure)
  remote-pod sh nvidia-smi                 a command on the pod: output, exit code, Ctrl-C as if local
  remote-pod cp ./bench.py :               copy to the pod (":" marks the pod side; ":" alone is the workdir)
  remote-pod sync engine                   send only the changed files of a source tree
  remote-pod sh "cd engine && cmake --build build/release"
  remote-pod sh -d python3 long_job.py     a detached job: survives disconnects
  remote-pod logs <job> -f                 follow it (reconnects and resumes on its own)
  remote-pod stop                          stop paying for the GPU

One command for a training run (datasets and checkpoints in R2; the pod stops itself at the end):
  remote-pod dataset add --name pretrain-mix --from datasets/pretrain-mix-bpe32768
  remote-pod train runs/my-run/config.json --gpu "RTX 5090"
  remote-pod pull my-run

Settings: ~/.config/remote-pod/config.toml (RunPod key, R2 endpoint / bucket / keys, images);
the current pod: .remote-pod/state.json in the repository. Every command takes --json for scripts and agents.`

func main() {
	root := &cobra.Command{
		Use:           "remote-pod",
		Short:         "Work on a RunPod GPU as if it were this machine",
		Long:          overview,
		Version:       proto.Version,
		SilenceUsage:  true,
		SilenceErrors: true,
	}
	root.PersistentFlags().BoolVar(&ui.JSON, "json", false, "print machine-readable JSON instead of tables")
	root.AddGroup(
		&cobra.Group{ID: "pod", Title: ui.Title("Pod")},
		&cobra.Group{ID: "work", Title: ui.Title("Work on the pod")},
		&cobra.Group{ID: "data", Title: ui.Title("Datasets and runs (R2)")},
		&cobra.Group{ID: "build", Title: ui.Title("Build")},
		&cobra.Group{ID: "internal", Title: ui.Title("Inside the pod")},
	)
	root.AddCommand(podCommands()...)
	root.AddCommand(workCommands()...)
	root.AddCommand(dataCommands()...)
	root.AddCommand(buildCommands()...)
	root.AddCommand(agentCommand())
	root.SetHelpCommandGroupID("pod")
	root.SetCompletionCommandGroupID("build")

	if err := root.Execute(); err != nil {
		var ee exitError
		if errors.As(err, &ee) {
			os.Exit(ee.code)
		}
		var h hinted
		if errors.As(err, &h) {
			ui.Fail(h.err, h.hint)
		} else {
			ui.Fail(err, "")
		}
		os.Exit(1)
	}
}
