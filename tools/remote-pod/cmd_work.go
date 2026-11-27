package main

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"os/signal"
	"path/filepath"
	"regexp"
	"strings"
	"syscall"
	"time"

	"github.com/spf13/cobra"
	"golang.org/x/term"

	"github.com/SamuelHaidu/rnn-research/tools/remote-pod/internal/client"
	"github.com/SamuelHaidu/rnn-research/tools/remote-pod/internal/config"
	"github.com/SamuelHaidu/rnn-research/tools/remote-pod/internal/proto"
	"github.com/SamuelHaidu/rnn-research/tools/remote-pod/internal/syncx"
	"github.com/SamuelHaidu/rnn-research/tools/remote-pod/internal/ui"
)

func printJobs(v any) {
	js, ok := v.([]proto.Job)
	if !ok || len(js) == 0 {
		return
	}
	var rows [][]string
	for _, j := range js {
		state := "RUNNING"
		took := time.Since(j.Started)
		if !j.Running {
			state = fmt.Sprintf("ENDED %d", j.ExitCode)
			if j.ExitCode != 0 {
				state = fmt.Sprintf("FAILED %d", j.ExitCode)
			}
			took = j.Ended.Sub(j.Started)
		}
		cmd := j.Command
		if len(cmd) > 48 {
			cmd = cmd[:45] + "..."
		}
		rows = append(rows, []string{j.ID, j.Name, ui.State(strings.Fields(state)[0]) + strings.TrimPrefix(state, strings.Fields(state)[0]),
			took.Round(time.Second).String(), ui.Bytes(j.LogBytes), cmd})
	}
	ui.Table([]string{"Job", "Name", "State", "Time", "Log", "Command"}, rows)
}

// isPod tells a pod path (":path", ":" = the workdir) from a local one.
func isPod(p string) (string, bool) {
	if strings.HasPrefix(p, ":") {
		return strings.TrimPrefix(p, ":"), true
	}
	return p, false
}

func workCommands() []*cobra.Command {
	var tty, detach bool
	var cwd, name string
	var envs []string
	sh := &cobra.Command{
		Use:   "sh [flags] COMMAND...",
		Short: "Run a command on the pod as if it were local (output, exit code, stdin, Ctrl-C)",
		Long: `Run a command on the pod through bash -lc. Its stdout and stderr come back separately, the exit code is
returned as this command's own (so set -e, &&, || and loops work), piped stdin is forwarded, Ctrl-C reaches the
remote command. The working directory is the pod's workdir (/work) unless --cwd says otherwise.

With --detach the command becomes a job that survives disconnects: follow it with remote-pod logs.`,
		Example: `  remote-pod sh nvidia-smi
  remote-pod sh "cd engine && cmake --build build/release && ./build/release/lbc bench --batch 64"
  for b in 64 128 256; do remote-pod sh ./lbc bench --batch $b | tee -a bench.txt; done
  cat cfg.json | remote-pod sh "cat > runs/x/config.json"
  remote-pod sh -t htop                                interactive (a terminal)
  remote-pod sh -d --name tok python3 tokenize.py      a job; then: remote-pod logs tok -f`,
		GroupID:            "work",
		Args:               cobra.MinimumNArgs(1),
		DisableFlagParsing: false,
		RunE: func(cmd *cobra.Command, args []string) error {
			cl, _, err := agentClient()
			if err != nil {
				return err
			}
			env := map[string]string{}
			for _, kv := range envs {
				k, v, ok := strings.Cut(kv, "=")
				if !ok {
					return fmt.Errorf("--env wants KEY=VALUE (got %q)", kv)
				}
				env[k] = v
			}
			start := proto.ExecStart{Command: strings.Join(args, " "), Cwd: cwd, Env: env, TTY: tty, Name: name}
			if detach {
				j, err := cl.StartJob(start)
				if err != nil {
					return err
				}
				ui.Print(j, func() {
					ui.Success("job %s started (%s)", j.ID, j.Name)
					fmt.Fprintln(os.Stderr, ui.Dim("  follow: ")+"remote-pod logs "+j.Name+" -f")
				})
				return nil
			}
			return runExec(cl, start)
		},
	}
	sh.Flags().SetInterspersed(false) // flags after the command belong to it: remote-pod sh ls -la
	sh.Flags().BoolVarP(&tty, "tty", "t", false, "a terminal on the pod (interactive programs, progress bars)")
	sh.Flags().BoolVarP(&detach, "detach", "d", false, "run as a job that survives disconnects")
	sh.Flags().StringVar(&name, "name", "", "the job's name (with --detach; default: its id)")
	sh.Flags().StringVarP(&cwd, "cwd", "C", "", "working directory on the pod (default: the workdir, /work)")
	sh.Flags().StringArrayVarP(&envs, "env", "e", nil, "environment variable KEY=VALUE (repeatable)")

	var excludes []string
	cp := &cobra.Command{
		Use:   "cp FROM TO",
		Short: `Copy files or directories to or from the pod (":" marks the pod side)`,
		Long: `Copy a file or a directory between this machine and the pod, as cp -r would: into TO when TO is a
directory (or ends with /), else as TO. The pod side starts with ":"; relative pod paths are under the workdir
(/work), ":" alone is the workdir itself.`,
		Example: `  remote-pod cp ./bench.py :                       to /work/bench.py
  remote-pod cp configs :runs/new/                  a directory into /work/runs/new/
  remote-pod cp :engine/prof.nsys-rep .             a profile back here
  remote-pod cp :/work/runs/x/metrics.csv x.csv`,
		GroupID: "work",
		Args:    cobra.ExactArgs(2),
		RunE: func(cmd *cobra.Command, args []string) error {
			cl, _, err := agentClient()
			if err != nil {
				return err
			}
			from, fromPod := isPod(args[0])
			to, toPod := isPod(args[1])
			if fromPod == toPod {
				return errors.New(`exactly one side is the pod: mark it with ":" (remote-pod cp ./x :, remote-pod cp :x .)`)
			}
			t0, p := time.Now(), &ui.Progress{Label: "copying"}
			var target string
			if toPod {
				target, err = cl.Upload(from, to, append(append([]string{}, syncx.DefaultExcludes[:2]...), excludes...), p.Update)
			} else {
				target, err = cl.Download(from, to, p.Update)
			}
			p.Done()
			if err != nil {
				return err
			}
			side := "here"
			if toPod {
				side = "on the pod"
			}
			ui.Print(map[string]string{"target": target}, func() { ui.Success("%s %s (%s)", target, side, time.Since(t0).Round(time.Millisecond)) })
			return nil
		},
	}
	cp.Flags().StringArrayVar(&excludes, "exclude", nil, "skip names matching this pattern when sending a directory (repeatable)")

	var del, noDefaults bool
	var syncEx []string
	syncCmd := &cobra.Command{
		Use:   "sync LOCAL_DIR [:POD_DIR]",
		Short: "Send only the changed files of a directory (the edit, build, run loop)",
		Long: `Make POD_DIR match LOCAL_DIR for every file that differs (by content hash); only those are sent.
Editing one kernel and syncing sends one file. POD_DIR defaults to the directory's name under the workdir.
Skipped by default: ` + strings.Join(syncx.DefaultExcludes, ", ") + ` (--no-default-excludes to send them).`,
		Example: `  remote-pod sync engine                         to /work/engine
  remote-pod sync engine && remote-pod sh "cd engine && cmake --build build/release && ./build/release/lbc bench"
  remote-pod sync model :/work/rnn/model --delete   also delete pod files that are not here`,
		GroupID: "work",
		Args:    cobra.RangeArgs(1, 2),
		RunE: func(cmd *cobra.Command, args []string) error {
			cl, _, err := agentClient()
			if err != nil {
				return err
			}
			local := args[0]
			remote := filepath.Base(filepath.Clean(local))
			if len(args) == 2 {
				r, ok := isPod(args[1])
				if !ok {
					return errors.New(`the second argument is the pod directory: mark it with ":" (:/work/x)`)
				}
				remote = r
			}
			ex := syncEx
			if !noDefaults {
				ex = append(append([]string{}, syncx.DefaultExcludes...), syncEx...)
			}
			t0 := time.Now()
			sent, deleted, n, err := cl.Sync(local, remote, ex, del)
			if err != nil {
				return err
			}
			out := map[string]any{"sent": sent, "deleted": deleted, "bytes": n, "seconds": time.Since(t0).Seconds()}
			ui.Print(out, func() {
				if len(sent) == 0 && len(deleted) == 0 {
					ui.Success("%s is up to date on the pod (%s)", local, time.Since(t0).Round(time.Millisecond))
					return
				}
				for _, f := range sent {
					fmt.Fprintln(os.Stderr, ui.OK("  + ")+f)
				}
				for _, f := range deleted {
					fmt.Fprintln(os.Stderr, ui.Bad("  - ")+f)
				}
				ui.Success("synced %d file(s), %s, deleted %d (%s)", len(sent), ui.Bytes(n), len(deleted), time.Since(t0).Round(time.Millisecond))
			})
			return nil
		},
	}
	syncCmd.Flags().BoolVar(&del, "delete", false, "delete pod files that are not in LOCAL_DIR")
	syncCmd.Flags().StringArrayVar(&syncEx, "exclude", nil, "skip names matching this pattern (repeatable)")
	syncCmd.Flags().BoolVar(&noDefaults, "no-default-excludes", false, "also send .git, build, node_modules, ...")

	jobs := &cobra.Command{
		Use:     "jobs",
		Short:   "The pod's detached jobs",
		Example: "  remote-pod jobs",
		GroupID: "work",
		RunE: func(cmd *cobra.Command, args []string) error {
			cl, _, err := agentClient()
			if err != nil {
				return err
			}
			js, err := cl.Jobs()
			if err != nil {
				return err
			}
			ui.Print(js, func() {
				if len(js) == 0 {
					fmt.Println(ui.Dim("No jobs.") + "  remote-pod sh -d COMMAND")
				}
				printJobs(js)
			})
			return nil
		},
	}

	var sig string
	kill := &cobra.Command{
		Use:     "kill JOB",
		Short:   "Send a signal to a job (default INT, as Ctrl-C)",
		Example: "  remote-pod kill tok\n  remote-pod kill tok --signal KILL",
		GroupID: "work",
		Args:    cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			cl, _, err := agentClient()
			if err != nil {
				return err
			}
			if err := cl.SignalJob(args[0], strings.ToUpper(sig)); err != nil {
				return err
			}
			ui.Success("sent %s to %s", strings.ToUpper(sig), args[0])
			return nil
		},
	}
	kill.Flags().StringVar(&sig, "signal", "INT", "INT, TERM, KILL, HUP or QUIT")

	return []*cobra.Command{sh, cp, syncCmd, jobs, kill, logsCommand()}
}

// runExec runs a command with the local terminal attached: stdin when piped (or with a tty), signals, resizes.
func runExec(cl *client.Client, start proto.ExecStart) error {
	stdinTTY := term.IsTerminal(int(os.Stdin.Fd()))
	var in io.Reader
	if start.TTY || !stdinTTY {
		in = os.Stdin
	}
	sigs, resize := make(chan string, 4), make(chan [2]uint16, 4)
	local := make(chan os.Signal, 4)
	signal.Notify(local, syscall.SIGINT, syscall.SIGTERM, syscall.SIGWINCH)
	defer signal.Stop(local)
	if start.TTY && stdinTTY {
		if w, h, err := term.GetSize(int(os.Stdin.Fd())); err == nil {
			start.Cols, start.Rows = uint16(w), uint16(h)
		}
		old, err := term.MakeRaw(int(os.Stdin.Fd()))
		if err == nil {
			defer term.Restore(int(os.Stdin.Fd()), old)
		}
	}
	go func() {
		for s := range local {
			switch s {
			case syscall.SIGINT:
				sigs <- "INT"
			case syscall.SIGTERM:
				sigs <- "TERM"
			case syscall.SIGWINCH:
				if w, h, err := term.GetSize(int(os.Stdin.Fd())); err == nil {
					resize <- [2]uint16{uint16(h), uint16(w)}
				}
			}
		}
	}()
	code, err := cl.Exec(start, client.ExecIO{Stdin: in, Stdout: os.Stdout, Stderr: os.Stderr, Signals: sigs, Resize: resize})
	if err != nil {
		return err
	}
	if code != 0 {
		return exitError{code}
	}
	return nil
}

var (
	ansi      = regexp.MustCompile(`\x1b\[[0-9;?]*[A-Za-z]`)
	epochLine = regexp.MustCompile(`(^Epoch \d+/\d+)|val_loss|scored targets`)
)

// epochWriter keeps the end-of-epoch lines of a training log (progress bars rewrite one line with \r).
type epochWriter struct {
	w   io.Writer
	buf strings.Builder
}

func (e *epochWriter) Write(p []byte) (int, error) {
	e.buf.Write(p)
	s := e.buf.String()
	i := strings.LastIndexAny(s, "\n\r")
	if i < 0 {
		return len(p), nil
	}
	done, rest := s[:i+1], s[i+1:]
	e.buf.Reset()
	e.buf.WriteString(rest)
	for _, line := range strings.FieldsFunc(done, func(r rune) bool { return r == '\n' || r == '\r' }) {
		line = ansi.ReplaceAllString(line, "")
		if epochLine.MatchString(line) {
			fmt.Fprintln(e.w, strings.TrimSpace(line))
		}
	}
	return len(p), nil
}

func logsCommand() *cobra.Command {
	var follow, epochs bool
	var since int64
	cmd := &cobra.Command{
		Use:   "logs JOB_OR_RUN",
		Short: "A job's or run's output: live from the pod, or from R2 when the pod is gone",
		Long: `Print a job's output (stdout and stderr together, from its start). With -f, follow it until it ends: if the
connection drops, it reconnects and resumes at the last byte received. When the pod is unreachable or gone, the copy
in R2 is read (runs/<run>/train.log, or jobs/<tag>/<id>.log). While following, it also signs the pod's store
requests (see remote-pod signer). --epochs keeps the end-of-epoch lines of a training log.`,
		Example: `  remote-pod logs tok -f
  remote-pod logs my-run -f --epochs          one line per epoch: loss, val_loss, lr, tokens/s
  remote-pod logs my-run                      the whole log (from R2 after the pod stopped)`,
		GroupID: "work",
		Args:    cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			var w io.Writer = os.Stdout
			if epochs {
				ew := &epochWriter{w: os.Stdout}
				defer ew.Write([]byte("\n"))
				w = ew
			}
			bw := bufio.NewWriterSize(w, 1)
			defer bw.Flush()
			cl, pst, err := agentClient()
			if err == nil {
				if _, err = cl.Job(args[0]); err == nil {
					if follow { // sign the pod's store requests while following (it holds no key)
						if cfg, cerr := config.Load(); cerr == nil {
							ctx, cancel := context.WithCancel(context.Background())
							defer cancel()
							if serr := serveSigner(ctx, cfg, pst, cl, false); serr != nil {
								ui.Note("not signing the pod's uploads: %v", serr)
							}
						}
					}
					code, err := cl.FollowLog(args[0], since, follow, w, func(e error) {
						ui.Note("connection lost (%v); reconnecting", e)
					})
					if err != nil {
						return err
					}
					if code != 0 && follow {
						return exitError{code}
					}
					return nil
				}
			}
			// the pod is gone, or the name is a run that is not a job here: R2
			b, rerr := readStoreLog(args[0])
			if rerr != nil {
				if err != nil {
					return hint(fmt.Errorf("%v; and in R2: %v", err, rerr), "remote-pod jobs (what runs on the pod)")
				}
				return rerr
			}
			_, _ = w.Write(b[min(int(since), len(b)):])
			return nil
		},
	}
	cmd.Flags().BoolVarP(&follow, "follow", "f", false, "keep following until the job ends")
	cmd.Flags().BoolVar(&epochs, "epochs", false, "only the end-of-epoch lines of a training log")
	cmd.Flags().Int64Var(&since, "offset", 0, "start at this byte of the log")
	return cmd
}
