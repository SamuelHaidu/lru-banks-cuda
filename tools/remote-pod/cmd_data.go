package main

import (
	"context"
	"encoding/base64"
	"encoding/csv"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"time"

	"github.com/spf13/cobra"

	"github.com/SamuelHaidu/rnn-research/tools/remote-pod/internal/agent"
	"github.com/SamuelHaidu/rnn-research/tools/remote-pod/internal/client"
	"github.com/SamuelHaidu/rnn-research/tools/remote-pod/internal/config"
	"github.com/SamuelHaidu/rnn-research/tools/remote-pod/internal/grant"
	"github.com/SamuelHaidu/rnn-research/tools/remote-pod/internal/proto"
	"github.com/SamuelHaidu/rnn-research/tools/remote-pod/internal/store"
	"github.com/SamuelHaidu/rnn-research/tools/remote-pod/internal/syncx"
	"github.com/SamuelHaidu/rnn-research/tools/remote-pod/internal/ui"
)

func openStore() (*store.Store, *config.Config, error) {
	cfg, err := config.Load()
	if err != nil {
		return nil, nil, err
	}
	if !cfg.HasR2() {
		// inside a pod: the settings come in the environment (remote-pod up / train pass them)
		if sc, ok := store.FromEnv(); ok {
			st, err := store.New(sc)
			return st, cfg, err
		}
		return nil, nil, hint(errors.New("R2 is not configured"), "remote-pod config (shows the settings to add)")
	}
	st, err := store.New(store.Config{Endpoint: cfg.R2.Endpoint, Bucket: cfg.R2.Bucket, AccessKey: cfg.R2.AccessKey,
		SecretKey: cfg.R2.SecretKey, Region: cfg.R2.Region})
	return st, cfg, err
}

func readStoreLog(name string) ([]byte, error) {
	st, _, err := openStore()
	if err != nil {
		return nil, err
	}
	ctx := context.Background()
	if b, err := st.GetBytes(ctx, "runs/"+name+"/train.log"); err == nil {
		return b, nil
	}
	if ps, _ := config.LoadState(); ps.Tag != "" { // the pod's own job logs: jobs/<tag>/<id>.log
		if b, err := st.GetBytes(ctx, grant.JobLogPrefix(ps.Tag)+name+".log"); err == nil {
			return b, nil
		}
	}
	return st.GetBytes(ctx, "jobs/"+name+".log") // pods made before grants
}

func dataCommands() []*cobra.Command {
	var name, from string
	add := &cobra.Command{
		Use:   "add --name NAME --from DIR",
		Short: "Upload a dataset directory to R2 once (from here, or from the pod: --from :/path)",
		Long: `Upload a dataset directory to R2 as one tar.zst object (streamed, nothing staged on disk), with a manifest.
Pods then download it at datacenter speed. A pod path (":" first) uploads from the pod straight to R2, so data built
on a CPU pod never passes through this machine.`,
		Example: `  remote-pod dataset add --name pretrain-code-mix-bpe32768 --from datasets/pretrain-code-mix-bpe32768
  remote-pod dataset add --name cx --from datasets/calc-x-bpe32768
  remote-pod dataset add --name toucan-2b --from :/work/data/toucan-2b      built on a CPU pod, uploaded from it`,
		Args: cobra.NoArgs,
		RunE: func(cmd *cobra.Command, args []string) error {
			if podPath, onPod := isPod(from); onPod { // the pod's own copy of this tool uploads it
				cl, _, err := agentClient()
				if err != nil {
					return err
				}
				return runExec(cl, proto.ExecStart{Command: fmt.Sprintf("remote-pod dataset add --name %q --from %q", name, podPath)})
			}
			st, _, err := openStore()
			if err != nil {
				return err
			}
			fi, err := os.Stat(from)
			if err != nil || !fi.IsDir() {
				return fmt.Errorf("--from must be a local directory (got %q)", from)
			}
			var raw int64
			nfiles := 0
			_ = filepath.Walk(from, func(_ string, i os.FileInfo, err error) error {
				if err == nil && i.Mode().IsRegular() {
					raw += i.Size()
					nfiles++
				}
				return nil
			})
			ctx, t0, p := context.Background(), time.Now(), &ui.Progress{Label: "uploading " + name}
			ui.Note("%s: %d files, %s (zstd-compressed on the way)", name, nfiles, ui.Bytes(raw))
			size, err := st.PutDir(ctx, store.DatasetKey(name), from, p.Update)
			p.Done()
			if err != nil {
				return err
			}
			d := store.Dataset{Name: name, Source: from, Created: time.Now().UTC(), Files: nfiles, Bytes: raw,
				Compressed: size, Key: store.DatasetKey(name)}
			if err := st.PutDatasetManifest(ctx, d); err != nil {
				return err
			}
			ui.Print(d, func() {
				ui.Success("dataset %s in R2: %s -> %s (%s)", name, ui.Bytes(raw), ui.Bytes(size), time.Since(t0).Round(time.Second))
			})
			return nil
		},
	}
	add.Flags().StringVar(&name, "name", "", "the dataset's name (what configs and pods refer to)")
	add.Flags().StringVar(&from, "from", "", "the local directory to upload")
	_ = add.MarkFlagRequired("name")
	_ = add.MarkFlagRequired("from")

	ls := &cobra.Command{
		Use:     "ls",
		Short:   "Datasets in R2",
		Example: "  remote-pod dataset ls",
		RunE: func(cmd *cobra.Command, args []string) error {
			st, _, err := openStore()
			if err != nil {
				return err
			}
			ds, err := st.Datasets(context.Background())
			if err != nil {
				return err
			}
			var rows [][]string
			for _, d := range ds {
				rows = append(rows, []string{d.Name, fmt.Sprint(d.Files), ui.Bytes(d.Bytes), ui.Bytes(d.Compressed),
					d.Created.Format("2006-01-02 15:04"), d.Source})
			}
			ui.Print(ds, func() { ui.Table([]string{"Dataset", "Files", "Size", "In R2", "Added", "From"}, rows) })
			return nil
		},
	}
	dataset := &cobra.Command{Use: "dataset", Short: "Datasets in R2: add, ls", GroupID: "data"}
	dataset.AddCommand(add, ls)

	var all bool
	pull := &cobra.Command{
		Use:   "pull RUN",
		Short: "Download a run from R2 to runs/RUN (no pod needed)",
		Long: `Download a run's files from R2 (config, metrics, log, model.lbc and best.lbc; --all: every checkpoint) to
runs/RUN here. Files already here with the same size are skipped.`,
		Example: "  remote-pod pull my-run\n  remote-pod pull my-run --all",
		GroupID: "data",
		Args:    cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			st, _, err := openStore()
			if err != nil {
				return err
			}
			ctx, run := context.Background(), args[0]
			objs, err := st.List(ctx, "runs/"+run+"/", true)
			if err != nil {
				return err
			}
			if len(objs) == 0 {
				return hint(fmt.Errorf("no run %q in R2", run), "remote-pod status (what runs now)")
			}
			var got []string
			for _, o := range objs {
				rel := strings.TrimPrefix(o.Key, "runs/"+run+"/")
				if !all && strings.Contains(rel, "checkpoint_") {
					continue
				}
				// a pod with the store's keys can name an object anything: only plain paths inside the run land here
				// ("runs/x/../../../.bashrc" would otherwise overwrite a file of this machine)
				if !syncx.SafeRelPath(rel) {
					ui.Note("skipped %q: not a plain path inside the run", o.Key)
					continue
				}
				dst := filepath.Join(config.ProjectRoot(), "runs", run, filepath.FromSlash(rel))
				if fi, err := os.Stat(dst); err == nil && fi.Size() == o.Size {
					continue
				}
				ui.Note("%s (%s)", rel, ui.Bytes(o.Size))
				if err := st.GetFile(ctx, o.Key, dst); err != nil {
					return err
				}
				got = append(got, rel)
			}
			ui.Print(got, func() { ui.Success("runs/%s: %d file(s) downloaded", run, len(got)) })
			return nil
		},
	}
	pull.Flags().BoolVar(&all, "all", false, "every numbered checkpoint too")

	var cols string
	metrics := &cobra.Command{
		Use:   "metrics RUN",
		Short: "A run's metrics.csv as a table (from the pod, else R2)",
		Example: `  remote-pod metrics my-run
  remote-pod metrics my-run --columns epoch,loss,val_loss,lr
  remote-pod metrics my-run --columns all`,
		GroupID: "data",
		Args:    cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			b, err := readMetrics(args[0])
			if err != nil {
				return err
			}
			recs, err := csv.NewReader(strings.NewReader(string(b))).ReadAll()
			if err != nil || len(recs) == 0 {
				return fmt.Errorf("metrics.csv: %v", err)
			}
			head := recs[0]
			want := map[string]bool{}
			if cols != "all" {
				for _, c := range strings.Split(cols, ",") {
					want[strings.TrimSpace(c)] = true
				}
			}
			var idx []int
			for i, h := range head {
				if cols == "all" || want[h] {
					idx = append(idx, i)
				}
			}
			pick := func(r []string) []string {
				out := make([]string, len(idx))
				for k, i := range idx {
					if i < len(r) {
						out[k] = r[i]
					}
				}
				return out
			}
			var rows [][]string
			var asMaps []map[string]string
			for _, r := range recs[1:] {
				rows = append(rows, pick(r))
				m := map[string]string{}
				for i, h := range head {
					if i < len(r) {
						m[h] = r[i]
					}
				}
				asMaps = append(asMaps, m)
			}
			ui.Print(asMaps, func() { ui.Table(pick(head), rows) })
			return nil
		},
	}
	metrics.Flags().StringVar(&cols, "columns", "epoch,loss,val_loss,val_mix_loss,bpc,lr,grad_norm,tokens_per_s",
		`columns to show (comma-separated), or "all"`)

	return []*cobra.Command{dataset, pull, metrics, trainCommand()}
}

func readMetrics(run string) ([]byte, error) {
	if cl, _, err := agentClient(); err == nil {
		tmp, err := os.MkdirTemp("", "remote-pod-metrics")
		if err == nil {
			defer os.RemoveAll(tmp)
			if target, err := cl.Download("runs/"+run+"/metrics.csv", tmp+"/", nil); err == nil {
				return os.ReadFile(target)
			}
		}
	}
	st, _, err := openStore()
	if err != nil {
		return nil, err
	}
	return st.GetBytes(context.Background(), "runs/"+run+"/metrics.csv")
}

var (
	dataPath = regexp.MustCompile(`^/(?:root/data|data|workspace/data)/([^/]+)/?$`)
	runPath  = regexp.MustCompile(`^/(?:workspace|work|root)/runs/([^/]+)/(.+)$`)
)

// planTrain rewrites a run config for the pod: dataset directories become /data/<name> (fetched from R2), checkpoint
// paths under runs/ become /work/runs/... (fetched from runs/... in R2).
func planTrain(run string, raw []byte) (agent.TrainSpec, error) {
	var cfg map[string]any
	if err := json.Unmarshal(raw, &cfg); err != nil {
		return agent.TrainSpec{}, fmt.Errorf("config: %w", err)
	}
	spec := agent.TrainSpec{Run: run, Fetch: map[string]string{}, Files: map[string]string{},
		RunDir: "/work/runs/" + run, Prefix: "runs/" + run + "/", DataDir: "/data", StopWhenDone: true,
		Command: "lbc train /work/runs/" + run}
	seen := map[string]bool{}
	var walk func(v any) any
	walk = func(v any) any {
		switch x := v.(type) {
		case map[string]any:
			for k, e := range x {
				x[k] = walk(e)
			}
		case []any:
			for i, e := range x {
				x[i] = walk(e)
			}
		case string:
			if m := dataPath.FindStringSubmatch(x); m != nil {
				if !seen[m[1]] {
					seen[m[1]] = true
					spec.Datasets = append(spec.Datasets, m[1])
				}
				return "/data/" + m[1]
			}
			if m := runPath.FindStringSubmatch(x); m != nil {
				spec.Fetch["runs/"+m[1]+"/"+m[2]] = "/work/runs/" + m[1] + "/" + m[2]
				return "/work/runs/" + m[1] + "/" + m[2]
			}
		}
		return v
	}
	cfg = walk(cfg).(map[string]any)
	b, _ := json.MarshalIndent(cfg, "", "  ")
	spec.Files["/work/runs/"+run+"/config.json"] = string(b)
	return spec, nil
}

func trainCommand() *cobra.Command {
	var gpu, cloud, image, provider string
	var disk int
	var maxPrice float64
	var detach, dry bool
	cmd := &cobra.Command{
		Use:   "train CONFIG.json --gpu GPU",
		Short: "Run a training job end to end: create, fetch, train, upload, stop",
		Long: `Create a pod for one run of lbc train. The config's dataset paths (/root/data/<name>, /data/<name>) name
datasets in R2 (remote-pod dataset add), and paths under runs/ (init_from) name files in R2's runs/. On the pod the
agent fetches them, runs the training, uploads checkpoints, metrics and the log to runs/<run>/ as they appear (keeping
only the newest numbered checkpoint on the pod), and stops the pod when the run ends. This command follows the
progress; Ctrl-C only stops following (the run goes on): resume with remote-pod logs <run> -f --epochs.

On vast.ai (--provider vast) the pod is the cheapest good offer of the GPU (as remote-pod up --provider vast) on a
public CUDA image; lbc comes from R2, so build it first (remote-pod build). Stopping at the end releases the GPU; the
disk is kept and billed for storage until remote-pod down.`,
		Example: `  remote-pod train runs/anneal-t40/config.json --gpu "RTX 5090"
  remote-pod train runs/x/config.json --gpu 4090 --detach     start it and return
  remote-pod train runs/x/config.json --gpu 4090 --dry-run    show what would be fetched and run
  remote-pod train runs/x/config.json --provider vast --gpu 4090 --max-price 0.5`,
		GroupID: "data",
		Args:    cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			raw, err := os.ReadFile(args[0])
			if err != nil {
				return err
			}
			run := filepath.Base(filepath.Dir(filepath.Clean(args[0])))
			spec, err := planTrain(run, raw)
			if err != nil {
				return err
			}
			if dry {
				ui.Print(spec, func() {
					ui.Table([]string{"Run", "Command", "Datasets", "Fetch"}, [][]string{{run, spec.Command,
						strings.Join(spec.Datasets, ", "), fmt.Sprint(len(spec.Fetch))}})
					for k, v := range spec.Fetch {
						fmt.Println(ui.Dim("  fetch ") + k + " -> " + v)
					}
				})
				return nil
			}
			st, cfg, err := openStore()
			if err != nil {
				return err
			}
			ctx := context.Background()
			spec.DatasetKeys = map[string]string{}
			for _, d := range spec.Datasets { // fail before paying for a GPU
				ds, err := st.Dataset(ctx, d)
				if err != nil {
					return err
				}
				if strings.Contains(ds.Key, "://") {
					return fmt.Errorf("dataset %s lives outside the store (%s): a pod can only be granted store keys", d, ds.Key)
				}
				spec.DatasetKeys[d] = ds.Key
			}
			for key := range spec.Fetch {
				if !st.Exists(ctx, key) {
					return hint(fmt.Errorf("%s is not in R2", key), "remote-pod pull / upload the run first")
				}
			}
			if s, _ := config.LoadState(); s.PodID != "" {
				return hint(fmt.Errorf("this project already has pod %s", s.PodID), "remote-pod down (or stop) first")
			}
			prov, err := providerFlag(cfg, provider)
			if err != nil {
				return err
			}
			// the pod gets no key: a policy (its inputs, its run's prefix, its job logs) and grants pre-issued for 7 days
			// for its inputs and its fixed-name outputs; the rest is signed while this command (or a signer) runs
			token, tag := newToken(), newTag()
			var gets []string
			for _, d := range spec.Datasets {
				gets = append(gets, spec.DatasetKeys[d])
			}
			for key := range spec.Fetch {
				gets = append(gets, key)
			}
			sort.Strings(gets)
			policy, err := grant.PodPolicy(tag, []string{spec.Prefix}, gets)
			if err != nil {
				return err
			}
			// what this run will upload, from its config: the pod may upload that much, at that pace, and no more
			epochs, every, epochTokens := runShape(raw)
			plan := grant.RunUploadPlan(spec.Prefix, epochs, every, epochTokens)
			policy.Plans = []grant.UploadPlan{plan}
			ui.Note("upload plan: at most %d files under %s, %d new checkpoints an hour, a checkpoint rewritten at most every %s",
				plan.MaxObjects, spec.Prefix, plan.MaxNewHeavyPerHour, plan.MinRewrite.Round(time.Second))
			var fixed []string
			for _, f := range grant.RunFixedFiles {
				fixed = append(fixed, spec.Prefix+f)
			}
			pre, err := grant.Preissue(ctx, st, policy, fixed)
			if err != nil {
				return err
			}
			env, err := grant.PodEnv(tag, pre)
			if err != nil {
				return err
			}
			specJSON, _ := json.Marshal(spec)
			env["REMOTE_POD_TOKEN"], env["REMOTE_POD_IDLE_MIN"] = token, "20"
			env["REMOTE_POD_JOB"] = base64.StdEncoding.EncodeToString(specJSON)
			var id string
			if prov == "vast" {
				if cmd.Flags().Changed("cloud") {
					return hint(errors.New("--cloud is RunPod's"), "remote-pod train CONFIG --provider vast --gpu 4090")
				}
				if !cmd.Flags().Changed("image") {
					image = "cuda"
				}
				vs, err := createVast(ctx, cfg, vastPlan{GPU: gpu, MaxPrice: maxPrice, DiskGB: disk,
					Image: imageRef(cfg, image), Label: "train-" + run, Env: env})
				if err != nil {
					return err
				}
				vs.Token, vs.Tag, vs.Policy = token, tag, policy
				_ = config.SaveState(vs)
				id = vs.PodID
			} else {
				c, _, err := api()
				if err != nil {
					return err
				}
				gpuID, err := resolveGPU(ctx, c, gpu)
				if err != nil {
					return err
				}
				ref := imageRef(cfg, image)
				if id, err = createPod(ctx, c, cfg, gpuID, cloud, ref, disk, 0, env, "train-"+run, ""); err != nil {
					return err
				}
				_ = config.SaveState(&config.State{Provider: "runpod", PodID: id, Token: token, Image: ref, GPU: gpuID,
					Tag: tag, Policy: policy})
			}
			ui.Success("pod %s for run %s (%s); it stops itself when the run ends", id, run, gpu)
			if detach {
				fmt.Fprintln(os.Stderr, ui.Dim("  follow: ")+"remote-pod logs "+run+" -f --epochs   (signs the pod's uploads while it follows)")
				fmt.Fprintln(os.Stderr, ui.Dim("  or:     ")+"remote-pod signer   (numbered checkpoints wait on the pod until one runs)")
				return nil
			}
			cl, _, _ := agentClient()
			if err := waitAgent(cl, 20*time.Minute); err != nil {
				return err
			}
			return followRun(cl, run)
		},
	}
	cmd.Flags().StringVar(&gpu, "gpu", "", `GPU type ("RTX 5090", "4090"; see remote-pod gpus)`)
	cmd.Flags().StringVar(&provider, "provider", "", "runpod or vast (default: the settings' provider, else runpod)")
	cmd.Flags().StringVar(&cloud, "cloud", "any", "RunPod: community, secure, or any (community first)")
	cmd.Flags().Float64Var(&maxPrice, "max-price", 0, "vast.ai: the most $/hr to pay (0: any)")
	cmd.Flags().StringVar(&image, "image", "train", "train, dev, cuda (vast.ai's default), or an image reference")
	cmd.Flags().IntVar(&disk, "disk", 60, "container disk in GB (datasets + the run)")
	cmd.Flags().BoolVar(&detach, "detach", false, "start the run and return")
	cmd.Flags().BoolVar(&dry, "dry-run", false, "print the plan (datasets, fetches, command) and stop")
	_ = cmd.MarkFlagRequired("gpu")
	return cmd
}

// followRun waits for the run's job (the agent fetches the inputs first), then follows its epoch lines.
func followRun(cl *client.Client, run string) error {
	if cfg, err := config.Load(); err == nil { // sign the pod's uploads while following
		if st, err := config.LoadState(); err == nil {
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			if err := serveSigner(ctx, cfg, st, cl, false); err != nil {
				ui.Note("not signing the pod's uploads: %v", err)
			}
		}
	}
	ui.Note("the agent is fetching the datasets; the run starts after")
	for {
		if _, err := cl.Job(run); err == nil {
			break
		}
		time.Sleep(5 * time.Second)
	}
	code, err := cl.FollowLog(run, 0, true, &epochWriter{w: os.Stdout}, func(e error) { ui.Note("reconnecting (%v)", e) })
	if err != nil {
		return err
	}
	if code != 0 {
		ui.Fail(fmt.Errorf("run %s ended with code %d", run, code), "remote-pod logs "+run)
		return exitError{code}
	}
	ui.Success("run %s done; checkpoints in R2: remote-pod pull %s", run, run)
	return nil
}

// runShape reads what the upload plan needs from an lbc run config: the epochs in all (the stages' sum when there
// are stages), how often a numbered checkpoint is written, and the tokens of an epoch (0: unknown).
func runShape(raw []byte) (epochs, checkpointEvery int, epochTokens float64) {
	var c struct {
		Data struct {
			EpochTokens float64 `json:"epoch_tokens"`
		} `json:"data"`
		Training struct {
			Epochs          int  `json:"epochs"`
			CheckpointEvery *int `json:"checkpoint_every"`
			Stages          []struct {
				Epochs int `json:"epochs"`
			} `json:"stages"`
		} `json:"training"`
	}
	_ = json.Unmarshal(raw, &c)
	epochs = c.Training.Epochs
	if len(c.Training.Stages) > 0 {
		epochs = 0
		for _, st := range c.Training.Stages {
			epochs += st.Epochs
		}
	}
	if epochs <= 0 {
		epochs = 100 // lbc's default
	}
	checkpointEvery = 1 // lbc's default
	if c.Training.CheckpointEvery != nil {
		checkpointEvery = *c.Training.CheckpointEvery
	}
	return epochs, checkpointEvery, c.Data.EpochTokens
}
