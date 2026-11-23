package main

import (
	"bufio"
	"context"
	"crypto/rand"
	"encoding/hex"
	"errors"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/spf13/cobra"
	"golang.org/x/term"

	"github.com/SamuelHaidu/rnn-research/tools/remote-pod/internal/client"
	"github.com/SamuelHaidu/rnn-research/tools/remote-pod/internal/config"
	"github.com/SamuelHaidu/rnn-research/tools/remote-pod/internal/runpod"
	"github.com/SamuelHaidu/rnn-research/tools/remote-pod/internal/store"
	"github.com/SamuelHaidu/rnn-research/tools/remote-pod/internal/ui"
)

func api() (*runpod.Client, *config.Config, error) {
	cfg, err := config.Load()
	if err != nil {
		return nil, nil, err
	}
	key, err := cfg.RunPodKey()
	if err != nil {
		return nil, nil, hint(err, "remote-pod config (shows where the settings go)")
	}
	return runpod.New(key), cfg, nil
}

func agentClient() (*client.Client, *config.State, error) {
	st, err := config.LoadState()
	if err != nil {
		return nil, nil, err
	}
	url, token, err := st.Endpoint()
	if err != nil {
		return nil, nil, hint(err, `remote-pod up --gpu "RTX 4090"`)
	}
	return client.New(url, token), st, nil
}

func newToken() string {
	b := make([]byte, 24)
	_, _ = rand.Read(b)
	return hex.EncodeToString(b)
}

// resolveGPU matches a GPU id or display name ("RTX 4090", "4090", "NVIDIA GeForce RTX 4090").
func resolveGPU(ctx context.Context, c *runpod.Client, want string) (string, error) {
	gpus, err := c.GPUs(ctx)
	if err != nil {
		return "", err
	}
	w := strings.ToLower(want)
	for _, g := range gpus {
		if strings.ToLower(g.ID) == w || strings.ToLower(g.DisplayName) == w {
			return g.ID, nil
		}
	}
	var hits []string
	for _, g := range gpus {
		if strings.Contains(strings.ToLower(g.ID), w) || strings.Contains(strings.ToLower(g.DisplayName), w) {
			hits = append(hits, g.ID)
		}
	}
	if len(hits) == 1 {
		return hits[0], nil
	}
	if len(hits) > 1 {
		return "", fmt.Errorf("%q matches several GPUs: %s", want, strings.Join(hits, ", "))
	}
	return "", hint(fmt.Errorf("no GPU matches %q", want), "remote-pod gpus")
}

// createPod tries community then secure (cloud "any"), with our image and the agent's environment.
func createPod(ctx context.Context, c *runpod.Client, cfg *config.Config, gpu, cloud, image string, disk, volume int,
	env map[string]string, name, dockerArgs string) (string, error) {
	clouds := map[string][]string{"any": {"COMMUNITY", "SECURE"}, "community": {"COMMUNITY"}, "secure": {"SECURE"}}[cloud]
	if clouds == nil {
		return "", fmt.Errorf("--cloud is community, secure or any (got %q)", cloud)
	}
	if cfg.HasR2() {
		for k, v := range (store.Config{Endpoint: cfg.R2.Endpoint, Bucket: cfg.R2.Bucket, AccessKey: cfg.R2.AccessKey,
			SecretKey: cfg.R2.SecretKey, Region: cfg.R2.Region}).EnvVars() {
			env[k] = v
		}
	}
	var last error
	for _, cl := range clouds {
		ui.Note("asking for a %s %s pod", strings.ToLower(cl), gpu)
		id, err := c.Create(ctx, runpod.CreateInput{Name: name, Image: image, GPUTypeID: gpu, Cloud: cl, DiskGB: disk,
			VolumeGB: volume, Env: env, Ports: fmt.Sprintf("%d/http", config.AgentPort), DockerArgs: dockerArgs})
		if err == nil {
			return id, nil
		}
		last = err
	}
	return "", hint(last, "remote-pod gpus (what is in stock)")
}

// waitAgent polls the agent's health until it answers.
func waitAgent(cl *client.Client, timeout time.Duration) error {
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if _, err := cl.Health(); err == nil {
			return nil
		}
		time.Sleep(3 * time.Second)
	}
	return fmt.Errorf("the agent did not answer within %s (image still downloading? check: remote-pod status)", timeout)
}

// resolveCPU picks a CPU flavor by kind (compute, general, memory: the cheapest in stock) or by id (cpu5c).
func resolveCPU(ctx context.Context, c *runpod.Client, want string, vcpus int) (runpod.CPUFlavor, error) {
	fl, err := c.CPUFlavors(ctx, vcpus)
	if err != nil {
		return runpod.CPUFlavor{}, err
	}
	suffix := map[string]string{"compute": "c", "general": "g", "memory": "m"}[strings.ToLower(want)]
	var best *runpod.CPUFlavor
	for i, f := range fl {
		ok := f.ID == want || (suffix != "" && strings.HasSuffix(f.ID, suffix))
		if !ok || vcpus < f.MinVCPU || vcpus > f.MaxVCPU || f.Specifics == nil || f.Specifics.StockStatus == "" {
			continue
		}
		if best == nil || f.Specifics.SecurePrice < best.Specifics.SecurePrice { // the cheapest of that kind in stock
			best = &fl[i]
		}
	}
	if best == nil {
		return runpod.CPUFlavor{}, hint(fmt.Errorf("no %s CPU pod with %d vCPUs in stock", want, vcpus), "remote-pod cpus --vcpu "+fmt.Sprint(vcpus))
	}
	return *best, nil
}

func createCPUPod(ctx context.Context, c *runpod.Client, cfg *config.Config, instance, image string, disk int,
	env map[string]string, name, dockerArgs string) (string, error) {
	if cfg.HasR2() {
		for k, v := range (store.Config{Endpoint: cfg.R2.Endpoint, Bucket: cfg.R2.Bucket, AccessKey: cfg.R2.AccessKey,
			SecretKey: cfg.R2.SecretKey, Region: cfg.R2.Region}).EnvVars() {
			env[k] = v
		}
	}
	ui.Note("asking for a CPU pod %s", instance)
	return c.CreateCPU(ctx, runpod.CreateInput{Name: name, Image: image, DiskGB: disk, Env: env,
		Ports: fmt.Sprintf("%d/http", config.AgentPort), DockerArgs: dockerArgs}, instance)
}

func imageRef(cfg *config.Config, image string) string {
	switch image {
	case "cpu":
		return cfg.Images.CPU
	case "", "train":
		return cfg.Images.Train
	case "dev":
		return cfg.Images.Dev
	}
	return image
}

func podCommands() []*cobra.Command {
	var gpu, cloud, image, name, flavor string
	var disk, volume, idle, cpus int
	up := &cobra.Command{
		Use:   "up (--gpu GPU | --cpu N)",
		Short: "Create a GPU or CPU-only pod with our image and wait for its agent",
		Long: `Create a pod running our image (its entrypoint is the agent that sh, cp, sync and logs talk to) and wait
until the agent answers. The pod is reached through RunPod's HTTPS proxy, so community pods without a public IP work.
It stops itself after --idle minutes with no command and nothing running.

--cpu N makes a CPU-only pod (N vCPUs, $0.03 to $0.07 per vCPU-hour; see remote-pod cpus) with the cpu image (Python, uv, zstd, git): for
downloading and cleaning data, training tokenizers, compressing datasets; then remote-pod dataset add from there.`,
		Example: `  remote-pod up --gpu "RTX 4090"                    community first, then secure
  remote-pod up --gpu 5090 --cloud secure --disk 80
  remote-pod up --gpu "RTX 4090" --image dev         nvcc, cmake, ninja, nsys / ncu: the optimization loop
  remote-pod up --cpu 16 --disk 200                  data work: 16 vCPUs, 64 GB RAM (general purpose)
  remote-pod up --cpu 8 --flavor memory              8 vCPUs, 64 GB RAM (see remote-pod cpus)
  remote-pod up --cpu 32 --image buildpack-deps:noble    any public image: the agent is fetched from R2 at boot
  remote-pod up --gpu "RTX 4090" --image nvidia/cuda:12.8.1-devel-ubuntu24.04`,
		GroupID: "pod",
		RunE: func(cmd *cobra.Command, args []string) error {
			c, cfg, err := api()
			if err != nil {
				return err
			}
			st, _ := config.LoadState()
			if st.PodID != "" {
				return hint(fmt.Errorf("this project already has pod %s", st.PodID), "remote-pod status (or remote-pod down to replace it)")
			}
			if (gpu == "") == (cpus == 0) {
				return hint(errors.New("give either --gpu or --cpu"), `remote-pod up --gpu "RTX 4090"  or  remote-pod up --cpu 8`)
			}
			ctx := context.Background()
			token := newToken()
			env := map[string]string{"REMOTE_POD_TOKEN": token, "REMOTE_POD_IDLE_MIN": fmt.Sprint(idle)}
			var id, gpuID, ref, dockerArgs string
			if cpus > 0 && !cmd.Flags().Changed("image") {
				image = "cpu"
			}
			ref = imageRef(cfg, image)
			if !isOurImage(cfg, ref) { // a public image: the pod downloads the agent at boot
				args, extra, err := bootstrap(ctx, cfg)
				if err != nil {
					return err
				}
				dockerArgs = args
				for k, v := range extra {
					env[k] = v
				}
				ui.Note("public image %s: the pod downloads the agent from R2 at boot", ref)
			}
			if cpus > 0 {
				f, err := resolveCPU(ctx, c, flavor, cpus)
				if err != nil {
					return err
				}
				gpuID = runpod.CPUInstanceID(f, cpus)
				if id, err = createCPUPod(ctx, c, cfg, gpuID, ref, disk, env, name, dockerArgs); err != nil {
					return err
				}
			} else {
				if gpuID, err = resolveGPU(ctx, c, gpu); err != nil {
					return err
				}
				if id, err = createPod(ctx, c, cfg, gpuID, cloud, ref, disk, volume, env, name, dockerArgs); err != nil {
					return err
				}
			}
			if err := config.SaveState(&config.State{PodID: id, Token: token, Image: ref, GPU: gpuID}); err != nil {
				return err
			}
			ui.Success("pod %s created (%s, %s); waiting for its agent", id, gpuID, ref)
			agentCl, _, _ := agentClient()
			if err := waitAgent(agentCl, 15*time.Minute); err != nil {
				return err
			}
			ui.Success("ready: remote-pod sh %s", map[bool]string{true: "nproc", false: "nvidia-smi"}[cpus > 0])
			return nil
		},
	}
	up.Flags().StringVar(&gpu, "gpu", "", `GPU type: an id or display name ("RTX 4090", "5090"); see remote-pod gpus`)
	up.Flags().StringVar(&cloud, "cloud", "any", "community, secure, or any (community first)")
	up.Flags().IntVar(&cpus, "cpu", 0, "a CPU-only pod with this many vCPUs (2 to 32; see remote-pod cpus)")
	up.Flags().StringVar(&flavor, "flavor", "general", "CPU pods: compute (2 GB RAM / vCPU), general (4 GB), memory (8 GB), or an id (cpu3c)")
	up.Flags().StringVar(&image, "image", "train", "train (lbc + agent), dev (+ nvcc, cmake, ninja, Nsight), cpu (data work), or an image reference")
	up.Flags().IntVar(&disk, "disk", 40, "container disk in GB (datasets and runs live here; checkpoints also go to R2)")
	up.Flags().IntVar(&volume, "volume", 0, "persistent /workspace volume in GB (0: none; data lives in R2)")
	up.Flags().IntVar(&idle, "idle", 60, "stop the pod after this many minutes with nothing running (0: never)")
	up.Flags().StringVar(&name, "name", "remote-pod", "the pod's name in the RunPod console")

	status := &cobra.Command{
		Use:     "status",
		Short:   "The pod, its agent, its jobs and the account balance",
		Example: "  remote-pod status\n  remote-pod status --json",
		GroupID: "pod",
		RunE: func(cmd *cobra.Command, args []string) error {
			st, _ := config.LoadState()
			out := map[string]any{"state": st}
			c, _, apiErr := api()
			if apiErr == nil {
				ctx := context.Background()
				if bal, perHr, err := c.Balance(ctx); err == nil {
					out["balance"], out["spend_per_hour"] = bal, perHr
				}
				if st.PodID != "" {
					if pod, err := c.Get(ctx, st.PodID); err == nil {
						out["pod"] = pod
					} else {
						out["pod_error"] = err.Error()
					}
				}
			}
			var jobs any
			if cl, _, err := agentClient(); err == nil {
				if h, err := cl.Health(); err == nil {
					out["agent"] = h
					if js, err := cl.Jobs(); err == nil {
						jobs = js
						out["jobs"] = js
					}
				} else {
					out["agent_error"] = err.Error()
				}
			}
			ui.Print(out, func() {
				if b, ok := out["balance"].(float64); ok {
					fmt.Printf("%s $%.2f  (spending $%.3f/hr)\n", ui.Bold("Balance"), b, out["spend_per_hour"])
				}
				if st.PodID == "" && st.URL == "" {
					fmt.Println(ui.Dim("No pod for this project.") + "  remote-pod up --gpu \"RTX 4090\"")
					return
				}
				if p, ok := out["pod"].(*runpod.Pod); ok {
					up := "-"
					if p.Runtime != nil {
						up = (time.Duration(p.Runtime.UptimeInSeconds) * time.Second).String()
					}
					ui.Table([]string{"Pod", "State", "GPU", "$/hr", "Uptime", "Image"},
						[][]string{{p.ID, ui.State(p.DesiredStatus), p.Machine.GPUDisplayName, fmt.Sprintf("%.2f", p.CostPerHr), up, p.ImageName}})
				} else if e, ok := out["pod_error"]; ok {
					fmt.Println(ui.Bad("Pod: ") + fmt.Sprint(e))
				}
				if e, ok := out["agent_error"]; ok {
					fmt.Println(ui.Warn("Agent: ") + fmt.Sprint(e))
				} else if h, ok := out["agent"]; ok {
					fmt.Printf("%s %v\n", ui.OK("Agent: answering"), ui.Dim(fmt.Sprintf("%+v", h)))
				}
				printJobs(jobs)
			})
			return nil
		},
	}

	gpus := &cobra.Command{
		Use:     "gpus",
		Short:   "GPU types in stock now, cheapest first",
		Example: "  remote-pod gpus\n  remote-pod gpus --all",
		GroupID: "pod",
		RunE: func(cmd *cobra.Command, args []string) error {
			c, _, err := api()
			if err != nil {
				return err
			}
			all, _ := cmd.Flags().GetBool("all")
			list, err := c.GPUs(context.Background())
			if err != nil {
				return err
			}
			var rows [][]string
			var shown []runpod.GPU
			for _, g := range list {
				if !all && g.LowestPrice.StockStatus == "" {
					continue
				}
				shown = append(shown, g)
				rows = append(rows, []string{g.ID, fmt.Sprintf("%d GB", g.MemoryInGb),
					fmt.Sprintf("$%.2f", g.LowestPrice.UninterruptablePrice), fmt.Sprintf("$%.2f", g.CommunityPrice),
					fmt.Sprintf("$%.2f", g.SecurePrice), ui.State(strings.ToUpper(g.LowestPrice.StockStatus))})
			}
			ui.Print(shown, func() { ui.Table([]string{"GPU", "VRAM", "From/hr", "Community", "Secure", "Stock"}, rows) })
			return nil
		},
	}
	gpus.Flags().Bool("all", false, "also the types out of stock")

	stop := &cobra.Command{
		Use:     "stop",
		Short:   "Stop the pod (billing stops; the container disk is wiped)",
		Example: "  remote-pod stop",
		GroupID: "pod",
		RunE: func(cmd *cobra.Command, args []string) error {
			c, _, err := api()
			if err != nil {
				return err
			}
			st, _ := config.LoadState()
			if st.PodID == "" {
				return errors.New("no pod for this project")
			}
			if err := c.Stop(context.Background(), st.PodID); err != nil {
				return err
			}
			ui.Success("pod %s stopped", st.PodID)
			return nil
		},
	}

	var yes bool
	down := &cobra.Command{
		Use:     "down",
		Short:   "Delete the pod and its volume (checkpoints in R2 stay)",
		Example: "  remote-pod down\n  remote-pod down --yes        no question (scripts)",
		GroupID: "pod",
		RunE: func(cmd *cobra.Command, args []string) error {
			c, _, err := api()
			if err != nil {
				return err
			}
			st, _ := config.LoadState()
			if st.PodID == "" {
				return errors.New("no pod for this project")
			}
			if !yes {
				if !term.IsTerminal(int(os.Stdin.Fd())) {
					return hint(errors.New("down deletes the pod and its volume: confirm with --yes"), "remote-pod down --yes")
				}
				fmt.Fprintf(os.Stderr, "Delete pod %s and its volume? [y/N] ", st.PodID)
				ans, _ := bufio.NewReader(os.Stdin).ReadString('\n')
				if strings.ToLower(strings.TrimSpace(ans)) != "y" {
					return errors.New("not deleted")
				}
			}
			if err := c.Terminate(context.Background(), st.PodID); err != nil {
				if !strings.Contains(err.Error(), "not found") {
					return err
				}
				ui.Note("pod %s no longer exists (already deleted)", st.PodID) // only the local state is left
			}
			_ = config.SaveState(&config.State{})
			ui.Success("pod %s deleted", st.PodID)
			return nil
		},
	}
	down.Flags().BoolVarP(&yes, "yes", "y", false, "do not ask")

	var vcpu int
	cpusCmd := &cobra.Command{
		Use:     "cpus",
		Short:   "CPU-only pod families, with the stock and price of a size",
		Example: "  remote-pod cpus\n  remote-pod cpus --vcpu 16",
		GroupID: "pod",
		RunE: func(cmd *cobra.Command, args []string) error {
			c, _, err := api()
			if err != nil {
				return err
			}
			fl, err := c.CPUFlavors(context.Background(), vcpu)
			if err != nil {
				return err
			}
			var rows [][]string
			for _, f := range fl {
				stock, price := "-", "-"
				if f.Specifics != nil {
					stock, price = strings.ToUpper(f.Specifics.StockStatus), fmt.Sprintf("$%.3f", f.Specifics.SecurePrice)
				}
				rows = append(rows, []string{f.ID, f.DisplayName, fmt.Sprintf("%d GB", f.RAMPerVCPU),
					fmt.Sprintf("%d-%d", f.MinVCPU, f.MaxVCPU), runpod.CPUInstanceID(f, vcpu), price, ui.State(stock)})
			}
			ui.Print(fl, func() {
				ui.Table([]string{"Flavor", "Kind", "RAM/vCPU", "vCPUs", fmt.Sprintf("Size at %d vCPUs", vcpu), "$/hr", "Stock"}, rows)
			})
			return nil
		},
	}
	cpusCmd.Flags().IntVar(&vcpu, "vcpu", 8, "the size to price (vCPUs)")
	return []*cobra.Command{gpus, cpusCmd, up, status, stop, down, configCommand()}
}

func configCommand() *cobra.Command {
	return &cobra.Command{
		Use:     "config",
		Short:   "Where the settings live, and what is set (secrets hidden)",
		GroupID: "pod",
		Example: "  remote-pod config",
		RunE: func(cmd *cobra.Command, args []string) error {
			cfg, err := config.Load()
			if err != nil {
				return err
			}
			_, keyErr := cfg.RunPodKey()
			set := func(ok bool) string {
				if ok {
					return ui.OK("set")
				}
				return ui.Warn("missing")
			}
			out := map[string]any{"path": config.Path(), "runpod_key": keyErr == nil, "r2": cfg.HasR2(),
				"r2_endpoint": cfg.R2.Endpoint, "r2_bucket": cfg.R2.Bucket, "images": cfg.Images}
			ui.Print(out, func() {
				fmt.Println(ui.Bold("Settings file: ") + config.Path())
				ui.Table([]string{"Setting", "Value"}, [][]string{
					{"RunPod API key", set(keyErr == nil)},
					{"R2 endpoint", cfg.R2.Endpoint},
					{"R2 bucket", cfg.R2.Bucket},
					{"R2 keys", set(cfg.HasR2())},
					{"Train image", cfg.Images.Train},
					{"Dev image", cfg.Images.Dev},
				})
				fmt.Println(ui.Dim(`Example config.toml:
  [runpod]
  api_key = "..."                       # or leave out: runpodctl's key is used
  [r2]
  endpoint = "https://<account>.r2.cloudflarestorage.com"
  bucket = "rnn-research"
  access_key = "..."
  secret_key = "..."`))
			})
			return nil
		},
	}
}
