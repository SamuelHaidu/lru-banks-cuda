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
	env map[string]string, name string) (string, error) {
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
			VolumeGB: volume, Env: env, Ports: fmt.Sprintf("%d/http", config.AgentPort)})
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

func imageRef(cfg *config.Config, image string) string {
	switch image {
	case "", "train":
		return cfg.Images.Train
	case "dev":
		return cfg.Images.Dev
	}
	return image
}

func podCommands() []*cobra.Command {
	var gpu, cloud, image, name string
	var disk, volume, idle int
	up := &cobra.Command{
		Use:   "up --gpu GPU",
		Short: "Create a pod with our image and wait for its agent",
		Long: `Create a pod running our image (its entrypoint is the agent that sh, cp, sync and logs talk to) and wait
until the agent answers. The pod is reached through RunPod's HTTPS proxy, so community pods without a public IP work.
It stops itself after --idle minutes with no command and nothing running.`,
		Example: `  remote-pod up --gpu "RTX 4090"                    community first, then secure
  remote-pod up --gpu 5090 --cloud secure --disk 80
  remote-pod up --gpu "RTX 4090" --image dev         nvcc, cmake, ninja, nsys / ncu: the optimization loop`,
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
			ctx := context.Background()
			gpuID, err := resolveGPU(ctx, c, gpu)
			if err != nil {
				return err
			}
			token, ref := newToken(), imageRef(cfg, image)
			env := map[string]string{"REMOTE_POD_TOKEN": token, "REMOTE_POD_IDLE_MIN": fmt.Sprint(idle)}
			id, err := createPod(ctx, c, cfg, gpuID, cloud, ref, disk, volume, env, name)
			if err != nil {
				return err
			}
			if err := config.SaveState(&config.State{PodID: id, Token: token, Image: ref, GPU: gpuID}); err != nil {
				return err
			}
			ui.Success("pod %s created (%s, %s); waiting for its agent", id, gpuID, ref)
			agentCl, _, _ := agentClient()
			if err := waitAgent(agentCl, 15*time.Minute); err != nil {
				return err
			}
			ui.Success("ready: remote-pod sh nvidia-smi")
			return nil
		},
	}
	up.Flags().StringVar(&gpu, "gpu", "", `GPU type: an id or display name ("RTX 4090", "5090"); see remote-pod gpus`)
	up.Flags().StringVar(&cloud, "cloud", "any", "community, secure, or any (community first)")
	up.Flags().StringVar(&image, "image", "train", "train (lbc + agent), dev (+ nvcc, cmake, ninja, Nsight), or an image reference")
	up.Flags().IntVar(&disk, "disk", 40, "container disk in GB (datasets and runs live here; checkpoints also go to R2)")
	up.Flags().IntVar(&volume, "volume", 0, "persistent /workspace volume in GB (0: none; data lives in R2)")
	up.Flags().IntVar(&idle, "idle", 60, "stop the pod after this many minutes with nothing running (0: never)")
	up.Flags().StringVar(&name, "name", "remote-pod", "the pod's name in the RunPod console")
	_ = up.MarkFlagRequired("gpu")

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
				return err
			}
			_ = config.SaveState(&config.State{})
			ui.Success("pod %s deleted", st.PodID)
			return nil
		},
	}
	down.Flags().BoolVarP(&yes, "yes", "y", false, "do not ask")

	return []*cobra.Command{gpus, up, status, stop, down, configCommand()}
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
