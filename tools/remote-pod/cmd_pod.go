package main

import (
	"bufio"
	"context"
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
	"github.com/SamuelHaidu/rnn-research/tools/remote-pod/internal/entropy"
	"github.com/SamuelHaidu/rnn-research/tools/remote-pod/internal/grant"
	"github.com/SamuelHaidu/rnn-research/tools/remote-pod/internal/runpod"
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
	c := runpod.New(key)
	if u := os.Getenv("REMOTE_POD_RUNPOD_API"); u != "" { // tests: a fake provider
		c.URL = u
	}
	return c, cfg, nil
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
	if err := client.CheckBase(url); err != nil {
		return nil, nil, err
	}
	cl := client.New(url, token)
	if st.Cert != "" {
		if err := cl.Pin([]byte(st.Cert)); err != nil {
			return nil, nil, fmt.Errorf(".remote-pod/state.json: cert: %w", err)
		}
	}
	return cl, st, nil
}

func newToken() string {
	return hex.EncodeToString(entropy.Bytes(24)) // hardware + kernel randomness (internal/entropy)
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
	case "cuda":
		return cfg.Images.CUDA
	}
	return image
}

func podCommands() []*cobra.Command {
	var gpu, cloud, image, name, flavor, provider string
	var disk, volume, idle, cpus int
	var maxPrice float64
	var offer int64
	var allowPut, allowGet []string
	up := &cobra.Command{
		Use:   "up (--gpu GPU | --cpu N)",
		Short: "Create a GPU or CPU-only pod with our image and wait for its agent",
		Long: `Create a pod running the agent that sh, cp, sync and logs talk to, and wait until it answers. It stops itself
after --idle minutes with no command and nothing running.

RunPod (default): the pod runs our image (its entrypoint is the agent) and is reached through RunPod's HTTPS proxy,
so community pods without a public IP work. --cpu N makes a CPU-only pod (N vCPUs, $0.03 to $0.07 per vCPU-hour; see
remote-pod cpus) with the cpu image (Python, uv, zstd, git): for downloading and cleaning data, training tokenizers,
compressing datasets; then remote-pod dataset add from there.

vast.ai (--provider vast, or provider = "vast" in the settings): the cheapest verified single-GPU offer with the disk
asked, reliability >= 0.98 and >= 500 Mbps download (--offer picks one yourself; --max-price caps the price). Its hosts
cannot pull our private images, so the default image is a public CUDA one (--image cuda): the agent, and lbc when
engine/build/dist/lbc exists (remote-pod build), come from R2 at boot. The agent's port is published on the host's
public IP; the connection is TLS with a certificate made for this pod and pinned in .remote-pod/state.json. SSH also
works (remote-pod status shows the command).`,
		Example: `  remote-pod up --gpu "RTX 4090"                    RunPod: community first, then secure
  remote-pod up --gpu 5090 --cloud secure --disk 80
  remote-pod up --gpu "RTX 4090" --image dev         nvcc, cmake, ninja, nsys / ncu: the optimization loop
  remote-pod up --cpu 16 --disk 200                  data work: 16 vCPUs, 64 GB RAM (general purpose)
  remote-pod up --cpu 8 --flavor memory              8 vCPUs, 64 GB RAM (see remote-pod cpus)
  remote-pod up --cpu 32 --image buildpack-deps:noble    any public image: the agent is fetched from R2 at boot
  remote-pod up --provider vast --gpu 4090            vast.ai: the cheapest good RTX 4090 offer
  remote-pod up --provider vast --gpu "A100 SXM4" --disk 100 --max-price 0.9
  remote-pod up --provider vast --offer 61538559     a specific offer (from remote-pod gpus --provider vast --gpu ...)`,
		GroupID: "pod",
		RunE: func(cmd *cobra.Command, args []string) error {
			cfg, err := config.Load()
			if err != nil {
				return err
			}
			prov, err := providerFlag(cfg, provider)
			if err != nil {
				return err
			}
			st, _ := config.LoadState()
			if st.PodID != "" {
				return hint(fmt.Errorf("this project already has pod %s (%s)", st.PodID, st.ProviderName()),
					"remote-pod status (or remote-pod down to replace it)")
			}
			ctx := context.Background()
			token, tag := newToken(), newTag()
			policy, err := grant.PodPolicy(tag, allowPut, allowGet)
			if err != nil {
				return err
			}
			env, err := grant.PodEnv(tag, nil) // no key: the pod's store access is signed by this machine (remote-pod signer)
			if err != nil {
				return err
			}
			env["REMOTE_POD_TOKEN"], env["REMOTE_POD_IDLE_MIN"] = token, fmt.Sprint(idle)
			if prov == "vast" {
				for _, f := range []string{"cpu", "cloud", "volume", "flavor"} {
					if cmd.Flags().Changed(f) {
						return hint(fmt.Errorf("--%s is RunPod's (vast.ai rents GPU machines from offers)", f),
							"remote-pod up --provider vast --gpu 4090")
					}
				}
				if gpu == "" && offer == 0 {
					return hint(errors.New("give --gpu or --offer"), "remote-pod up --provider vast --gpu 4090")
				}
				if !cmd.Flags().Changed("image") {
					image = "cuda"
				}
				if !cmd.Flags().Changed("disk") {
					disk = 80
				}
				ref := imageRef(cfg, image)
				vs, err := createVast(ctx, cfg, vastPlan{GPU: gpu, Offer: offer, MaxPrice: maxPrice, DiskGB: disk,
					Image: ref, Label: name, Env: env})
				if err != nil {
					return err
				}
				vs.Token, vs.Tag, vs.Policy = token, tag, policy
				if err := config.SaveState(vs); err != nil {
					return err
				}
				ui.Note("instance %s running at %s; waiting for its agent (the image may still be starting)", vs.PodID, vs.URL)
				agentCl, _, err := agentClient()
				if err != nil {
					return err
				}
				if err := waitAgent(agentCl, 15*time.Minute); err != nil {
					return err
				}
				ui.Success("ready: remote-pod sh nvidia-smi")
				return nil
			}
			if offer != 0 || maxPrice != 0 {
				return hint(errors.New("--offer and --max-price are vast.ai's"), "remote-pod up --provider vast --gpu 4090")
			}
			c, _, err := api()
			if err != nil {
				return err
			}
			if (gpu == "") == (cpus == 0) {
				return hint(errors.New("give either --gpu or --cpu"), `remote-pod up --gpu "RTX 4090"  or  remote-pod up --cpu 8`)
			}
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
			if err := config.SaveState(&config.State{Provider: "runpod", PodID: id, Token: token, Image: ref, GPU: gpuID,
				Tag: tag, Policy: policy}); err != nil {
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
	up.Flags().StringVar(&provider, "provider", "", `runpod or vast (default: the settings' provider, else runpod)`)
	up.Flags().StringVar(&gpu, "gpu", "", `GPU type: an id or display name ("RTX 4090", "5090"); see remote-pod gpus`)
	up.Flags().StringVar(&cloud, "cloud", "any", "RunPod: community, secure, or any (community first)")
	up.Flags().IntVar(&cpus, "cpu", 0, "RunPod: a CPU-only pod with this many vCPUs (2 to 32; see remote-pod cpus)")
	up.Flags().StringVar(&flavor, "flavor", "general", "RunPod CPU pods: compute (2 GB RAM / vCPU), general (4 GB), memory (8 GB), or an id (cpu3c)")
	up.Flags().StringVar(&image, "image", "train", "train (lbc + agent), dev (+ nvcc, cmake, ninja, Nsight), cpu (data work), cuda (public CUDA runtime; vast.ai's default), or an image reference")
	up.Flags().IntVar(&disk, "disk", 40, "container disk in GB (datasets and runs live here; checkpoints also go to R2; vast.ai default 80)")
	up.Flags().IntVar(&volume, "volume", 0, "RunPod: persistent /workspace volume in GB (0: none; data lives in R2)")
	up.Flags().IntVar(&idle, "idle", 60, "stop the pod after this many minutes with nothing running (0: never)")
	up.Flags().StringVar(&name, "name", "remote-pod", "the pod's name in the provider's console")
	up.Flags().Float64Var(&maxPrice, "max-price", 0, "vast.ai: the most $/hr to pay (0: any)")
	up.Flags().Int64Var(&offer, "offer", 0, "vast.ai: rent this offer id (see remote-pod gpus --provider vast --gpu ...)")
	up.Flags().StringArrayVar(&allowPut, "allow-put", nil, "the pod may upload under this store prefix too, e.g. runs/my-run/ (repeatable; see remote-pod signer)")
	up.Flags().StringArrayVar(&allowGet, "allow-get", nil, "the pod may download this exact store key too (repeatable)")

	status := &cobra.Command{
		Use:   "status",
		Short: "The pod, its agent, its jobs and the account balance",
		Long: `The project's pod (on whichever provider it runs), whether its agent answers, its jobs, and that provider's
balance. On vast.ai it also shows the ssh command, and updates the agent's address if the instance moved to
another public port (after a stop and start).`,
		Example: "  remote-pod status\n  remote-pod status --json",
		GroupID: "pod",
		RunE: func(cmd *cobra.Command, args []string) error {
			st, _ := config.LoadState()
			out := map[string]any{"state": st}
			cfg, err := config.Load()
			if err != nil {
				return err
			}
			ctx := context.Background()
			prov := st.ProviderName()
			if st.PodID == "" {
				prov = cfg.Provider
			}
			if cl, err := cloudFor(cfg, prov); err == nil {
				out["provider"] = prov
				if bal, perHr, err := cl.Balance(ctx); err == nil {
					out["balance"], out["spend_per_hour"] = bal, perHr
				}
				if st.PodID != "" {
					if pod, err := cl.Pod(ctx, st.PodID); err == nil {
						out["pod"] = pod
						if prov == "vast" && pod.AgentURL != "" && pod.AgentURL != st.URL { // moved after a restart
							st.URL = pod.AgentURL
							_ = config.SaveState(st)
						}
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
					fmt.Printf("%s $%.2f on %s  (spending $%.3f/hr)\n", ui.Bold("Balance"), b, prov, out["spend_per_hour"])
				}
				if st.PodID == "" && st.URL == "" {
					fmt.Println(ui.Dim("No pod for this project.") + "  remote-pod up --gpu \"RTX 4090\"")
					return
				}
				if p, ok := out["pod"].(*podInfo); ok {
					up := "-"
					if p.Uptime > 0 {
						up = p.Uptime.String()
					}
					ui.Table([]string{"Pod", "Provider", "State", "GPU", "$/hr", "Uptime", "Image"},
						[][]string{{p.ID, p.Provider, ui.State(strings.ToUpper(p.State)), p.GPU, fmt.Sprintf("%.3f", p.PerHour), up, p.Image}})
					if p.SSH != "" {
						fmt.Println(ui.Dim("SSH: ") + p.SSH)
					}
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

	var all bool
	var gpusProvider, gpusGPU string
	var gpusDisk, gpusLimit int
	gpus := &cobra.Command{
		Use:   "gpus",
		Short: "GPU types in stock now, cheapest first",
		Long: `RunPod: every GPU type in stock with its lowest, community and secure price per hour.

vast.ai (--provider vast): without --gpu, one line per GPU type: the cheapest offer that passes the floors up uses
(verified host, single GPU, reliability >= 0.98, >= 500 Mbps download, --disk GB free), and how many offers do. With
--gpu, the cheapest offers of that type, with the offer id that up --offer takes.`,
		Example: `  remote-pod gpus
  remote-pod gpus --all                                     also the types out of stock
  remote-pod gpus --provider vast                           vast.ai, one line per GPU type
  remote-pod gpus --provider vast --gpu 4090 --disk 100     the cheapest RTX 4090 offers with 100 GB free`,
		GroupID: "pod",
		RunE: func(cmd *cobra.Command, args []string) error {
			cfg, err := config.Load()
			if err != nil {
				return err
			}
			prov, err := providerFlag(cfg, gpusProvider)
			if err != nil {
				return err
			}
			if prov == "vast" {
				return vastGPUs(context.Background(), cfg, gpusGPU, gpusDisk, gpusLimit)
			}
			c, _, err := api()
			if err != nil {
				return err
			}
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
				if gpusGPU != "" && !strings.Contains(strings.ToLower(g.ID+" "+g.DisplayName), strings.ToLower(gpusGPU)) {
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
	gpus.Flags().StringVar(&gpusProvider, "provider", "", "runpod or vast (default: the settings' provider)")
	gpus.Flags().BoolVar(&all, "all", false, "RunPod: also the types out of stock")
	gpus.Flags().StringVar(&gpusGPU, "gpu", "", `only this GPU type ("4090", "A100"); on vast.ai: list its offers`)
	gpus.Flags().IntVar(&gpusDisk, "disk", 80, "vast.ai: the free disk an offer needs (GB)")
	gpus.Flags().IntVar(&gpusLimit, "limit", 15, "vast.ai --gpu: how many offers")

	stop := &cobra.Command{
		Use:   "stop",
		Short: "Stop the pod (the GPU is released and billing stops)",
		Long: `Stop the project's pod. RunPod: the container disk is wiped, a /workspace volume is kept. vast.ai: the disk is
kept and still billed for storage (cents per day), and the GPU may be rented by someone else before a restart;
remote-pod down deletes the instance and ends all billing.`,
		Example: "  remote-pod stop",
		GroupID: "pod",
		RunE: func(cmd *cobra.Command, args []string) error {
			cl, st, err := podCloud()
			if err != nil {
				return err
			}
			if err := cl.Stop(context.Background(), st.PodID); err != nil {
				return err
			}
			ui.Success("pod %s stopped (%s)", st.PodID, cl.Name())
			return nil
		},
	}

	var yes bool
	down := &cobra.Command{
		Use:     "down",
		Short:   "Delete the pod and its disk (checkpoints in R2 stay); all billing ends",
		Example: "  remote-pod down\n  remote-pod down --yes        no question (scripts)",
		GroupID: "pod",
		RunE: func(cmd *cobra.Command, args []string) error {
			cl, st, err := podCloud()
			if err != nil {
				return err
			}
			if !yes {
				if !term.IsTerminal(int(os.Stdin.Fd())) {
					return hint(errors.New("down deletes the pod and its disk: confirm with --yes"), "remote-pod down --yes")
				}
				fmt.Fprintf(os.Stderr, "Delete %s pod %s and its disk? [y/N] ", cl.Name(), st.PodID)
				ans, _ := bufio.NewReader(os.Stdin).ReadString('\n')
				if strings.ToLower(strings.TrimSpace(ans)) != "y" {
					return errors.New("not deleted")
				}
			}
			if err := cl.Remove(context.Background(), st.PodID); err != nil {
				if !strings.Contains(strings.ToLower(err.Error()), "not found") {
					return err
				}
				ui.Note("pod %s no longer exists (already deleted)", st.PodID) // only the local state is left
			}
			_ = config.SaveState(&config.State{})
			ui.Success("pod %s deleted (%s)", st.PodID, cl.Name())
			return nil
		},
	}
	down.Flags().BoolVarP(&yes, "yes", "y", false, "do not ask")

	var vcpu int
	cpusCmd := &cobra.Command{
		Use:     "cpus",
		Short:   "CPU-only pod families, with the stock and price of a size (RunPod)",
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
	return []*cobra.Command{gpus, cpusCmd, up, status, stop, down, balanceCommand(), configCommand()}
}

// podCloud is the project's pod and its provider's API.
func podCloud() (cloud, *config.State, error) {
	cfg, err := config.Load()
	if err != nil {
		return nil, nil, err
	}
	st, _ := config.LoadState()
	if st.PodID == "" {
		return nil, nil, hint(errors.New("no pod for this project"), `remote-pod up --gpu "RTX 4090"`)
	}
	cl, err := cloudFor(cfg, st.ProviderName())
	return cl, st, err
}

// vastGPUs prints vast.ai's offers: one line per GPU type, or the offers of one type.
func vastGPUs(ctx context.Context, cfg *config.Config, gpu string, disk, limit int) error {
	cl, err := cloudFor(cfg, "vast")
	if err != nil {
		return err
	}
	c := cl.(vastCloud).c
	q := vastOfferFloors
	q.DiskGB = disk
	if gpu == "" {
		q.Limit = 3000
		offers, err := c.Offers(ctx, q)
		if err != nil {
			return err
		}
		type row struct {
			GPU      string  `json:"gpu"`
			VRAM     float64 `json:"vram_gb"`
			From     float64 `json:"from_per_hour"`
			Offers   int     `json:"offers"`
			Location string  `json:"cheapest_location"`
		}
		by := map[string]*row{}
		var order []string
		for _, o := range offers { // cheapest first: the first offer of a type is its price
			r, ok := by[o.GPUName]
			if !ok {
				r = &row{GPU: o.GPUName, VRAM: o.GPURAM / 1024, From: o.Price, Location: o.Location}
				by[o.GPUName] = r
				order = append(order, o.GPUName)
			}
			r.Offers++
		}
		var shown []row
		var rows [][]string
		for _, n := range order {
			r := by[n]
			shown = append(shown, *r)
			rows = append(rows, []string{r.GPU, fmt.Sprintf("%.0f GB", r.VRAM), fmt.Sprintf("$%.3f", r.From),
				fmt.Sprint(r.Offers), r.Location})
		}
		ui.Print(shown, func() {
			ui.Table([]string{"GPU", "VRAM", "From/hr", "Offers", "Cheapest in"}, rows)
			fmt.Println(ui.Dim(fmt.Sprintf("verified hosts, 1 GPU, reliability >= %.2f, >= %.0f Mbps, %d GB disk free; offers: remote-pod gpus --provider vast --gpu NAME",
				q.MinReliability, q.MinInetDown, disk)))
		})
		return nil
	}
	name, err := vastGPUName(ctx, c, gpu)
	if err != nil {
		return err
	}
	q.GPU, q.Limit = name, limit
	offers, err := c.Offers(ctx, q)
	if err != nil {
		return err
	}
	var rows [][]string
	for _, o := range offers {
		rows = append(rows, []string{fmt.Sprint(o.ID), o.GPUName, fmt.Sprintf("$%.3f", o.Price),
			fmt.Sprintf("%.0f", o.CPUCores), fmt.Sprintf("%.0f GB", o.CPURAM/1024), fmt.Sprintf("%.0f GB", o.DiskSpace),
			fmt.Sprintf("%.0f", o.InetDown), fmt.Sprintf("%.3f", o.Reliability), o.Location})
	}
	ui.Print(offers, func() {
		ui.Table([]string{"Offer", "GPU", "$/hr", "vCPUs", "RAM", "Disk free", "Mbps down", "Reliability", "Location"}, rows)
		fmt.Println(ui.Dim("rent one: remote-pod up --provider vast --offer ID   (or --gpu \"" + name + "\" for the cheapest)"))
	})
	return nil
}

func balanceCommand() *cobra.Command {
	return &cobra.Command{
		Use:   "balance",
		Short: "Credit left and spend per hour on every provider with a key",
		Long: `The account credit on each provider whose API key is set, what its running pods cost per hour together, and how
long the credit lasts at that rate. A provider without a key is listed as not configured.`,
		Example: "  remote-pod balance\n  remote-pod balance --json",
		GroupID: "pod",
		RunE: func(cmd *cobra.Command, args []string) error {
			cfg, err := config.Load()
			if err != nil {
				return err
			}
			ctx := context.Background()
			type row struct {
				Provider  string  `json:"provider"`
				Credit    float64 `json:"credit"`
				PerHour   float64 `json:"spend_per_hour"`
				HoursLeft float64 `json:"hours_left,omitempty"`
				Error     string  `json:"error,omitempty"`
			}
			var out []row
			var rows [][]string
			for _, p := range config.Providers {
				cl, err := cloudFor(cfg, p)
				if err != nil {
					out = append(out, row{Provider: p, Error: "not configured"})
					rows = append(rows, []string{p, ui.Dim("not configured"), "", ""})
					continue
				}
				credit, perHr, err := cl.Balance(ctx)
				if err != nil {
					out = append(out, row{Provider: p, Error: err.Error()})
					rows = append(rows, []string{p, ui.Bad(err.Error()), "", ""})
					continue
				}
				r := row{Provider: p, Credit: credit, PerHour: perHr}
				left := "-"
				if perHr > 0 {
					r.HoursLeft = credit / perHr
					left = fmt.Sprintf("%.1f h", r.HoursLeft)
					if r.HoursLeft < 3 {
						left = ui.Warn(left)
					}
				}
				out = append(out, r)
				credStr := fmt.Sprintf("$%.2f", credit)
				if credit <= 0 {
					credStr = ui.Bad(credStr)
				}
				rows = append(rows, []string{p, credStr, fmt.Sprintf("$%.3f", perHr), left})
			}
			ui.Print(out, func() { ui.Table([]string{"Provider", "Credit", "Spend/hr", "Lasts"}, rows) })
			return nil
		},
	}
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
			_, vastErr := cfg.VastKey()
			set := func(ok bool) string {
				if ok {
					return ui.OK("set")
				}
				return ui.Warn("missing")
			}
			out := map[string]any{"path": config.Path(), "provider": cfg.Provider, "runpod_key": keyErr == nil,
				"vast_key": vastErr == nil, "r2": cfg.HasR2(), "r2_endpoint": cfg.R2.Endpoint, "r2_bucket": cfg.R2.Bucket,
				"images": cfg.Images}
			ui.Print(out, func() {
				fmt.Println(ui.Bold("Settings file: ") + config.Path())
				ui.Table([]string{"Setting", "Value"}, [][]string{
					{"Default provider", cfg.Provider},
					{"RunPod API key", set(keyErr == nil)},
					{"vast.ai API key", set(vastErr == nil)},
					{"R2 endpoint", cfg.R2.Endpoint},
					{"R2 bucket", cfg.R2.Bucket},
					{"R2 keys", set(cfg.HasR2())},
					{"Train image", cfg.Images.Train},
					{"Dev image", cfg.Images.Dev},
					{"CUDA image", cfg.Images.CUDA},
					{"Randomness for secrets", strings.Join(entropy.Sources(), " + ")},
				})
				fmt.Println(ui.Dim(`Example config.toml:
  provider = "runpod"                   # or "vast": where up / train rent a GPU (--provider overrides)
  [runpod]
  api_key = "..."                       # or leave out: runpodctl's key is used
  [vast]
  api_key = "..."                       # or leave out: $VAST_API_KEY, else the vastai CLI's key
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
