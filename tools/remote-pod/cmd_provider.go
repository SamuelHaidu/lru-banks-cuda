package main

// The GPU clouds behind the pod commands. status, stop, down and balance go through `cloud` and behave the same on
// both; up and train create the pod with each provider's own client (RunPod: community / secure cloud and CPU pods;
// vast.ai: a machine offer, the agent on the host's public IP behind a pinned TLS certificate).

import (
	"context"
	"encoding/base64"
	"errors"
	"fmt"
	"os"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/SamuelHaidu/rnn-research/tools/remote-pod/internal/boot"
	"github.com/SamuelHaidu/rnn-research/tools/remote-pod/internal/config"
	"github.com/SamuelHaidu/rnn-research/tools/remote-pod/internal/pin"
	"github.com/SamuelHaidu/rnn-research/tools/remote-pod/internal/runpod"
	"github.com/SamuelHaidu/rnn-research/tools/remote-pod/internal/ui"
	"github.com/SamuelHaidu/rnn-research/tools/remote-pod/internal/vast"
)

// podInfo is a pod as the status table shows it, whatever the provider.
type podInfo struct {
	ID       string        `json:"id"`
	Provider string        `json:"provider"`
	State    string        `json:"state"`
	GPU      string        `json:"gpu"`
	PerHour  float64       `json:"per_hour"`
	Uptime   time.Duration `json:"uptime_ns"`
	Image    string        `json:"image"`
	Location string        `json:"location,omitempty"`
	SSH      string        `json:"ssh,omitempty"` // vast.ai: the ssh command
	AgentURL string        `json:"agent_url,omitempty"`
}

// cloud is what the provider-neutral commands need.
type cloud interface {
	Name() string
	Balance(ctx context.Context) (credit, perHour float64, err error)
	Pod(ctx context.Context, id string) (*podInfo, error)
	Stop(ctx context.Context, id string) error
	Remove(ctx context.Context, id string) error
}

// cloudFor opens a provider's API with its key from the settings.
func cloudFor(cfg *config.Config, provider string) (cloud, error) {
	switch provider {
	case "runpod":
		key, err := cfg.RunPodKey()
		if err != nil {
			return nil, hint(err, "remote-pod config (shows where the settings go)")
		}
		c := runpod.New(key)
		if u := os.Getenv("REMOTE_POD_RUNPOD_API"); u != "" { // tests: a fake provider
			c.URL = u
		}
		return runpodCloud{c}, nil
	case "vast":
		key, err := cfg.VastKey()
		if err != nil {
			return nil, hint(err, "remote-pod config (shows where the settings go)")
		}
		c := vast.New(key)
		if u := os.Getenv("REMOTE_POD_VAST_API"); u != "" { // tests: a fake provider
			c.URL = u
		}
		return vastCloud{c}, nil
	}
	return nil, config.CheckProvider(provider)
}

// providerFlag is --provider's value, else the settings' default; it must be a known provider.
func providerFlag(cfg *config.Config, flag string) (string, error) {
	if flag == "" {
		return cfg.Provider, nil
	}
	return flag, config.CheckProvider(flag)
}

type runpodCloud struct{ c *runpod.Client }

func (r runpodCloud) Name() string { return "runpod" }
func (r runpodCloud) Balance(ctx context.Context) (float64, float64, error) {
	return r.c.Balance(ctx)
}
func (r runpodCloud) Pod(ctx context.Context, id string) (*podInfo, error) {
	p, err := r.c.Get(ctx, id)
	if err != nil {
		return nil, err
	}
	pi := &podInfo{ID: p.ID, Provider: "runpod", State: p.DesiredStatus, GPU: p.Machine.GPUDisplayName,
		PerHour: p.CostPerHr, Image: p.ImageName}
	if p.Runtime != nil {
		pi.Uptime = time.Duration(p.Runtime.UptimeInSeconds) * time.Second
	}
	return pi, nil
}
func (r runpodCloud) Stop(ctx context.Context, id string) error   { return r.c.Stop(ctx, id) }
func (r runpodCloud) Remove(ctx context.Context, id string) error { return r.c.Terminate(ctx, id) }

type vastCloud struct{ c *vast.Client }

func vastID(id string) (int64, error) {
	n, err := strconv.ParseInt(id, 10, 64)
	if err != nil {
		return 0, fmt.Errorf("not a vast.ai instance id: %q", id)
	}
	return n, nil
}

func (v vastCloud) Name() string { return "vast" }
func (v vastCloud) Balance(ctx context.Context) (float64, float64, error) {
	return v.c.Balance(ctx)
}
func (v vastCloud) Pod(ctx context.Context, id string) (*podInfo, error) {
	n, err := vastID(id)
	if err != nil {
		return nil, err
	}
	in, err := v.c.Get(ctx, n)
	if err != nil {
		return nil, err
	}
	pi := &podInfo{ID: id, Provider: "vast", State: in.State(), GPU: in.GPUName, PerHour: in.PerHour,
		Uptime: in.Uptime(), Image: in.Image, Location: in.Location}
	if in.PublicIP != "" && in.HostPort(22) > 0 {
		pi.SSH = fmt.Sprintf("ssh -p %d root@%s", in.HostPort(22), in.PublicIP)
	}
	if p := in.HostPort(config.AgentPort); in.PublicIP != "" && p > 0 {
		pi.AgentURL = fmt.Sprintf("https://%s:%d", in.PublicIP, p)
	}
	return pi, nil
}
func (v vastCloud) Stop(ctx context.Context, id string) error {
	n, err := vastID(id)
	if err != nil {
		return err
	}
	return v.c.Stop(ctx, n)
}
func (v vastCloud) Remove(ctx context.Context, id string) error {
	n, err := vastID(id)
	if err != nil {
		return err
	}
	return v.c.Destroy(ctx, n)
}

// vastGPUName matches a GPU against the names vast.ai offers ("4090" -> "RTX 4090"; exact names win).
func vastGPUName(ctx context.Context, c *vast.Client, want string) (string, error) {
	offers, err := c.Offers(ctx, vast.OfferQuery{Limit: 3000})
	if err != nil {
		return "", err
	}
	names := map[string]bool{}
	for _, o := range offers {
		names[o.GPUName] = true
	}
	w := strings.ToLower(strings.ReplaceAll(want, "_", " "))
	var hits []string
	for n := range names {
		if strings.ToLower(n) == w {
			return n, nil
		}
		if strings.Contains(strings.ToLower(n), w) {
			hits = append(hits, n)
		}
	}
	sort.Strings(hits)
	if len(hits) == 1 {
		return hits[0], nil
	}
	if len(hits) > 1 {
		return "", hint(fmt.Errorf("%q matches several vast.ai GPUs: %s", want, strings.Join(hits, ", ")),
			"remote-pod gpus --provider vast")
	}
	return "", hint(fmt.Errorf("no vast.ai offer has a GPU matching %q", want), "remote-pod gpus --provider vast")
}

// vastPlan is what up / train ask vast.ai for.
type vastPlan struct {
	GPU      string  // as the user wrote it
	Offer    int64   // > 0: this offer, no search
	MaxPrice float64 // $/hr, 0: any
	DiskGB   int
	Image    string
	Label    string
	Env      map[string]string // the agent's settings and grantEnv: never a store key
}

// vastOfferFloors keep flaky or slow hosts out of the automatic choice (--offer bypasses them).
var vastOfferFloors = vast.OfferQuery{MinReliability: 0.98, MinInetDown: 500}

// createVast rents the cheapest matching offer (the next ones when it is taken meanwhile) with the agent's bootstrap,
// waits until the agent's port is published, and returns the state to save (the agent is not answering yet).
func createVast(ctx context.Context, cfg *config.Config, p vastPlan) (*config.State, error) {
	cl, err := cloudFor(cfg, "vast")
	if err != nil {
		return nil, err
	}
	c := cl.(vastCloud).c
	if isOurImage(cfg, p.Image) {
		return nil, hint(fmt.Errorf("vast.ai hosts cannot pull our private image %s", p.Image),
			"--image cuda (a public CUDA image; the agent and lbc come from R2)")
	}
	_, bootEnv, err := bootstrap(ctx, cfg)
	if err != nil {
		return nil, err
	}
	certPEM, keyPEM, err := pin.New()
	if err != nil {
		return nil, err
	}
	env := map[string]string{"REMOTE_POD_TLS_CERT": base64.StdEncoding.EncodeToString(certPEM),
		"REMOTE_POD_TLS_KEY": base64.StdEncoding.EncodeToString(keyPEM)}
	for k, v := range bootEnv {
		env[k] = v
	}
	for k, v := range p.Env {
		env[k] = v
	}
	var offers []vast.Offer
	gpuName := p.GPU
	if p.Offer > 0 {
		offers = []vast.Offer{{ID: p.Offer, GPUName: p.GPU}}
	} else {
		if gpuName, err = vastGPUName(ctx, c, p.GPU); err != nil {
			return nil, err
		}
		q := vastOfferFloors
		q.GPU, q.DiskGB, q.MaxPrice, q.Limit = gpuName, p.DiskGB, p.MaxPrice, 5
		if offers, err = c.Offers(ctx, q); err != nil {
			return nil, err
		}
		if len(offers) == 0 {
			return nil, hint(fmt.Errorf("no vast.ai %s offer with %d GB disk under the floors (reliability %.2f, %.0f Mbps)%s",
				gpuName, p.DiskGB, q.MinReliability, q.MinInetDown, priceNote(p.MaxPrice)),
				"remote-pod gpus --provider vast --gpu \""+gpuName+"\"")
		}
	}
	var id int64
	var last error
	for _, o := range offers[:min(3, len(offers))] {
		ui.Note("renting vast.ai offer %d: %s at $%.3f/hr (%s)", o.ID, o.GPUName, o.Price, o.Location)
		id, last = c.Create(ctx, vast.CreateInput{Offer: o.ID, Label: p.Label, Image: p.Image, DiskGB: p.DiskGB,
			Env: env, Ports: []int{config.AgentPort}, Onstart: boot.VastOnstart})
		if last == nil {
			gpuName = o.GPUName
			break
		}
		ui.Note("%v", last)
	}
	if id == 0 {
		return nil, hint(last, "remote-pod gpus --provider vast")
	}
	st := &config.State{Provider: "vast", PodID: strconv.FormatInt(id, 10), Cert: string(certPEM), Image: p.Image,
		GPU: gpuName}
	// the agent's address is known once the container runs and its port is published
	if err := config.SaveState(st); err != nil {
		return nil, err
	}
	ui.Success("instance %d created (%s); waiting for it to start", id, gpuName)
	deadline, lastMsg := time.Now().Add(20*time.Minute), ""
	for time.Now().Before(deadline) {
		in, err := c.Get(ctx, id)
		if err == nil {
			if msg := strings.TrimSpace(in.StatusMsg); msg != "" && msg != lastMsg {
				ui.Note("%s", firstLine(msg))
				lastMsg = msg
			}
			if p := in.HostPort(config.AgentPort); in.ActualStatus == "running" && in.PublicIP != "" && p > 0 {
				st.URL = fmt.Sprintf("https://%s:%d", in.PublicIP, p)
				return st, config.SaveState(st)
			}
		}
		time.Sleep(5 * time.Second)
	}
	return st, hint(errors.New("the instance did not start within 20 minutes"), "remote-pod status (or remote-pod down)")
}

func priceNote(maxPrice float64) string {
	if maxPrice > 0 {
		return fmt.Sprintf(", under $%.2f/hr", maxPrice)
	}
	return ""
}

func firstLine(s string) string {
	if i := strings.IndexByte(s, '\n'); i >= 0 {
		return s[:i]
	}
	return s
}
