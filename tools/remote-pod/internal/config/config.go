// Package config holds the user's settings (~/.config/remote-pod/config.toml) and the project's pod state
// (.remote-pod/state.json at the repository root).
package config

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/BurntSushi/toml"

	"github.com/SamuelHaidu/rnn-research/tools/remote-pod/internal/grant"
)

// Config is the user's settings. Secrets live only here (mode 0600), never in the repository or an image.
type Config struct {
	Provider string `toml:"provider"` // where up / train rent a GPU: "runpod" (default) or "vast"
	RunPod   RunPod `toml:"runpod"`
	Vast     Vast   `toml:"vast"`
	R2       R2     `toml:"r2"`
	Images   Images `toml:"images"`
}

type RunPod struct {
	APIKey string `toml:"api_key"` // empty: read from ~/.runpod/config.toml (runpodctl's)
}

type Vast struct {
	APIKey string `toml:"api_key"` // empty: $VAST_API_KEY, else ~/.config/vastai/vast_api_key (the vastai CLI's)
}

// Providers are the GPU clouds remote-pod rents from.
var Providers = []string{"runpod", "vast"}

// R2 is any S3-compatible store (Cloudflare R2: https://<account>.r2.cloudflarestorage.com).
type R2 struct {
	Endpoint  string `toml:"endpoint"`
	Bucket    string `toml:"bucket"`
	AccessKey string `toml:"access_key"`
	SecretKey string `toml:"secret_key"`
	Region    string `toml:"region"` // "auto" for R2
}

type Images struct {
	Train string `toml:"train"`
	Dev   string `toml:"dev"`
	CPU   string `toml:"cpu"`
	// CUDA is a public CUDA runtime image ("--image cuda"): the agent and lbc are fetched from R2 at boot. The default
	// on vast.ai, whose hosts cannot pull our private images.
	CUDA string `toml:"cuda"`
}

// Dir is ~/.config/remote-pod (or $REMOTE_POD_CONFIG_DIR).
func Dir() string {
	if d := os.Getenv("REMOTE_POD_CONFIG_DIR"); d != "" {
		return d
	}
	home, _ := os.UserHomeDir()
	return filepath.Join(home, ".config", "remote-pod")
}

func Path() string { return filepath.Join(Dir(), "config.toml") }

// Load reads the config; a missing file is an empty config.
func Load() (*Config, error) {
	c := &Config{}
	if _, err := os.Stat(Path()); err == nil {
		if _, err := toml.DecodeFile(Path(), c); err != nil {
			return nil, fmt.Errorf("%s: %w", Path(), err)
		}
	}
	if c.R2.Region == "" {
		c.R2.Region = "auto"
	}
	if c.Images.Train == "" {
		c.Images.Train = "ghcr.io/samuelhaidu/remote-pod:train"
	}
	if c.Images.CPU == "" {
		c.Images.CPU = "ghcr.io/samuelhaidu/remote-pod:cpu"
	}
	if c.Images.Dev == "" {
		c.Images.Dev = "ghcr.io/samuelhaidu/remote-pod:dev"
	}
	if c.Images.CUDA == "" {
		c.Images.CUDA = "nvidia/cuda:12.8.1-runtime-ubuntu24.04"
	}
	if c.Provider == "" {
		c.Provider = "runpod"
	}
	if err := CheckProvider(c.Provider); err != nil {
		return nil, fmt.Errorf("%s: provider: %w", Path(), err)
	}
	return c, nil
}

// CheckProvider accepts the known provider names.
func CheckProvider(p string) error {
	for _, x := range Providers {
		if p == x {
			return nil
		}
	}
	return fmt.Errorf("unknown provider %q (runpod or vast)", p)
}

// VastKey is the configured key, $VAST_API_KEY, or the vastai CLI's.
func (c *Config) VastKey() (string, error) {
	if c.Vast.APIKey != "" {
		return c.Vast.APIKey, nil
	}
	if k := os.Getenv("VAST_API_KEY"); k != "" {
		return k, nil
	}
	home, _ := os.UserHomeDir()
	if b, err := os.ReadFile(filepath.Join(home, ".config", "vastai", "vast_api_key")); err == nil {
		if k := strings.TrimSpace(string(b)); k != "" {
			return k, nil
		}
	}
	return "", errors.New("no vast.ai API key: set vast.api_key in " + Path() + " (or $VAST_API_KEY, or vastai set api-key)")
}

// RunPodKey is the configured key, $RUNPOD_API_KEY, or runpodctl's.
func (c *Config) RunPodKey() (string, error) {
	if c.RunPod.APIKey != "" {
		return c.RunPod.APIKey, nil
	}
	if k := os.Getenv("RUNPOD_API_KEY"); k != "" {
		return k, nil
	}
	home, _ := os.UserHomeDir()
	var rc struct {
		APIKey string `toml:"apikey"`
	}
	if _, err := toml.DecodeFile(filepath.Join(home, ".runpod", "config.toml"), &rc); err == nil && rc.APIKey != "" {
		return rc.APIKey, nil
	}
	return "", errors.New("no RunPod API key: set runpod.api_key in " + Path() + " (or $RUNPOD_API_KEY)")
}

// HasR2 reports whether the store is configured.
func (c *Config) HasR2() bool {
	return c.R2.Endpoint != "" && c.R2.Bucket != "" && c.R2.AccessKey != ""
}

// State is the project's current pod: where its agent listens and the token to talk to it.
type State struct {
	Provider string `json:"provider,omitempty"` // "" = runpod (states written before vast.ai support)
	PodID    string `json:"pod_id,omitempty"`
	URL      string `json:"url,omitempty"` // the agent: set for vast.ai (https://<ip>:<port>) or a local agent
	Token    string `json:"token,omitempty"`
	Cert     string `json:"cert,omitempty"` // the agent's pinned TLS certificate (PEM; vast.ai)
	Image    string `json:"image,omitempty"`
	GPU      string `json:"gpu,omitempty"`
	// Tag is the pod's random name in the store (its job logs: jobs/<tag>/); Policy is what the signer may grant it.
	// Both live only here: the pod cannot widen its own policy.
	Tag    string       `json:"tag,omitempty"`
	Policy grant.Policy `json:"policy,omitzero"`
}

// ProviderName is the pod's provider (runpod for states without one).
func (s *State) ProviderName() string {
	if s.Provider == "" {
		return "runpod"
	}
	return s.Provider
}

// ProjectRoot is the nearest directory upward with .git (or the working directory).
func ProjectRoot() string {
	wd, _ := os.Getwd()
	for d := wd; ; d = filepath.Dir(d) {
		if _, err := os.Stat(filepath.Join(d, ".git")); err == nil {
			return d
		}
		if filepath.Dir(d) == d {
			return wd
		}
	}
}

func statePath() string { return filepath.Join(ProjectRoot(), ".remote-pod", "state.json") }

func LoadState() (*State, error) {
	s := &State{}
	b, err := os.ReadFile(statePath())
	if errors.Is(err, os.ErrNotExist) {
		return s, nil
	}
	if err != nil {
		return nil, err
	}
	return s, json.Unmarshal(b, s)
}

func SaveState(s *State) error {
	if err := os.MkdirAll(filepath.Dir(statePath()), 0o700); err != nil {
		return err
	}
	b, _ := json.MarshalIndent(s, "", "  ")
	return os.WriteFile(statePath(), b, 0o600)
}

// Endpoint is where the agent listens and its token: $REMOTE_POD_URL / $REMOTE_POD_TOKEN override the state.
func (s *State) Endpoint() (url, token string, err error) {
	url, token = s.URL, s.Token
	if url == "" && s.PodID != "" && s.ProviderName() == "runpod" {
		url = fmt.Sprintf("https://%s-%d.proxy.runpod.net", s.PodID, AgentPort)
	}
	if v := os.Getenv("REMOTE_POD_URL"); v != "" {
		url = v
	}
	if v := os.Getenv("REMOTE_POD_TOKEN"); v != "" {
		token = v
	}
	if url == "" {
		if s.PodID != "" {
			return "", "", fmt.Errorf("pod %s has no agent address yet (still starting? remote-pod status)", s.PodID)
		}
		return "", "", errors.New(`no pod: run remote-pod up --gpu "RTX 4090" (or set REMOTE_POD_URL for a local agent)`)
	}
	return url, token, nil
}

// AgentPort is the port the agent listens on inside the pod (exposed as <pod>-8080.proxy.runpod.net).
const AgentPort = 8080
