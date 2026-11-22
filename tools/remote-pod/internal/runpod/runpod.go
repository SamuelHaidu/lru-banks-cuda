// Package runpod is a small client of RunPod's GraphQL API (https://api.runpod.io/graphql): create, stop, terminate
// and read pods, the GPU types in stock, the account balance.
package runpod

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"sort"
	"strings"
	"time"
)

const Endpoint = "https://api.runpod.io/graphql"

type Client struct {
	Key  string
	URL  string // default Endpoint (tests point it at a fake server)
	HTTP *http.Client
}

func New(key string) *Client {
	return &Client{Key: key, URL: Endpoint, HTTP: &http.Client{Timeout: 60 * time.Second}}
}

// Do sends one GraphQL query and decodes `data` into out.
func (c *Client) Do(ctx context.Context, query string, vars map[string]any, out any) error {
	body, _ := json.Marshal(map[string]any{"query": query, "variables": vars})
	req, err := http.NewRequestWithContext(ctx, "POST", c.URL, bytes.NewReader(body))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+c.Key)
	resp, err := c.HTTP.Do(req)
	if err != nil {
		return fmt.Errorf("runpod api: %w", err)
	}
	defer resp.Body.Close()
	var r struct {
		Data   json.RawMessage `json:"data"`
		Errors []struct {
			Message string `json:"message"`
		} `json:"errors"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&r); err != nil {
		return fmt.Errorf("runpod api: HTTP %d: %w", resp.StatusCode, err)
	}
	if len(r.Errors) > 0 {
		msgs := make([]string, len(r.Errors))
		for i, e := range r.Errors {
			msgs[i] = e.Message
		}
		return fmt.Errorf("runpod api: %s", strings.Join(msgs, "; "))
	}
	if resp.StatusCode != 200 {
		return fmt.Errorf("runpod api: HTTP %d", resp.StatusCode)
	}
	if out != nil {
		return json.Unmarshal(r.Data, out)
	}
	return nil
}

// Balance is the account's credit and current spend per hour.
func (c *Client) Balance(ctx context.Context) (balance, perHour float64, err error) {
	var d struct {
		Myself struct {
			ClientBalance     float64 `json:"clientBalance"`
			CurrentSpendPerHr float64 `json:"currentSpendPerHr"`
		} `json:"myself"`
	}
	err = c.Do(ctx, `query { myself { clientBalance currentSpendPerHr } }`, nil, &d)
	return d.Myself.ClientBalance, d.Myself.CurrentSpendPerHr, err
}

// GPU is one GPU type with its prices and stock (lowest price for one GPU).
type GPU struct {
	ID            string  `json:"id"`
	DisplayName   string  `json:"displayName"`
	MemoryInGb    int     `json:"memoryInGb"`
	SecurePrice   float64 `json:"securePrice"`
	CommunityPrice float64 `json:"communityPrice"`
	LowestPrice   struct {
		UninterruptablePrice float64 `json:"uninterruptablePrice"`
		StockStatus          string  `json:"stockStatus"`
	} `json:"lowestPrice"`
}

// GPUs lists the GPU types, cheapest first among those in stock.
func (c *Client) GPUs(ctx context.Context) ([]GPU, error) {
	var d struct {
		GPUTypes []GPU `json:"gpuTypes"`
	}
	q := `query { gpuTypes { id displayName memoryInGb securePrice communityPrice
	        lowestPrice(input: {gpuCount: 1}) { uninterruptablePrice stockStatus } } }`
	if err := c.Do(ctx, q, nil, &d); err != nil {
		return nil, err
	}
	g := d.GPUTypes
	inStock := func(x GPU) bool { return x.LowestPrice.StockStatus != "" && x.LowestPrice.UninterruptablePrice > 0 }
	sort.SliceStable(g, func(i, j int) bool {
		if inStock(g[i]) != inStock(g[j]) {
			return inStock(g[i])
		}
		return g[i].LowestPrice.UninterruptablePrice < g[j].LowestPrice.UninterruptablePrice
	})
	return g, nil
}

// CreateInput is a pod to create.
type CreateInput struct {
	Name          string
	Image         string
	GPUTypeID     string
	Cloud         string // COMMUNITY, SECURE or ALL
	DiskGB        int    // container disk
	VolumeGB      int    // /workspace volume (0: none)
	Env           map[string]string
	Ports         string // "8080/http"
}

// Pod is what the API reports about a pod.
type Pod struct {
	ID            string  `json:"id"`
	Name          string  `json:"name"`
	DesiredStatus string  `json:"desiredStatus"`
	CostPerHr     float64 `json:"costPerHr"`
	ImageName     string  `json:"imageName"`
	Machine       struct {
		GPUDisplayName string `json:"gpuDisplayName"`
	} `json:"machine"`
	Runtime *struct {
		UptimeInSeconds int `json:"uptimeInSeconds"`
	} `json:"runtime"`
}

// Create deploys an on-demand pod and returns its id.
func (c *Client) Create(ctx context.Context, in CreateInput) (string, error) {
	env := []map[string]string{}
	for k, v := range in.Env {
		env = append(env, map[string]string{"key": k, "value": v})
	}
	sort.Slice(env, func(i, j int) bool { return env[i]["key"] < env[j]["key"] })
	vars := map[string]any{"input": map[string]any{
		"cloudType": in.Cloud, "gpuCount": 1, "gpuTypeId": in.GPUTypeID, "name": in.Name, "imageName": in.Image,
		"containerDiskInGb": in.DiskGB, "volumeInGb": in.VolumeGB, "ports": in.Ports, "env": env,
		"volumeMountPath": "/workspace",
	}}
	var d struct {
		Pod struct {
			ID string `json:"id"`
		} `json:"podFindAndDeployOnDemand"`
	}
	q := `mutation($input: PodFindAndDeployOnDemandInput) { podFindAndDeployOnDemand(input: $input) { id } }`
	if err := c.Do(ctx, q, vars, &d); err != nil {
		return "", err
	}
	if d.Pod.ID == "" {
		return "", fmt.Errorf("runpod api: no pod created (no %s machine with a %s free?)", in.Cloud, in.GPUTypeID)
	}
	return d.Pod.ID, nil
}

func (c *Client) Get(ctx context.Context, id string) (*Pod, error) {
	var d struct {
		Pod *Pod `json:"pod"`
	}
	q := `query($input: PodFilter) { pod(input: $input) { id name desiredStatus costPerHr imageName
	        machine { gpuDisplayName } runtime { uptimeInSeconds } } }`
	if err := c.Do(ctx, q, map[string]any{"input": map[string]any{"podId": id}}, &d); err != nil {
		return nil, err
	}
	if d.Pod == nil {
		return nil, fmt.Errorf("no pod %s", id)
	}
	return d.Pod, nil
}

// Stop stops the pod (the container disk is wiped, the volume kept; billing stops except for the volume).
func (c *Client) Stop(ctx context.Context, id string) error {
	return c.Do(ctx, `mutation($input: PodStopInput!) { podStop(input: $input) { id } }`,
		map[string]any{"input": map[string]any{"podId": id}}, nil)
}

// Terminate deletes the pod and its volume.
func (c *Client) Terminate(ctx context.Context, id string) error {
	return c.Do(ctx, `mutation($input: PodTerminateInput!) { podTerminate(input: $input) }`,
		map[string]any{"input": map[string]any{"podId": id}}, nil)
}
