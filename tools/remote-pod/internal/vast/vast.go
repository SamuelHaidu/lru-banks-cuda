// Package vast is a small client of vast.ai's REST API (https://console.vast.ai/api/v0 and v1: the endpoints the official
// vastai CLI uses): offers (machines for rent), create / read / stop / start / destroy instances, the account credit.
//
// vast.ai rents whole machines from hosts; an instance is a container on one of them. Unlike RunPod there is no HTTPS
// proxy: a container port is published on the host's public IP under another number (Instance.HostPort).
package vast

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"github.com/SamuelHaidu/rnn-research/tools/remote-pod/internal/tlsroots"
	"io"
	"net/http"
	"net/url"
	"sort"
	"strings"
	"time"
)

const Endpoint = "https://console.vast.ai"

type Client struct {
	Key  string
	URL  string // default Endpoint (tests point it at a fake server)
	HTTP *http.Client
}

func New(key string) *Client {
	return &Client{Key: key, URL: Endpoint, HTTP: tlsroots.Client(60 * time.Second)}
}

// do sends one request (body: JSON of in, unless nil) and decodes the JSON answer into out.
func (c *Client) do(ctx context.Context, method, path string, in, out any) error {
	var body io.Reader
	if in != nil {
		b, err := json.Marshal(in)
		if err != nil {
			return err
		}
		body = bytes.NewReader(b)
	}
	req, err := http.NewRequestWithContext(ctx, method, c.URL+path, body)
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+c.Key)
	resp, err := c.HTTP.Do(req)
	if err != nil {
		return fmt.Errorf("vast api: %w", tlsroots.Explain("console.vast.ai", err))
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(resp.Body)
	if resp.StatusCode >= 300 {
		var e struct {
			Msg   string `json:"msg"`
			Error string `json:"error"`
		}
		_ = json.Unmarshal(raw, &e)
		msg := strings.TrimSpace(e.Msg)
		if msg == "" {
			msg = strings.TrimSpace(e.Error)
		}
		if msg == "" {
			msg = strings.TrimSpace(string(raw))
		}
		if len(msg) > 300 {
			msg = msg[:300] + "..."
		}
		return fmt.Errorf("vast api: HTTP %d: %s", resp.StatusCode, msg)
	}
	if out != nil {
		if err := json.Unmarshal(raw, out); err != nil {
			return fmt.Errorf("vast api: %s %s: %w", method, path, err)
		}
	}
	return nil
}

// Balance is the account's credit and what the running instances cost per hour together.
func (c *Client) Balance(ctx context.Context) (credit, perHour float64, err error) {
	var u struct {
		Credit float64 `json:"credit"`
	}
	if err = c.do(ctx, "GET", "/api/v0/users/current", nil, &u); err != nil {
		return 0, 0, err
	}
	list, err := c.Instances(ctx)
	if err != nil {
		return u.Credit, 0, err
	}
	for _, in := range list {
		if in.ActualStatus == "running" {
			perHour += in.PerHour
		}
	}
	return u.Credit, perHour, nil
}

// Offer is one machine's offer: Price is the total per hour for one GPU with the asked disk (dph_total).
type Offer struct {
	ID          int64   `json:"id"`
	GPUName     string  `json:"gpu_name"`
	NumGPUs     int     `json:"num_gpus"`
	GPURAM      float64 `json:"gpu_ram"` // MB
	Price       float64 `json:"dph_total"`
	CPUCores    float64 `json:"cpu_cores_effective"`
	CPURAM      float64 `json:"cpu_ram"` // MB (the instance's share)
	DiskSpace   float64 `json:"disk_space"`
	InetDown    float64 `json:"inet_down"` // Mbps
	Reliability float64 `json:"reliability2"`
	Location    string  `json:"geolocation"`
	CUDAMax     float64 `json:"cuda_max_good"`
	Verified    any     `json:"verification"`
}

// OfferQuery is a search: a GPU name ("" for every type), the disk to price, and the floors that keep flaky hosts out.
type OfferQuery struct {
	GPU            string  // exact vast name ("RTX 4090"); "" = any
	DiskGB         int     // the disk the instance will ask for (offers with less space are left out)
	MinReliability float64 // e.g. 0.98
	MinInetDown    float64 // Mbps, e.g. 500
	MaxPrice       float64 // $/hr, 0 = no limit
	Limit          int
}

// Offers lists on-demand single-GPU offers from verified hosts, cheapest first.
func (c *Client) Offers(ctx context.Context, q OfferQuery) ([]Offer, error) {
	f := map[string]any{
		"verified": map[string]any{"eq": true}, "external": map[string]any{"eq": false},
		"rentable": map[string]any{"eq": true}, "rented": map[string]any{"eq": false},
		"num_gpus": map[string]any{"eq": 1},
		"order":    [][]string{{"dph_total", "asc"}}, "type": "on-demand",
		"allocated_storage": max(q.DiskGB, 10),
	}
	if q.GPU != "" {
		f["gpu_name"] = map[string]any{"eq": q.GPU}
	}
	if q.DiskGB > 0 {
		f["disk_space"] = map[string]any{"gte": q.DiskGB}
	}
	if q.MinReliability > 0 {
		f["reliability"] = map[string]any{"gte": q.MinReliability}
	}
	if q.MinInetDown > 0 {
		f["inet_down"] = map[string]any{"gte": q.MinInetDown}
	}
	if q.MaxPrice > 0 {
		f["dph_total"] = map[string]any{"lte": q.MaxPrice}
	}
	if q.Limit > 0 {
		f["limit"] = q.Limit
	}
	var r struct {
		Offers []Offer `json:"offers"`
	}
	if err := c.do(ctx, "POST", "/api/v0/bundles/", f, &r); err != nil {
		return nil, err
	}
	sort.SliceStable(r.Offers, func(i, j int) bool { return r.Offers[i].Price < r.Offers[j].Price })
	return r.Offers, nil
}

// CreateInput is an instance to create on an offer. Env holds the container's environment; Ports the container ports
// to publish on the host's public IP (TCP). Onstart runs when the container starts (in the background of vast's init:
// it reads the environment from /proc/1/environ, see remote-pod's bootstrap).
type CreateInput struct {
	Offer   int64
	Label   string
	Image   string
	DiskGB  int
	Env     map[string]string
	Ports   []int
	Onstart string
}

// Create rents the offer and returns the new instance's id. SSH stays available (vast's ssh runtype, direct + proxy).
func (c *Client) Create(ctx context.Context, in CreateInput) (int64, error) {
	env := map[string]string{}
	for k, v := range in.Env {
		env[k] = v
	}
	for _, p := range in.Ports {
		env[fmt.Sprintf("-p %d:%d", p, p)] = "1"
	}
	body := map[string]any{
		"client_id": "me", "image": in.Image, "env": env, "disk": in.DiskGB, "label": in.Label,
		"onstart": in.Onstart, "runtype": "ssh_direc ssh_proxy", "cancel_unavail": true,
	}
	var r struct {
		Success     bool   `json:"success"`
		NewContract int64  `json:"new_contract"`
		Msg         string `json:"msg"`
	}
	if err := c.do(ctx, "PUT", fmt.Sprintf("/api/v0/asks/%d/", in.Offer), body, &r); err != nil {
		return 0, err
	}
	if !r.Success || r.NewContract == 0 {
		return 0, fmt.Errorf("vast api: offer %d not rented: %s", in.Offer, r.Msg)
	}
	return r.NewContract, nil
}

// Instance is what the API reports about one of our instances.
type Instance struct {
	ID            int64                          `json:"id"`
	Label         string                         `json:"label"`
	ActualStatus  string                         `json:"actual_status"`   // loading, running, exited, ... (null: created)
	IntendedState string                         `json:"intended_status"` // running, stopped
	StatusMsg     string                         `json:"status_msg"`
	GPUName       string                         `json:"gpu_name"`
	PerHour       float64                        `json:"dph_total"`
	Image         string                         `json:"image_uuid"`
	StartDate     float64                        `json:"start_date"`
	PublicIP      string                         `json:"public_ipaddr"`
	SSHHost       string                         `json:"ssh_host"`
	SSHPort       int                            `json:"ssh_port"`
	Location      string                         `json:"geolocation"`
	Ports         map[string][]map[string]string `json:"ports"`
}

// State is a one-word state for the status table: running, loading, stopped, ...
func (in *Instance) State() string {
	if in.ActualStatus == "" {
		return "created"
	}
	if in.IntendedState == "stopped" && in.ActualStatus != "running" {
		return "stopped"
	}
	return in.ActualStatus
}

// HostPort is the public port a container TCP port is published on (0: not published yet).
func (in *Instance) HostPort(port int) int {
	for _, m := range in.Ports[fmt.Sprintf("%d/tcp", port)] {
		var p int
		if _, err := fmt.Sscan(m["HostPort"], &p); err == nil && p > 0 {
			return p
		}
	}
	return 0
}

// Uptime since the instance started.
func (in *Instance) Uptime() time.Duration {
	if in.StartDate <= 0 {
		return 0
	}
	return time.Since(time.Unix(int64(in.StartDate), 0)).Truncate(time.Second)
}

func (c *Client) Get(ctx context.Context, id int64) (*Instance, error) {
	var r struct {
		Instance *Instance `json:"instances"`
	}
	if err := c.do(ctx, "GET", fmt.Sprintf("/api/v0/instances/%d/?owner=me", id), nil, &r); err != nil {
		return nil, err
	}
	if r.Instance == nil {
		return nil, fmt.Errorf("no instance %d (not found)", id)
	}
	return r.Instance, nil
}

// Instances lists the account's instances (the v1 endpoint, paged).
func (c *Client) Instances(ctx context.Context) ([]Instance, error) {
	var all []Instance
	after := ""
	for {
		q := url.Values{"select_filters": {"{}"}, "order_by": {`[{"col": "id", "dir": "asc"}]`}, "limit": {"25"}}
		if after != "" {
			q.Set("after_token", after)
		}
		var r struct {
			Instances []Instance `json:"instances"`
			NextToken string     `json:"next_token"`
		}
		if err := c.do(ctx, "GET", "/api/v1/instances/?"+q.Encode(), nil, &r); err != nil {
			return nil, err
		}
		all = append(all, r.Instances...)
		if r.NextToken == "" || len(r.Instances) == 0 {
			return all, nil
		}
		after = r.NextToken
	}
}

// Stop stops the instance: the GPU is released and billing stops, except the disk's storage; the disk is kept, but the
// GPU may be rented by someone else before a restart.
func (c *Client) Stop(ctx context.Context, id int64) error {
	return c.do(ctx, "PUT", fmt.Sprintf("/api/v0/instances/%d/", id), map[string]any{"state": "stopped"}, nil)
}

// Start restarts a stopped instance (fails while its GPU is rented by someone else).
func (c *Client) Start(ctx context.Context, id int64) error {
	return c.do(ctx, "PUT", fmt.Sprintf("/api/v0/instances/%d/", id), map[string]any{"state": "running"}, nil)
}

// Destroy deletes the instance and its disk: all billing stops.
func (c *Client) Destroy(ctx context.Context, id int64) error {
	return c.do(ctx, "DELETE", fmt.Sprintf("/api/v0/instances/%d/", id), map[string]any{}, nil)
}

// AttachSSH adds a public key to one instance's authorized keys and returns the id vast.ai gives it ("" when the
// answer has none: the key then stays until the instance is destroyed, harmless once its private half is deleted).
func (c *Client) AttachSSH(ctx context.Context, id int64, publicKey string) (string, error) {
	var r map[string]any
	if err := c.do(ctx, "POST", fmt.Sprintf("/api/v0/instances/%d/ssh/", id), map[string]any{"ssh_key": publicKey}, &r); err != nil {
		return "", err
	}
	return findID(r), nil
}

// DetachSSH removes a key attached with AttachSSH.
func (c *Client) DetachSSH(ctx context.Context, id int64, keyID string) error {
	return c.do(ctx, "DELETE", fmt.Sprintf("/api/v0/instances/%d/ssh/%s/", id, url.PathEscape(keyID)), map[string]any{}, nil)
}

// findID looks for the key's id in an answer whose exact shape the API does not document.
func findID(v any) string {
	switch x := v.(type) {
	case map[string]any:
		for _, k := range []string{"ssh_key_id", "key_id", "id"} {
			switch id := x[k].(type) {
			case float64:
				return fmt.Sprint(int64(id))
			case string:
				if id != "" {
					return id
				}
			}
		}
		for _, e := range x {
			if id := findID(e); id != "" {
				return id
			}
		}
	}
	return ""
}
