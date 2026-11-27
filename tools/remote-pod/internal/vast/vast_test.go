package vast

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

// The requests the client sends (paths, auth, the create payload) and how it reads the answers.
func TestFakeAPI(t *testing.T) {
	var created map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer k" {
			http.Error(w, `{"msg": "bad key"}`, http.StatusUnauthorized)
			return
		}
		switch r.Method + " " + r.URL.Path {
		case "POST /api/v0/bundles/":
			var q map[string]any
			_ = json.NewDecoder(r.Body).Decode(&q)
			if q["gpu_name"].(map[string]any)["eq"] != "RTX 4090" || q["type"] != "on-demand" {
				t.Errorf("offer query: %v", q)
			}
			_, _ = w.Write([]byte(`{"offers": [{"id": 2, "gpu_name": "RTX 4090", "dph_total": 0.5},
				{"id": 1, "gpu_name": "RTX 4090", "dph_total": 0.4}]}`))
		case "PUT /api/v0/asks/1/":
			_ = json.NewDecoder(r.Body).Decode(&created)
			_, _ = w.Write([]byte(`{"success": true, "new_contract": 77}`))
		case "GET /api/v0/instances/77/":
			_, _ = w.Write([]byte(`{"instances": {"id": 77, "actual_status": "running", "intended_status": "running",
				"public_ipaddr": "1.2.3.4", "dph_total": 0.41,
				"ports": {"8080/tcp": [{"HostIp": "0.0.0.0", "HostPort": "40123"}]}}}`))
		case "GET /api/v0/users/current":
			_, _ = w.Write([]byte(`{"credit": 5.5}`))
		case "GET /api/v1/instances/":
			_, _ = w.Write([]byte(`{"instances": [{"id": 77, "actual_status": "running", "dph_total": 0.41}], "next_token": ""}`))
		case "PUT /api/v0/instances/77/", "DELETE /api/v0/instances/77/":
			_, _ = w.Write([]byte(`{"success": true}`))
		default:
			t.Errorf("unexpected %s %s", r.Method, r.URL.Path)
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()
	c := New("k")
	c.URL = srv.URL
	ctx := context.Background()

	offers, err := c.Offers(ctx, OfferQuery{GPU: "RTX 4090", DiskGB: 40})
	if err != nil || len(offers) != 2 || offers[0].ID != 1 {
		t.Fatalf("offers %v %v (want the cheapest first)", offers, err)
	}
	id, err := c.Create(ctx, CreateInput{Offer: 1, Image: "img", DiskGB: 40, Env: map[string]string{"A": "b"},
		Ports: []int{8080}, Onstart: "start"})
	if err != nil || id != 77 {
		t.Fatalf("create %d %v", id, err)
	}
	env := created["env"].(map[string]any)
	if env["A"] != "b" || env["-p 8080:8080"] != "1" || created["onstart"] != "start" ||
		created["runtype"] != "ssh_direc ssh_proxy" || created["image"] != "img" {
		t.Fatalf("create payload %v", created)
	}
	in, err := c.Get(ctx, 77)
	if err != nil || in.HostPort(8080) != 40123 || in.HostPort(22) != 0 || in.State() != "running" {
		t.Fatalf("get %+v %v", in, err)
	}
	credit, perHr, err := c.Balance(ctx)
	if err != nil || credit != 5.5 || perHr != 0.41 {
		t.Fatalf("balance %v %v %v", credit, perHr, err)
	}
	if err := c.Stop(ctx, 77); err != nil {
		t.Fatal(err)
	}
	if err := c.Destroy(ctx, 77); err != nil {
		t.Fatal(err)
	}
	c.Key = "wrong"
	if _, _, err := c.Balance(ctx); err == nil {
		t.Fatal("a refused key was not an error")
	}
}
