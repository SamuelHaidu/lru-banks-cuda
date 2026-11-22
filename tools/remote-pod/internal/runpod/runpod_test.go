package runpod

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func fake(t *testing.T, reply string, seen *map[string]any) *Client {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer k" {
			t.Errorf("auth header %q", r.Header.Get("Authorization"))
		}
		b, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(b, seen)
		io.WriteString(w, reply)
	}))
	t.Cleanup(ts.Close)
	c := New("k")
	c.URL = ts.URL
	return c
}

func TestCreateSendsTheInput(t *testing.T) {
	var seen map[string]any
	c := fake(t, `{"data":{"podFindAndDeployOnDemand":{"id":"abc"}}}`, &seen)
	id, err := c.Create(context.Background(), CreateInput{Name: "n", Image: "img", GPUTypeID: "NVIDIA GeForce RTX 4090",
		Cloud: "COMMUNITY", DiskGB: 40, Env: map[string]string{"B": "2", "A": "1"}, Ports: "8080/http"})
	if err != nil || id != "abc" {
		t.Fatalf("%v %s", err, id)
	}
	in := seen["variables"].(map[string]any)["input"].(map[string]any)
	if in["imageName"] != "img" || in["ports"] != "8080/http" || in["cloudType"] != "COMMUNITY" {
		t.Fatalf("input %v", in)
	}
	env := in["env"].([]any)
	if env[0].(map[string]any)["key"] != "A" {
		t.Fatalf("env not sorted: %v", env)
	}
}

func TestErrorsAreReported(t *testing.T) {
	var seen map[string]any
	c := fake(t, `{"errors":[{"message":"There are no longer any instances available"}]}`, &seen)
	if _, err := c.Create(context.Background(), CreateInput{}); err == nil || !strings.Contains(err.Error(), "no longer any instances") {
		t.Fatalf("err %v", err)
	}
}

func TestGPUsInStockFirst(t *testing.T) {
	var seen map[string]any
	c := fake(t, `{"data":{"gpuTypes":[
	  {"id":"A","lowestPrice":{"uninterruptablePrice":0,"stockStatus":""}},
	  {"id":"B","lowestPrice":{"uninterruptablePrice":0.9,"stockStatus":"High"}},
	  {"id":"C","lowestPrice":{"uninterruptablePrice":0.3,"stockStatus":"Low"}}]}}`, &seen)
	g, err := c.GPUs(context.Background())
	if err != nil || g[0].ID != "C" || g[1].ID != "B" || g[2].ID != "A" {
		t.Fatalf("%v %v", err, g)
	}
}
