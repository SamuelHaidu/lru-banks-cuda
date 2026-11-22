package runpod

// A live contract test against RunPod's API: free and harmless. RunPod disables introspection, so each request is
// sent with every field this client uses plus one deliberately mistyped value: GraphQL rejects it while checking the
// variables, before anything runs (no pod is created, stopped or deleted), and reports every unknown field there too.
// So the only error expected is the mistyped one. Skipped unless REMOTE_POD_LIVE=1:
//
//	REMOTE_POD_LIVE=1 go test ./internal/runpod/ -run Live -v

import (
	"context"
	"os"
	"strings"
	"testing"

	"github.com/SamuelHaidu/rnn-research/tools/remote-pod/internal/config"
)

func TestLiveRequestsUseKnownFields(t *testing.T) {
	if os.Getenv("REMOTE_POD_LIVE") != "1" {
		t.Skip("REMOTE_POD_LIVE=1 to check against the live RunPod API")
	}
	cfg, err := config.Load()
	if err != nil {
		t.Fatal(err)
	}
	key, err := cfg.RunPodKey()
	if err != nil {
		t.Skip(err)
	}
	c := New(key)
	cases := []struct {
		name, query, mistyped string
		input                 map[string]any
	}{
		{"create", `mutation($input: PodFindAndDeployOnDemandInput) { podFindAndDeployOnDemand(input: $input) { id } }`,
			"gpuCount", map[string]any{"cloudType": "COMMUNITY", "gpuCount": "not-a-number", "gpuTypeId": "x",
				"name": "contract-test", "imageName": "x", "containerDiskInGb": 1, "volumeInGb": 0, "ports": "8080/http",
				"env": []map[string]string{{"key": "A", "value": "1"}}, "volumeMountPath": "/workspace"}},
		{"create cpu", `mutation($input: deployCpuPodInput!) { deployCpuPod(input: $input) { id } }`,
			"containerDiskInGb", map[string]any{"instanceId": "cpu5g-8-32", "cloudType": "SECURE",
				"containerDiskInGb": "not-a-number", "imageName": "x", "name": "contract-test", "ports": "8080/http",
				"env": []map[string]string{{"key": "A", "value": "1"}}}},
		{"stop", `mutation($input: PodStopInput!) { podStop(input: $input) { id } }`,
			"podId", map[string]any{"podId": 123}},
		{"terminate", `mutation($input: PodTerminateInput!) { podTerminate(input: $input) }`,
			"podId", map[string]any{"podId": 123}},
		{"get", `query($input: PodFilter) { pod(input: $input) { id } }`,
			"podId", map[string]any{"podId": 123}},
	}
	for _, tc := range cases {
		err := c.Do(context.Background(), tc.query, map[string]any{"input": tc.input}, nil)
		if err == nil {
			t.Fatalf("%s: the mistyped value was accepted (the test must never get past validation)", tc.name)
		}
		msg := err.Error()
		if strings.Contains(msg, "is not defined by type") || strings.Contains(msg, "Unknown type") {
			t.Errorf("%s: a field or type this client uses is unknown: %s", tc.name, msg)
		}
		if !strings.Contains(msg, tc.mistyped) {
			t.Errorf("%s: expected only the mistyped %q to be rejected, got: %s", tc.name, tc.mistyped, msg)
		}
		t.Logf("%s: %s", tc.name, msg)
	}
}
