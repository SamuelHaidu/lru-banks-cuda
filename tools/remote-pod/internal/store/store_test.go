package store_test

// Tests against an in-memory S3 server (gofakes3): the same API R2 speaks, no network or credentials.

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/johannesboyne/gofakes3"
	"github.com/johannesboyne/gofakes3/backend/s3mem"

	"github.com/SamuelHaidu/rnn-research/tools/remote-pod/internal/agent"
	"github.com/SamuelHaidu/rnn-research/tools/remote-pod/internal/store"
)

func testStore(t *testing.T) *store.Store {
	t.Helper()
	backend := s3mem.New()
	bucket := fmt.Sprintf("rp-test-%d", time.Now().UnixNano())
	if err := backend.CreateBucket(bucket); err != nil {
		t.Fatal(err)
	}
	ts := httptest.NewServer(gofakes3.New(backend).Server())
	t.Cleanup(ts.Close)
	st, err := store.New(store.Config{Endpoint: ts.URL, Bucket: bucket, AccessKey: "test", SecretKey: "testtesttest"})
	if err != nil {
		t.Fatal(err)
	}
	return st
}

func TestDatasetRoundTrip(t *testing.T) {
	st, ctx := testStore(t), context.Background()
	src := t.TempDir()
	os.WriteFile(filepath.Join(src, "train.npy"), []byte(strings.Repeat("tokens", 200000)), 0o644)
	os.WriteFile(filepath.Join(src, "metadata.json"), []byte(`{"name": "d"}`), 0o644)
	size, err := st.PutDir(ctx, store.DatasetKey("d"), src, nil)
	if err != nil {
		t.Fatal(err)
	}
	if size >= 1200000 {
		t.Fatalf("not compressed: %d bytes", size)
	}
	if err := st.PutDatasetManifest(ctx, store.Dataset{Name: "d", Key: store.DatasetKey("d"), Files: 2}); err != nil {
		t.Fatal(err)
	}
	ds, err := st.Datasets(ctx)
	if err != nil || len(ds) != 1 || ds[0].Name != "d" {
		t.Fatalf("list: %v %v", err, ds)
	}
	dst := t.TempDir()
	if files, _, err := st.GetDir(ctx, store.DatasetKey("d"), dst, nil); err != nil || files != 2 {
		t.Fatalf("get: %v %d", err, files)
	}
	b, _ := os.ReadFile(filepath.Join(dst, "train.npy"))
	if len(b) != 1200000 {
		t.Fatalf("content %d bytes", len(b))
	}
}

// The whole train flow with a fake training command: datasets and the init checkpoint fetched, checkpoints and the
// log uploaded as they appear, superseded numbered checkpoints pruned on the pod.
func TestTrainRunFetchesRunsAndUploads(t *testing.T) {
	st, ctx := testStore(t), context.Background()
	data := t.TempDir()
	os.WriteFile(filepath.Join(data, "train.npy"), []byte("x"), 0o644)
	if _, err := st.PutDir(ctx, store.DatasetKey("mix"), data, nil); err != nil {
		t.Fatal(err)
	}
	st.PutDatasetManifest(ctx, store.Dataset{Name: "mix", Key: store.DatasetKey("mix")})
	st.PutBytes(ctx, "runs/base/model.lbc", []byte("init weights"))

	work := t.TempDir()
	srv, err := agent.New(agent.Options{Token: "t", Workdir: work})
	if err != nil {
		t.Fatal(err)
	}
	cfg, _ := json.Marshal(map[string]any{"data": map[string]any{"dataset": filepath.Join(work, "data", "mix")}})
	// a fake trainer: checks its inputs, writes three numbered checkpoints and the final model
	cmd := `test -f ` + work + `/data/mix/train.npy && test -f runs/base/model.lbc && mkdir -p checkpoints &&
	  for e in 1 2 3; do echo "epoch $e val_loss 1.$e"; echo ckpt$e > checkpoints/checkpoint_0000$e.lbc; sleep 0.1; done &&
	  echo final > model.lbc`
	spec := agent.TrainSpec{Run: "r1", Command: "cd " + work + " && cd runs/r1 && " + strings.ReplaceAll(cmd, "runs/base", work+"/runs/base"),
		Datasets: []string{"mix"}, Fetch: map[string]string{"runs/base/model.lbc": "runs/base/model.lbc"},
		Files: map[string]string{"runs/r1/config.json": string(cfg)}, RunDir: "runs/r1", Prefix: "runs/r1/",
		DataDir: filepath.Join(work, "data")}
	if err := srv.RunTrain(ctx, st, spec); err != nil {
		t.Fatal(err)
	}
	objs, _ := st.List(ctx, "runs/r1/", true)
	keys := map[string]bool{}
	for _, o := range objs {
		keys[strings.TrimPrefix(o.Key, "runs/r1/")] = true
	}
	for _, want := range []string{"config.json", "model.lbc", "train.log", "checkpoints/checkpoint_00003.lbc"} {
		if !keys[want] {
			t.Fatalf("%s not uploaded: %v", want, keys)
		}
	}
	log, _ := st.GetBytes(ctx, "runs/r1/train.log")
	if !strings.Contains(string(log), "epoch 3 val_loss 1.3") {
		t.Fatalf("log %q", log)
	}
}
