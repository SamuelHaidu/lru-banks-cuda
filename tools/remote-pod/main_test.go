package main

import (
	"bytes"
	"strings"
	"testing"
)

func TestPlanTrainRewritesPaths(t *testing.T) {
	cfg := `{"data": {"dataset": "/root/data/pretrain-mix", "mix_datasets": ["/root/data/cx"]},
	         "training": {"init_from": "/workspace/runs/sft/checkpoints/best.lbc"}}`
	spec, err := planTrain("rl", []byte(cfg))
	if err != nil {
		t.Fatal(err)
	}
	if len(spec.Datasets) != 2 || spec.Fetch["runs/sft/checkpoints/best.lbc"] != "/work/runs/sft/checkpoints/best.lbc" {
		t.Fatalf("spec %+v", spec)
	}
	c := spec.Files["/work/runs/rl/config.json"]
	if !strings.Contains(c, `"/data/pretrain-mix"`) || !strings.Contains(c, `"/data/cx"`) || !strings.Contains(c, `"/work/runs/sft/checkpoints/best.lbc"`) {
		t.Fatalf("config %s", c)
	}
}

func TestEpochWriterKeepsEpochLines(t *testing.T) {
	var out bytes.Buffer
	w := &epochWriter{w: &out}
	w.Write([]byte("Epoch 1/2\n 1/50 \x1b[32m━━\x1b[0m loss: 2.1\r 2/50 loss: 2.0\r50/50 loss: 1.9 - val_loss: 1.8\n"))
	w.Write([]byte("noise\nEpoch 2/2\n"))
	if out.String() != "Epoch 1/2\n50/50 loss: 1.9 - val_loss: 1.8\nEpoch 2/2\n" {
		t.Fatalf("%q", out.String())
	}
}
