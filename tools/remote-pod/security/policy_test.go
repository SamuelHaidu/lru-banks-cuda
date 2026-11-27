package security

import (
	"context"
	"errors"
	"path"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/SamuelHaidu/rnn-research/tools/remote-pod/internal/grant"
)

// fakePresigner records every key it was asked to sign: the evidence that a refused request never got that far.
type fakePresigner struct {
	mu    sync.Mutex
	calls []string
}

func (f *fakePresigner) PresignGet(_ context.Context, key string, ttl time.Duration) (string, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.calls = append(f.calls, "get "+key)
	return "https://store.example/" + key + "?X-Amz-Expires=" + ttl.String() + "&op=get", nil
}

func (f *fakePresigner) PresignPut(_ context.Context, key string, ttl time.Duration, size int64) (string, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.calls = append(f.calls, "put "+key)
	return "https://store.example/" + key + "?X-Amz-Expires=" + ttl.String() + "&op=put", nil
}

func (f *fakePresigner) signed() []string {
	f.mu.Lock()
	defer f.mu.Unlock()
	return append([]string(nil), f.calls...)
}

const tag = "0123456789abcdef"

// testPolicy is a training pod's: its run's prefix, two inputs, its job logs.
func testPolicy(t testing.TB) grant.Policy {
	t.Helper()
	p, err := grant.PodPolicy(tag, []string{"runs/my-run/"}, []string{"datasets/fw/data.tar.zst", "runs/base/model.lbc"})
	if err != nil {
		t.Fatal(err)
	}
	return p
}

var allowed = []grant.Request{
	{Op: grant.Put, Key: "runs/my-run/model.lbc", Size: 1000},
	{Op: grant.Put, Key: "runs/my-run/checkpoints/checkpoint_00010.lbc", Size: 1000},
	{Op: grant.Put, Key: "runs/my-run/metrics.csv", Size: 1000},
	{Op: grant.Put, Key: "jobs/" + tag + "/a1b2c3.log", Size: 1000},
	{Op: grant.Get, Key: "datasets/fw/data.tar.zst"},
	{Op: grant.Get, Key: "runs/base/model.lbc"},
}

// hostile are requests a compromised pod might try; every one must be refused.
var hostile = map[string]grant.Request{
	"another run's checkpoint":        {Op: grant.Put, Key: "runs/other-run/model.lbc"},
	"prefix confusion":                {Op: grant.Put, Key: "runs/my-run-evil/model.lbc"},
	"the prefix itself":               {Op: grant.Put, Key: "runs/my-run/"},
	"the prefix without slash":        {Op: grant.Put, Key: "runs/my-run"},
	"dot-dot out of the run":          {Op: grant.Put, Key: "runs/my-run/../other-run/model.lbc"},
	"dot-dot at the end":              {Op: grant.Put, Key: "runs/my-run/.."},
	"dot segment":                     {Op: grant.Put, Key: "runs/my-run/./model.lbc"},
	"double slash":                    {Op: grant.Put, Key: "runs/my-run//model.lbc"},
	"absolute key":                    {Op: grant.Put, Key: "/runs/my-run/model.lbc"},
	"backslash traversal":             {Op: grant.Put, Key: `runs/my-run/..\..\bin\remote-pod`},
	"percent-encoded dot-dot":         {Op: grant.Put, Key: "runs/my-run/%2e%2e/other-run/model.lbc"},
	"percent-encoded slash":           {Op: grant.Put, Key: "runs/my-run%2f..%2fother"},
	"NUL byte":                        {Op: grant.Put, Key: "runs/my-run/model.lbc\x00.txt"},
	"newline":                         {Op: grant.Put, Key: "runs/my-run/model.lbc\nX-Injected: 1"},
	"unicode division slash":          {Op: grant.Put, Key: "runs/my-run∕..∕other-run∕model.lbc"},
	"fullwidth dots":                  {Op: grant.Put, Key: "runs/my-run/．．/other-run/model.lbc"},
	"space":                           {Op: grant.Put, Key: "runs/my-run/a b"},
	"key too long":                    {Op: grant.Put, Key: "runs/my-run/" + strings.Repeat("a", 600)},
	"empty key":                       {Op: grant.Put, Key: ""},
	"upper-case prefix":               {Op: grant.Put, Key: "RUNS/my-run/model.lbc"},
	"replace the agent binary":        {Op: grant.Put, Key: "bin/remote-pod-0123456789abcdef"},
	"replace lbc":                     {Op: grant.Put, Key: "bin/lbc-0123456789abcdef"},
	"poison a dataset":                {Op: grant.Put, Key: "datasets/fw/data.tar.zst"},
	"poison a dataset manifest":       {Op: grant.Put, Key: "datasets/fw/manifest.json"},
	"another pod's job log":           {Op: grant.Put, Key: "jobs/fedcba9876543210/x.log"},
	"old flat job log":                {Op: grant.Put, Key: "jobs/x.log"},
	"read another run":                {Op: grant.Get, Key: "runs/other-run/model.lbc"},
	"read its own outputs":            {Op: grant.Get, Key: "runs/my-run/model.lbc"},
	"read a dataset it was not given": {Op: grant.Get, Key: "datasets/private/data.tar.zst"},
	"read an input plus a suffix":     {Op: grant.Get, Key: "datasets/fw/data.tar.zst.bak"},
	"read an input's parent":          {Op: grant.Get, Key: "datasets/fw"},
	"list":                            {Op: "list", Key: "runs"},
	"delete":                          {Op: "delete", Key: "runs/my-run/model.lbc"},
	"head":                            {Op: "head", Key: "runs/my-run/model.lbc"},
	"upper-case op":                   {Op: "PUT", Key: "runs/my-run/model.lbc"},
	"empty op":                        {Op: "", Key: "runs/my-run/model.lbc"},
}

func TestPolicyAllowsTheRunsOwnFiles(t *testing.T) {
	p := testPolicy(t)
	for _, r := range allowed {
		if err := p.Check(r); err != nil {
			t.Errorf("%s %s refused: %v", r.Op, r.Key, err)
		}
	}
}

func TestPolicyRefusesHostileRequests(t *testing.T) {
	p := testPolicy(t)
	for name, r := range hostile {
		err := p.Check(r)
		if err == nil {
			t.Errorf("%s (%s %q) was allowed", name, r.Op, r.Key)
		} else if !errors.Is(err, grant.ErrRefused) {
			t.Errorf("%s: refused with an unexpected error type: %v", name, err)
		}
	}
}

// A refused request never reaches the presigner: no URL exists for it, even transiently.
func TestSignerNeverSignsARefusedRequest(t *testing.T) {
	fp := &fakePresigner{}
	s := &grant.Signer{Policy: testPolicy(t), Store: fp, PerMinute: 1000}
	ctx := context.Background()
	for name, r := range hostile {
		if g, err := s.Sign(ctx, r); err == nil {
			t.Errorf("%s: signed %+v", name, g)
		}
	}
	if calls := fp.signed(); len(calls) != 0 {
		t.Fatalf("the presigner was asked for %v", calls)
	}
	// that many refusals suspended this pod (abuse_test.go): an honest pod's requests go to a fresh signer
	s = &grant.Signer{Policy: testPolicy(t), Store: fp, PerMinute: 1000}
	for _, r := range allowed {
		g, err := s.Sign(ctx, r)
		if err != nil || g.Key != r.Key || g.Op != r.Op {
			t.Errorf("%s %s: %+v %v", r.Op, r.Key, g, err)
		}
		if ttl := time.Until(g.Expires); ttl > grant.DefaultTTL || ttl < grant.DefaultTTL-time.Minute {
			t.Errorf("%s: lives %s, want %s", r.Key, ttl, grant.DefaultTTL)
		}
	}
}

func TestSignerRateLimit(t *testing.T) {
	s := &grant.Signer{Policy: testPolicy(t), Store: &fakePresigner{}, PerMinute: 3}
	r := grant.Request{Op: grant.Put, Key: "runs/my-run/model.lbc", Size: 10}
	for i := 0; i < 3; i++ {
		if _, err := s.Sign(context.Background(), r); err != nil {
			t.Fatalf("grant %d refused: %v", i+1, err)
		}
	}
	if _, err := s.Sign(context.Background(), r); !errors.Is(err, grant.ErrRefused) {
		t.Fatalf("the 4th grant in a minute was not refused: %v", err)
	}
}

func TestBadPoliciesAreRejected(t *testing.T) {
	for name, p := range map[string]grant.Policy{
		"prefix without slash": {PutPrefixes: []string{"runs/x"}},
		"prefix with dot-dot":  {PutPrefixes: []string{"runs/../"}},
		"empty prefix (all)":   {PutPrefixes: []string{"/"}},
		"get key with dot-dot": {Get: []string{"datasets/../secret"}},
		"absolute get key":     {Get: []string{"/etc/passwd"}},
	} {
		if err := p.Validate(); err == nil {
			t.Errorf("%s: accepted", name)
		}
	}
	for name, tg := range map[string]string{"empty tag": "", "tag with slash": "a/b", "tag dot-dot": ".."} {
		if _, err := grant.PodPolicy(tg, nil, nil); err == nil {
			t.Errorf("%s: accepted", name)
		}
	}
}

// FuzzPolicyNeverGrantsOutside: whatever key a pod sends, an allowed one is a plain key (cleaned form equals itself,
// no "..", ASCII printable) and is either one of the exact inputs or strictly under one of the pod's prefixes.
func FuzzPolicyNeverGrantsOutside(f *testing.F) {
	for _, r := range allowed {
		f.Add(r.Op, r.Key)
	}
	for _, r := range hostile {
		f.Add(r.Op, r.Key)
	}
	p := testPolicy(f)
	f.Fuzz(func(t *testing.T, op, key string) {
		if p.Check(grant.Request{Op: op, Key: key}) != nil {
			return
		}
		if path.Clean(key) != key || strings.HasPrefix(key, "/") {
			t.Fatalf("allowed a key that is not plain: %q", key)
		}
		for _, seg := range strings.Split(key, "/") {
			if seg == ".." || seg == "." || seg == "" {
				t.Fatalf("allowed a key with segment %q: %q", seg, key)
			}
		}
		for _, r := range key {
			if r < 0x21 || r > 0x7e {
				t.Fatalf("allowed a key with %q: %q", r, key)
			}
		}
		switch op {
		case grant.Get:
			for _, k := range p.Get {
				if key == k {
					return
				}
			}
			t.Fatalf("allowed a download of %q, not an input", key)
		case grant.Put:
			for _, pre := range p.PutPrefixes {
				if strings.HasPrefix(key, pre) && len(key) > len(pre) {
					return
				}
			}
			t.Fatalf("allowed an upload to %q, outside %v", key, p.PutPrefixes)
		default:
			t.Fatalf("allowed operation %q", op)
		}
	})
}
