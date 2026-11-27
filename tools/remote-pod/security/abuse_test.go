package security

// The protocol a pod's requests must follow beyond its policy (grant.Limits, grant.UploadPlan): a compromised pod
// asking for things it is allowed, but too much, too fast, or not the way its run would, is refused, and one that
// keeps being refused is suspended. An honest run must never be.

import (
	"context"
	"errors"
	"fmt"
	"net/url"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/SamuelHaidu/rnn-research/tools/remote-pod/internal/grant"
	"github.com/SamuelHaidu/rnn-research/tools/remote-pod/internal/store"
)

func put(key string, size int64) grant.Request {
	return grant.Request{Op: grant.Put, Key: key, Size: size}
}

func signer(t *testing.T, l grant.Limits, plans ...grant.UploadPlan) *grant.Signer {
	p := testPolicy(t)
	p.Plans = plans
	return &grant.Signer{Policy: p, Store: &fakePresigner{}, Limits: &l}
}

func loose() grant.Limits {
	return grant.Limits{PerMinute: 1 << 20, PerHour: 1 << 20, PerKeyPerHour: 1 << 20, MaxObjects: 1 << 20,
		MaxFileBytes: 1 << 40, UploadBytesPerDay: 1 << 50, SuspendAfter: 1 << 20, SuspendWindow: time.Minute,
		SuspendFor: time.Minute}
}

func mustRefuse(t *testing.T, s *grant.Signer, r grant.Request, why string) {
	t.Helper()
	if _, err := s.Sign(context.Background(), r); !errors.Is(err, grant.ErrRefused) {
		t.Errorf("%s: %s %s (%d bytes) not refused: %v", why, r.Op, r.Key, r.Size, err)
	}
}

func mustGrant(t *testing.T, s *grant.Signer, r grant.Request, why string) {
	t.Helper()
	if _, err := s.Sign(context.Background(), r); err != nil {
		t.Errorf("%s: %s %s (%d bytes) refused: %v", why, r.Op, r.Key, r.Size, err)
	}
}

func TestUploadsDeclareABoundedSize(t *testing.T) {
	l := loose()
	l.MaxFileBytes = 1 << 30
	s := signer(t, l)
	mustRefuse(t, s, put("runs/my-run/model.lbc", 0), "no size")
	mustRefuse(t, s, put("runs/my-run/model.lbc", -5), "negative size")
	mustRefuse(t, s, put("runs/my-run/model.lbc", 1<<30+1), "over the per-file cap")
	mustRefuse(t, s, grant.Request{Op: grant.Get, Key: "datasets/fw/data.tar.zst", Size: 7}, "a download with a size")
	mustGrant(t, s, put("runs/my-run/model.lbc", 1<<30), "at the cap")
}

func TestDailyUploadBudget(t *testing.T) {
	l := loose()
	l.UploadBytesPerDay = 1000
	s := signer(t, l)
	mustGrant(t, s, put("runs/my-run/a", 600), "within budget")
	mustRefuse(t, s, put("runs/my-run/b", 600), "over the 24-hour budget")
	mustGrant(t, s, put("runs/my-run/c", 400), "exactly the rest")
}

func TestObjectFlood(t *testing.T) {
	l := loose()
	l.MaxObjects = 5
	s := signer(t, l)
	for i := 0; i < 5; i++ {
		mustGrant(t, s, put(fmt.Sprintf("runs/my-run/f%d", i), 10), "distinct objects up to the cap")
	}
	mustRefuse(t, s, put("runs/my-run/f5", 10), "a 6th distinct object")
	mustGrant(t, s, put("runs/my-run/f0", 10), "rewriting an existing object")
}

func TestHammeringOneObject(t *testing.T) {
	l := loose()
	l.PerKeyPerHour = 5
	s := signer(t, l)
	for i := 0; i < 5; i++ {
		mustGrant(t, s, put("runs/my-run/metrics.csv", 10), "rewrites up to the cap")
	}
	mustRefuse(t, s, put("runs/my-run/metrics.csv", 10), "the 6th grant for one object in an hour")
	mustGrant(t, s, put("runs/my-run/train.log", 10), "another object")
}

func TestRateLimits(t *testing.T) {
	l := loose()
	l.PerMinute = 10
	s := signer(t, l)
	for i := 0; i < 10; i++ {
		mustGrant(t, s, put(fmt.Sprintf("runs/my-run/x%d", i), 1), "within the rate")
	}
	mustRefuse(t, s, put("runs/my-run/x10", 1), "the 11th grant in a minute")
	l = loose()
	l.PerHour = 10
	s = signer(t, l)
	for i := 0; i < 10; i++ {
		mustGrant(t, s, put(fmt.Sprintf("runs/my-run/y%d", i), 1), "within the hourly rate")
	}
	mustRefuse(t, s, put("runs/my-run/y10", 1), "the 11th grant in an hour")
}

// A pod that keeps asking for what it may not have is suspended: even its allowed requests are refused for a while,
// and every decision is journaled.
func TestRepeatedRefusalsSuspendThePod(t *testing.T) {
	l := loose()
	l.SuspendAfter, l.SuspendWindow, l.SuspendFor = 5, time.Minute, 300*time.Millisecond
	s := signer(t, l)
	j := &grant.FileJournal{Path: filepath.Join(t.TempDir(), "grants.jsonl")}
	s.Journal = j
	for i := 0; i < 5; i++ {
		mustRefuse(t, s, put(fmt.Sprintf("runs/other-run/%d", i), 1), "outside the policy")
	}
	if s.Suspended().IsZero() {
		t.Fatal("not suspended after 5 refusals")
	}
	_, err := s.Sign(context.Background(), put("runs/my-run/model.lbc", 1))
	if !errors.Is(err, grant.ErrSuspended) {
		t.Fatalf("an allowed request while suspended: %v", err)
	}
	time.Sleep(400 * time.Millisecond)
	mustGrant(t, s, put("runs/my-run/model.lbc", 1), "after the suspension")
	es, err := j.Load()
	if err != nil || len(es) != 7 {
		t.Fatalf("journal: %d entries, %v (want 7: 5 refused, 1 suspended, 1 granted)", len(es), err)
	}
	if !strings.Contains(es[4].Reason, "suspended until") || es[6].OK != true {
		t.Errorf("journal does not record the suspension: %+v", es)
	}
}

// Restarting the signer does not reset a pod's budgets: they are rebuilt from the journal.
func TestBudgetsSurviveASignerRestart(t *testing.T) {
	l := loose()
	l.UploadBytesPerDay, l.MaxObjects = 1000, 3
	j := &grant.FileJournal{Path: filepath.Join(t.TempDir(), "grants.jsonl")}
	s1 := signer(t, l)
	s1.Journal = j
	mustGrant(t, s1, put("runs/my-run/a", 900), "first signer")
	mustGrant(t, s1, put("runs/my-run/b", 50), "first signer")
	s2 := signer(t, l)
	s2.Journal = j
	mustRefuse(t, s2, put("runs/my-run/c", 100), "the budget, after a restart")
	mustGrant(t, s2, put("runs/my-run/c", 50), "the rest of the budget")
	mustRefuse(t, s2, put("runs/my-run/d", 1), "the object cap, after a restart")
}

const ckpt = 541 << 20 // our 47M model's checkpoint

// The run's plan, from its config: 10 epochs, a numbered checkpoint every 2, 100M tokens an epoch.
func testPlan() grant.UploadPlan { return grant.RunUploadPlan("runs/my-run/", 10, 2, 1e8) }

func TestRunPlanBoundsCheckpoints(t *testing.T) {
	p := testPlan()
	if p.MinRewrite < 30*time.Second || p.MaxObjects != 10/2+1+len(grant.RunFixedFiles)+8 {
		t.Fatalf("plan %+v", p)
	}
	s := signer(t, loose(), p)
	mustGrant(t, s, put("runs/my-run/model.lbc", ckpt), "the first checkpoint")
	mustRefuse(t, s, put("runs/my-run/model.lbc", ckpt), "a checkpoint rewritten faster than any GPU trains an epoch")
	mustGrant(t, s, put("runs/my-run/metrics.csv", 4096), "a small file is not paced like a checkpoint")
	mustGrant(t, s, put("runs/my-run/metrics.csv", 4096), "a small file is not paced like a checkpoint")
	// a flood of new checkpoint names: more than the plan's hourly allowance is refused
	refused := 0
	for i := 0; i < 2*p.MaxNewHeavyPerHour; i++ {
		if _, err := s.Sign(context.Background(), put(fmt.Sprintf("runs/my-run/checkpoints/checkpoint_%05d.lbc", i), ckpt)); err != nil {
			refused++
		}
	}
	if refused == 0 {
		t.Errorf("%d new checkpoints in an hour were all granted (plan: %d)", 2*p.MaxNewHeavyPerHour, p.MaxNewHeavyPerHour)
	}
	// and more distinct files than the run would ever write
	s = signer(t, loose(), p)
	for i := 0; i < p.MaxObjects; i++ {
		mustGrant(t, s, put(fmt.Sprintf("runs/my-run/small-%d", i), 10), "files up to the plan")
	}
	mustRefuse(t, s, put("runs/my-run/one-too-many", 10), "more files than the run would write")
}

// memJournal is a journal with entries made up front: a run's past, to put the signer in the middle of it.
type memJournal struct{ es []grant.Entry }

func (m *memJournal) Append(e grant.Entry) error { m.es = append(m.es, e); return nil }
func (m *memJournal) Load() ([]grant.Entry, error) {
	return append([]grant.Entry(nil), m.es...), nil
}

// An honest run, at our real pace (7.6-minute epochs, a numbered checkpoint every 10, logs every minute), passes
// every check at every step: the limits must never stop the work they protect.
func TestAnHonestRunIsNeverRefused(t *testing.T) {
	const epochs, every = 95, 10
	plan := grant.RunUploadPlan("runs/my-run/", epochs, every, 99942400)
	p := testPolicy(t)
	p.Plans = []grant.UploadPlan{plan}
	j := &memJournal{}
	s := &grant.Signer{Policy: p, Store: &fakePresigner{}, Journal: j} // the default limits
	// replay the last 3 hours of the run into the journal (as if the signer had been answering all along), then ask
	// for what comes next: each step must be granted
	start := time.Now().Add(-3 * time.Hour)
	epoch := 456 * time.Second
	for m := time.Duration(0); m < 3*time.Hour; m += time.Minute {
		at := start.Add(m)
		j.es = append(j.es, grant.Entry{Time: at, Op: grant.Put, Key: "runs/my-run/train.log", Size: 1 << 20, OK: true},
			grant.Entry{Time: at, Op: grant.Put, Key: grant.JobLogPrefix(tag) + "job1.log", Size: 1 << 20, OK: true})
	}
	n := 0
	for e := time.Duration(0); e < 3*time.Hour; e += epoch {
		at := start.Add(e)
		n++
		j.es = append(j.es, grant.Entry{Time: at, Op: grant.Put, Key: "runs/my-run/model.lbc", Size: ckpt, OK: true},
			grant.Entry{Time: at, Op: grant.Put, Key: "runs/my-run/checkpoints/best.lbc", Size: ckpt, OK: true},
			grant.Entry{Time: at, Op: grant.Put, Key: "runs/my-run/metrics.csv", Size: 8 << 10, OK: true})
		if n%every == 0 {
			j.es = append(j.es, grant.Entry{Time: at, Op: grant.Put, Size: ckpt, OK: true,
				Key: fmt.Sprintf("runs/my-run/checkpoints/checkpoint_%05d.lbc", n)})
		}
	}
	for _, r := range []grant.Request{
		put("runs/my-run/train.log", 2<<20),
		put(grant.JobLogPrefix(tag)+"job1.log", 2<<20),
		put("runs/my-run/model.lbc", ckpt),
		put("runs/my-run/checkpoints/best.lbc", ckpt),
		put("runs/my-run/metrics.csv", 9<<10),
		put(fmt.Sprintf("runs/my-run/checkpoints/checkpoint_%05d.lbc", n+every), ckpt),
		{Op: grant.Get, Key: "datasets/fw/data.tar.zst"},
	} {
		mustGrant(t, s, r, "the honest run's next step")
	}
}

// A sized upload URL signs its Content-Length (so the store refuses any other length: live_r2_test.go shows it).
func TestSizedUploadURLSignsTheLength(t *testing.T) {
	st, err := store.New(store.Config{Endpoint: "https://example.r2.cloudflarestorage.com", Bucket: "b-test",
		AccessKey: "AK", SecretKey: "SK"})
	if err != nil {
		t.Fatal(err)
	}
	u, err := st.PresignPut(context.Background(), "runs/my-run/model.lbc", time.Minute, 12345)
	if err != nil {
		t.Fatal(err)
	}
	q, _ := url.Parse(u)
	if h := q.Query().Get("X-Amz-SignedHeaders"); !strings.Contains(h, "content-length") {
		t.Fatalf("signed headers %q: the length is not part of the signature", h)
	}
	if strings.Contains(u, "SK") && strings.Contains(u, "=SK") {
		t.Fatal("the URL carries the secret")
	}
}
