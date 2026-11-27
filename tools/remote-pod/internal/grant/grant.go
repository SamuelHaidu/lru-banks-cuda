// Package grant is how a pod reaches the object store without holding a key. The store's key stays on this machine;
// a pod gets short-lived presigned URLs, each for one object and one operation, and only for what its Policy allows.
//
//	pod process --127.0.0.1--> agent --(the agent's TLS websocket, opened by the CLI)--> CLI: Policy.Check, then sign
//	pod process <-- URL ------- agent <------------------------------------------------- CLI
//	pod process --HTTPS PUT/GET with the URL--> R2   (the URL carries a signature for that key, method and expiry)
//
// A pod's host is assumed hostile: whatever runs on the pod can ask for grants, so the Policy is the boundary. It
// allows downloads of exact keys (the run's datasets and inputs) and uploads under exact prefixes (runs/<run>/,
// jobs/<tag>/): never a listing, a delete, another run's files, the datasets or the binaries.
package grant

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"sync"
	"time"
	"unicode"
)

// Ops a grant can be for. There is deliberately no list or delete.
const (
	Get = "get"
	Put = "put"
)

// Request is what a pod asks for. An upload declares its Size: the URL is signed for exactly that many bytes, so
// it cannot carry more (Size 0 only in grants the CLI pre-issues itself, which S3 caps at 5 GB per request).
type Request struct {
	Op   string `json:"op"`
	Key  string `json:"key"`
	Size int64  `json:"size,omitempty"`
}

// Grant is a presigned URL for one object and one operation (an upload: for exactly Size bytes when Size > 0).
type Grant struct {
	Op      string    `json:"op"`
	Key     string    `json:"key"`
	Size    int64     `json:"size,omitempty"`
	URL     string    `json:"url"`
	Expires time.Time `json:"expires"`
}

// Policy is what one pod may be granted. Put prefixes must end with "/" (so "runs/a/" does not cover "runs/ab/").
type Policy struct {
	Get         []string     `json:"get,omitempty"`          // exact keys
	PutPrefixes []string     `json:"put_prefixes,omitempty"` // e.g. runs/<run>/, jobs/<tag>/
	Plans       []UploadPlan `json:"plans,omitempty"`        // the expected uploads under a prefix (a training run)
}

// HeavyBytes: an upload this large or larger is a checkpoint, held to its prefix's UploadPlan pace.
const HeavyBytes = 64 << 20

// UploadPlan is what a run is expected to upload, derived from its config when the pod is made (remote-pod train):
// a compromised pod can then upload only what the run itself would, at the pace it would.
type UploadPlan struct {
	Prefix     string `json:"prefix"`      // runs/<run>/
	MaxObjects int    `json:"max_objects"` // distinct keys under Prefix, ever: the fixed files + the numbered checkpoints
	// heavy uploads (checkpoints): at most MaxNewHeavyPerHour new keys an hour, and the same key at most once per
	// MinRewrite (about an epoch at the fastest a GPU could train)
	MaxNewHeavyPerHour int           `json:"max_new_heavy_per_hour"`
	MinRewrite         time.Duration `json:"min_rewrite"`
	MaxFileBytes       int64         `json:"max_file_bytes,omitempty"` // 0: Limits.MaxFileBytes
}

// RunUploadPlan derives a training run's plan: epochs in all, a numbered checkpoint every checkpointEvery epochs (0:
// none), epochTokens per epoch (0: unknown, then a minute is the fastest epoch). The throughput ceiling is generous
// (MaxTokensPerSecond): the plan must never stop an honest run.
func RunUploadPlan(prefix string, epochs, checkpointEvery int, epochTokens float64) UploadPlan {
	minEpoch := time.Minute
	if epochTokens > 0 {
		minEpoch = max(10*time.Second, time.Duration(epochTokens/MaxTokensPerSecond*float64(time.Second)))
	}
	numbered := 0
	if checkpointEvery > 0 {
		numbered = epochs/checkpointEvery + 1
	}
	perHour := 2 + 2 // model.lbc and best.lbc appear once, plus slack
	if checkpointEvery > 0 {
		perHour += int(time.Hour/(time.Duration(checkpointEvery)*minEpoch)) + 1
	}
	return UploadPlan{Prefix: prefix, MaxObjects: numbered + len(RunFixedFiles) + 8,
		MaxNewHeavyPerHour: perHour, MinRewrite: minEpoch / 2}
}

// MaxTokensPerSecond bounds how fast any GPU we rent could train (an RTX 4090 does ~0.22M tokens/s on the 47M model).
const MaxTokensPerSecond = 1.5e6

// RunFixedFiles are the files a run rewrites under its prefix (lbc's run directory and the agent's train.log).
var RunFixedFiles = []string{"model.lbc", "metrics.csv", "train.log", "config.json", "checkpoints/best.lbc"}

// MaxKeyLen bounds a key (S3 allows 1024 bytes; ours are short).
const MaxKeyLen = 512

// ErrRefused wraps every policy refusal.
var ErrRefused = errors.New("refused by the pod's policy")

func refuse(format string, a ...any) error {
	return fmt.Errorf("%w: %s", ErrRefused, fmt.Sprintf(format, a...))
}

// CleanKey accepts plain object keys only: relative, "/"-separated, no empty / "." / ".." segments, no backslash,
// no control or space characters, no percent-encoding (a key is used verbatim, never decoded), at most MaxKeyLen.
func CleanKey(key string) error {
	if key == "" || len(key) > MaxKeyLen {
		return refuse("key length %d", len(key))
	}
	if strings.HasPrefix(key, "/") || strings.HasSuffix(key, "/") {
		return refuse("key %q: leading or trailing /", key)
	}
	for _, r := range key {
		if r == '\\' || r == '%' || unicode.IsControl(r) || unicode.IsSpace(r) || r == unicode.ReplacementChar ||
			r > unicode.MaxASCII {
			return refuse("key %q: character %q", key, r)
		}
	}
	for _, seg := range strings.Split(key, "/") {
		if seg == "" || seg == "." || seg == ".." {
			return refuse("key %q: segment %q", key, seg)
		}
	}
	return nil
}

// Validate checks the policy itself (the CLI's own input): prefixes end with "/" and are clean.
func (p Policy) Validate() error {
	for _, k := range p.Get {
		if err := CleanKey(k); err != nil {
			return fmt.Errorf("policy get %q: %w", k, err)
		}
	}
	for _, pl := range p.Plans {
		if !strings.HasSuffix(pl.Prefix, "/") || pl.MaxObjects < 0 || pl.MaxNewHeavyPerHour < 0 {
			return fmt.Errorf("policy plan for %q: bad", pl.Prefix)
		}
	}
	for _, pre := range p.PutPrefixes {
		if !strings.HasSuffix(pre, "/") {
			return fmt.Errorf("policy put prefix %q must end with /", pre)
		}
		if err := CleanKey(strings.TrimSuffix(pre, "/")); err != nil {
			return fmt.Errorf("policy put prefix %q: %w", pre, err)
		}
	}
	return nil
}

// Check decides one request.
func (p Policy) Check(r Request) error {
	if err := CleanKey(r.Key); err != nil {
		return err
	}
	switch r.Op {
	case Get:
		for _, k := range p.Get {
			if r.Key == k {
				return nil
			}
		}
		return refuse("download of %q is not one of the pod's inputs", r.Key)
	case Put:
		for _, pre := range p.PutPrefixes {
			if strings.HasSuffix(pre, "/") && strings.HasPrefix(r.Key, pre) && len(r.Key) > len(pre) {
				return nil
			}
		}
		return refuse("upload to %q is outside the pod's prefixes %v", r.Key, p.PutPrefixes)
	}
	return refuse("operation %q (only get and put exist)", r.Op)
}

// Presigner makes the URLs (store.Store; a fake in tests). PresignPut with size > 0 signs the Content-Length too.
type Presigner interface {
	PresignGet(ctx context.Context, key string, ttl time.Duration) (string, error)
	PresignPut(ctx context.Context, key string, ttl time.Duration, size int64) (string, error)
}

// Limits is the protocol a pod's requests must follow beyond the policy: what a compromised pod could do with
// requests the policy allows (flood the store, hammer one object, burn operations) is bounded here.
type Limits struct {
	PerMinute, PerHour int           // grants of any kind
	PerKeyPerHour      int           // grants for the same object (a checkpoint rewritten every few minutes: fine)
	MaxObjects         int           // distinct objects a pod may upload, ever
	MaxFileBytes       int64         // one upload
	UploadBytesPerDay  int64         // all uploads in the last 24 hours (declared sizes: the URLs hold them to it)
	SuspendAfter       int           // refusals within SuspendWindow that suspend the pod (it looks compromised)
	SuspendWindow      time.Duration //
	SuspendFor         time.Duration // every request is refused meanwhile
}

// DefaultLimits fit a training run (a checkpoint every few minutes, logs every 10 s through re-used grants).
var DefaultLimits = Limits{PerMinute: 60, PerHour: 1200, PerKeyPerHour: 90, MaxObjects: 5000,
	MaxFileBytes: 5 << 30, UploadBytesPerDay: 200 << 30, SuspendAfter: 20, SuspendWindow: 10 * time.Minute,
	SuspendFor: 15 * time.Minute}

// Entry is one decision, as the journal keeps it.
type Entry struct {
	Time   time.Time `json:"time"`
	Op     string    `json:"op"`
	Key    string    `json:"key"`
	Size   int64     `json:"size,omitempty"`
	OK     bool      `json:"ok"`
	Reason string    `json:"reason,omitempty"`
}

// Journal keeps every decision (an audit trail; the budgets are rebuilt from it when the signer restarts).
type Journal interface {
	Append(Entry) error
	Load() ([]Entry, error)
}

// Signer answers a pod's requests: its Policy, then its Limits, then a URL valid for TTL.
type Signer struct {
	Policy  Policy
	Store   Presigner
	TTL     time.Duration // default DefaultTTL
	Limits  *Limits       // nil: DefaultLimits
	Journal Journal       // nil: none (decisions are only counted in memory)
	OnGrant func(Request, error)
	// PerMinute overrides Limits.PerMinute (kept for callers that only set a rate).
	PerMinute int

	mu        sync.Mutex
	loaded    bool
	history   []Entry // granted, last 24 hours
	objects   map[string]bool
	refusals  []time.Time
	suspended time.Time // refuse everything until then
}

const (
	DefaultTTL       = 15 * time.Minute
	DefaultPerMinute = 60
	// LongTTL is for the grants made when a pod is created (its inputs, its fixed-name outputs): they must work while
	// no signer is connected. 7 days is the longest an S3 presigned URL lives.
	LongTTL = 7 * 24 * time.Hour
)

// ErrSuspended: the pod sent too many refused requests; everything is refused for a while.
var ErrSuspended = errors.New("suspended")

func (s *Signer) limits() Limits {
	l := DefaultLimits
	if s.Limits != nil {
		l = *s.Limits
	}
	if s.PerMinute > 0 {
		l.PerMinute = s.PerMinute
	}
	return l
}

func (s *Signer) Sign(ctx context.Context, r Request) (Grant, error) {
	g, err := s.sign(ctx, r)
	if s.OnGrant != nil {
		s.OnGrant(r, err)
	}
	return g, err
}

func (s *Signer) sign(ctx context.Context, r Request) (Grant, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	now := time.Now()
	s.load()
	l := s.limits()
	decide := func(err error) (Grant, error) {
		e := Entry{Time: now, Op: r.Op, Key: r.Key, Size: r.Size, OK: err == nil}
		if err != nil {
			e.Reason = err.Error()
			if !errors.Is(err, ErrSuspended) {
				s.refusals = append(s.refusals, now)
				recent := 0
				for _, t := range s.refusals {
					if now.Sub(t) < l.SuspendWindow {
						recent++
					}
				}
				if l.SuspendAfter > 0 && recent >= l.SuspendAfter {
					s.suspended = now.Add(l.SuspendFor)
					e.Reason += fmt.Sprintf("; %d refusals in %s: pod suspended until %s", recent, l.SuspendWindow,
						s.suspended.Format(time.TimeOnly))
				}
			}
		}
		if s.Journal != nil {
			_ = s.Journal.Append(e)
		}
		return Grant{}, err
	}
	if now.Before(s.suspended) {
		return decide(fmt.Errorf("%w: %w until %s (too many refused requests: is the pod compromised?)",
			ErrRefused, ErrSuspended, s.suspended.Format(time.TimeOnly)))
	}
	if err := s.Policy.Check(r); err != nil {
		return decide(err)
	}
	if err := s.checkLimits(r, l, now); err != nil {
		return decide(err)
	}
	ttl := s.TTL
	if ttl <= 0 {
		ttl = DefaultTTL
	}
	g, err := Make(ctx, s.Store, r, ttl)
	if err != nil {
		return Grant{}, err // the store failed, not the pod: not a refusal
	}
	e := Entry{Time: now, Op: r.Op, Key: r.Key, Size: r.Size, OK: true}
	s.history = append(s.history, e)
	if r.Op == Put {
		s.objects[r.Key] = true
	}
	if s.Journal != nil {
		_ = s.Journal.Append(e)
	}
	return g, nil
}

func (s *Signer) load() {
	if s.loaded {
		return
	}
	s.loaded, s.objects = true, map[string]bool{}
	if s.Journal == nil {
		return
	}
	es, _ := s.Journal.Load()
	for _, e := range es {
		if !e.OK {
			continue
		}
		if e.Op == Put {
			s.objects[e.Key] = true
		}
		if time.Since(e.Time) < 24*time.Hour {
			s.history = append(s.history, e)
		}
	}
}

func (s *Signer) checkLimits(r Request, l Limits, now time.Time) error {
	if r.Op == Put {
		if r.Size <= 0 {
			return refuse("an upload must declare its size (the URL is signed for exactly that many bytes)")
		}
		if l.MaxFileBytes > 0 && r.Size > l.MaxFileBytes {
			return refuse("upload of %d bytes: more than %d per file", r.Size, l.MaxFileBytes)
		}
		if l.MaxObjects > 0 && !s.objects[r.Key] && len(s.objects) >= l.MaxObjects {
			return refuse("the pod already uploaded %d distinct objects (the most it may)", len(s.objects))
		}
		if err := s.checkPlan(r, now); err != nil {
			return err
		}
	} else if r.Size != 0 {
		return refuse("a download declares no size")
	}
	var minute, hour, sameKey int
	var bytes int64
	kept := s.history[:0]
	for _, e := range s.history {
		age := now.Sub(e.Time)
		if age >= 24*time.Hour {
			continue
		}
		kept = append(kept, e)
		if e.Op == Put {
			bytes += e.Size
		}
		if age < time.Hour {
			hour++
			if e.Key == r.Key && e.Op == r.Op {
				sameKey++
			}
		}
		if age < time.Minute {
			minute++
		}
	}
	s.history = kept
	switch {
	case l.PerMinute > 0 && minute >= l.PerMinute:
		return refuse("more than %d grants a minute", l.PerMinute)
	case l.PerHour > 0 && hour >= l.PerHour:
		return refuse("more than %d grants an hour", l.PerHour)
	case l.PerKeyPerHour > 0 && sameKey >= l.PerKeyPerHour:
		return refuse("more than %d grants an hour for %s", l.PerKeyPerHour, r.Key)
	case r.Op == Put && l.UploadBytesPerDay > 0 && bytes+r.Size > l.UploadBytesPerDay:
		return refuse("upload budget: %d of %d bytes used in 24 hours", bytes, l.UploadBytesPerDay)
	}
	return nil
}

// checkPlan holds an upload to its prefix's plan: distinct objects, new checkpoints an hour, rewrite pace.
func (s *Signer) checkPlan(r Request, now time.Time) error {
	for _, p := range s.Policy.Plans {
		if !strings.HasPrefix(r.Key, p.Prefix) {
			continue
		}
		if p.MaxFileBytes > 0 && r.Size > p.MaxFileBytes {
			return refuse("upload of %d bytes: the run's files are at most %d", r.Size, p.MaxFileBytes)
		}
		under := 0
		for k := range s.objects {
			if strings.HasPrefix(k, p.Prefix) {
				under++
			}
		}
		if p.MaxObjects > 0 && !s.objects[r.Key] && under >= p.MaxObjects {
			return refuse("the run's plan expects at most %d files under %s", p.MaxObjects, p.Prefix)
		}
		if r.Size < HeavyBytes {
			return nil
		}
		newHeavy := 0
		for _, e := range s.history {
			if e.Op != Put || e.Size < HeavyBytes || !strings.HasPrefix(e.Key, p.Prefix) || now.Sub(e.Time) >= time.Hour {
				continue
			}
			if e.Key == r.Key && p.MinRewrite > 0 && now.Sub(e.Time) < p.MinRewrite {
				return refuse("%s was granted %s ago: the run's plan rewrites a checkpoint at most every %s", r.Key,
					now.Sub(e.Time).Round(time.Second), p.MinRewrite.Round(time.Second))
			}
		}
		seen := map[string]bool{}
		for _, e := range s.history {
			if e.Op == Put && e.Size >= HeavyBytes && strings.HasPrefix(e.Key, p.Prefix) && now.Sub(e.Time) < time.Hour &&
				!s.firstSeenBefore(e.Key, e.Time) && !seen[e.Key] {
				seen[e.Key] = true
				newHeavy++
			}
		}
		if !s.objects[r.Key] && p.MaxNewHeavyPerHour > 0 && newHeavy >= p.MaxNewHeavyPerHour {
			return refuse("the run's plan expects at most %d new checkpoints an hour under %s", p.MaxNewHeavyPerHour, p.Prefix)
		}
		return nil
	}
	return nil
}

// firstSeenBefore reports whether key was granted earlier than t (so the grant at t was a rewrite, not a new file).
func (s *Signer) firstSeenBefore(key string, t time.Time) bool {
	for _, e := range s.history {
		if e.Key == key && e.Op == Put && e.Time.Before(t) {
			return true
		}
	}
	return false
}

// Suspended reports until when the pod is suspended (zero: it is not).
func (s *Signer) Suspended() time.Time {
	s.mu.Lock()
	defer s.mu.Unlock()
	if time.Now().Before(s.suspended) {
		return s.suspended
	}
	return time.Time{}
}

// Make signs a request without the policy or the rate limit: for the grants the CLI itself decides at creation.
func Make(ctx context.Context, st Presigner, r Request, ttl time.Duration) (Grant, error) {
	if err := CleanKey(r.Key); err != nil {
		return Grant{}, err
	}
	var u string
	var err error
	switch r.Op {
	case Get:
		u, err = st.PresignGet(ctx, r.Key, ttl)
	case Put:
		u, err = st.PresignPut(ctx, r.Key, ttl, r.Size)
	default:
		return Grant{}, refuse("operation %q", r.Op)
	}
	if err != nil {
		return Grant{}, err
	}
	return Grant{Op: r.Op, Key: r.Key, Size: r.Size, URL: u, Expires: time.Now().Add(ttl)}, nil
}

// JobLogPrefix is where a pod's job logs go: under its tag (a random name made at creation), so its upload grants
// cover its own logs only.
func JobLogPrefix(tag string) string { return "jobs/" + tag + "/" }

// PodPolicy is what a pod may be granted: uploads of its own job logs, plus the extra prefixes and exact keys given.
func PodPolicy(tag string, putPrefixes, getKeys []string) (Policy, error) {
	if err := CleanKey("jobs/" + tag); err != nil || tag == "" || strings.Contains(tag, "/") {
		return Policy{}, fmt.Errorf("pod tag %q: not a single plain name", tag)
	}
	p := Policy{PutPrefixes: []string{JobLogPrefix(tag)}}
	for _, pre := range putPrefixes {
		if !strings.HasSuffix(pre, "/") {
			pre += "/"
		}
		p.PutPrefixes = append(p.PutPrefixes, pre)
	}
	p.Get = append(p.Get, getKeys...)
	return p, p.Validate()
}

// PodEnv is the store part of a pod's environment: its tag and its pre-issued grants (JSON, base64). Never a key.
func PodEnv(tag string, pre []Grant) (map[string]string, error) {
	env := map[string]string{"REMOTE_POD_TAG": tag}
	if len(pre) > 0 {
		b, err := json.Marshal(pre)
		if err != nil {
			return nil, err
		}
		env["REMOTE_POD_GRANTS"] = base64.StdEncoding.EncodeToString(b)
	}
	return env, nil
}

// Preissue signs what a pod needs while no signer may be connected: every download of its policy, and uploads to
// the given keys (each must be inside the policy), valid for LongTTL.
func Preissue(ctx context.Context, st Presigner, p Policy, puts []string) ([]Grant, error) {
	var out []Grant
	for _, k := range p.Get {
		g, err := Make(ctx, st, Request{Op: Get, Key: k}, LongTTL)
		if err != nil {
			return nil, err
		}
		out = append(out, g)
	}
	for _, k := range puts {
		r := Request{Op: Put, Key: k}
		if err := p.Check(r); err != nil {
			return nil, err
		}
		g, err := Make(ctx, st, r, LongTTL)
		if err != nil {
			return nil, err
		}
		out = append(out, g)
	}
	return out, nil
}
