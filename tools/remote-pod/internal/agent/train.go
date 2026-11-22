package agent

import (
	"context"
	"io/fs"
	"log"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/SamuelHaidu/rnn-research/tools/remote-pod/internal/proto"
	"github.com/SamuelHaidu/rnn-research/tools/remote-pod/internal/store"
)

// TrainSpec is a run the agent executes on start (REMOTE_POD_JOB, JSON, set by `remote-pod train`).
type TrainSpec struct {
	Run          string            `json:"run"`
	Command      string            `json:"command"`   // e.g. "lbc train /work/runs/<run>"
	Datasets     []string          `json:"datasets"`  // store dataset names, fetched to /data/<name>
	Fetch        map[string]string `json:"fetch"`     // store key -> local path (e.g. the init checkpoint)
	Files        map[string]string `json:"files"`     // local path -> contents (the run's config.json)
	RunDir       string            `json:"run_dir"`   // uploaded to Prefix as files appear
	Prefix       string            `json:"prefix"`    // runs/<run>/
	DataDir      string            `json:"data_dir"`  // default /data
	StopWhenDone bool              `json:"stop_when_done"`
}

var numbered = regexp.MustCompile(`checkpoint_\d+\.lbc$`)

// Uploader copies new or changed files of a run directory to the store once they are stable (unchanged for a few
// seconds), and keeps only the newest numbered checkpoint locally (the others are in the store).
type Uploader struct {
	st     *store.Store
	dir    string
	prefix string
	mu     sync.Mutex
	sent   map[string]time.Time // path -> the mtime that was uploaded
}

func NewUploader(st *store.Store, dir, prefix string) *Uploader {
	return &Uploader{st: st, dir: dir, prefix: prefix, sent: map[string]time.Time{}}
}

// Pass uploads what changed; final uploads even files modified just now.
func (u *Uploader) Pass(ctx context.Context, final bool) {
	u.mu.Lock()
	defer u.mu.Unlock()
	var ckpts []string
	_ = filepath.WalkDir(u.dir, func(p string, d fs.DirEntry, err error) error {
		if err != nil || d.IsDir() || strings.HasSuffix(p, ".tmp") {
			return nil
		}
		fi, err := d.Info()
		if err != nil {
			return nil
		}
		if numbered.MatchString(p) {
			ckpts = append(ckpts, p)
		}
		if t, ok := u.sent[p]; ok && t.Equal(fi.ModTime()) {
			return nil
		}
		if !final && time.Since(fi.ModTime()) < 5*time.Second {
			return nil // still being written
		}
		rel, _ := filepath.Rel(u.dir, p)
		if err := u.st.PutFile(ctx, u.prefix+filepath.ToSlash(rel), p); err != nil {
			log.Printf("upload %s: %v", rel, err)
			return nil
		}
		u.sent[p] = fi.ModTime()
		return nil
	})
	sort.Strings(ckpts) // zero-padded epoch numbers: the newest last
	for i, p := range ckpts {
		if i < len(ckpts)-1 {
			if _, ok := u.sent[p]; ok {
				_ = os.Remove(p) // uploaded and superseded: free the disk
			}
		}
	}
}

// RunTrain fetches the inputs, runs the command as a job, uploads the run as it goes, and stops the pod at the end.
func (s *Server) RunTrain(ctx context.Context, st *store.Store, spec TrainSpec) error {
	dataDir := spec.DataDir
	if dataDir == "" {
		dataDir = "/data"
	}
	for _, name := range spec.Datasets {
		dst := filepath.Join(dataDir, name)
		if _, err := os.Stat(filepath.Join(dst, ".remote-pod-complete")); err == nil {
			continue
		}
		d, err := st.Dataset(ctx, name)
		if err != nil {
			return err
		}
		t0 := time.Now()
		files, n, err := st.GetDir(ctx, d.Key, dst, nil)
		if err != nil {
			return err
		}
		_ = os.WriteFile(filepath.Join(dst, ".remote-pod-complete"), nil, 0o644)
		log.Printf("dataset %s: %d files, %.1f GB in %s", name, files, float64(n)/1e9, time.Since(t0).Round(time.Second))
	}
	for key, path := range spec.Fetch {
		if err := st.GetFile(ctx, key, s.Resolve(path)); err != nil {
			return err
		}
	}
	for path, contents := range spec.Files {
		p := s.Resolve(path)
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			return err
		}
		if err := os.WriteFile(p, []byte(contents), 0o644); err != nil {
			return err
		}
	}
	runDir := s.Resolve(spec.RunDir)
	up := NewUploader(st, runDir, spec.Prefix)
	done := make(chan proto.Job, 1)
	s.jobs.OnExit(func(j proto.Job) {
		if j.Name == spec.Run {
			done <- j
		}
	})
	job, err := s.RunJob(proto.ExecStart{Command: spec.Command, Cwd: runDir, Name: spec.Run})
	if err != nil {
		return err
	}
	log.Printf("run %s: job %s started", spec.Run, job.ID)
	tick := time.NewTicker(30 * time.Second)
	defer tick.Stop()
	for {
		select {
		case <-tick.C:
			up.Pass(ctx, false)
			_ = st.PutFile(ctx, spec.Prefix+"train.log", s.jobs.LogPath(job.ID))
		case j := <-done:
			up.Pass(ctx, true)
			_ = st.PutFile(ctx, spec.Prefix+"train.log", s.jobs.LogPath(job.ID))
			log.Printf("run %s: ended with code %d, uploaded", spec.Run, j.ExitCode)
			if spec.StopWhenDone && s.opt.OnIdle != nil {
				s.opt.OnIdle()
			}
			return nil
		}
	}
}

// UploadJobLogs copies every job's log to jobs/<id>.log every interval (and once more when a job ends).
func (s *Server) UploadJobLogs(ctx context.Context, st *store.Store, every time.Duration) {
	s.jobs.OnExit(func(j proto.Job) { _ = st.PutFile(ctx, "jobs/"+j.ID+".log", s.jobs.LogPath(j.ID)) })
	for range time.Tick(every) {
		for _, j := range s.jobs.List() {
			if j.Running {
				_ = st.PutFile(ctx, "jobs/"+j.ID+".log", s.jobs.LogPath(j.ID))
			}
		}
	}
}
