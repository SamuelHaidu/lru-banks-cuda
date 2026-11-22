package agent

import (
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"sync"
	"syscall"
	"time"

	"github.com/gorilla/websocket"

	"github.com/SamuelHaidu/rnn-research/tools/remote-pod/internal/proto"
)

// Jobs are detached commands: they outlive the connection that started them; their output (stdout and stderr
// together) goes to <dir>/<id>.log, which clients follow from any byte offset.
type Jobs struct {
	dir  string
	mu   sync.Mutex
	jobs map[string]*job
	hooks []func(proto.Job) // called when a job ends (the log upload, the train runner)
}

// OnExit adds a function called (in the job's goroutine) when any job ends.
func (j *Jobs) OnExit(fn func(proto.Job)) {
	j.mu.Lock()
	defer j.mu.Unlock()
	j.hooks = append(j.hooks, fn)
}

type job struct {
	proto.Job
	pgid int
}

func NewJobs(dir string) *Jobs { return &Jobs{dir: dir, jobs: map[string]*job{}} }

func (j *Jobs) LogPath(id string) string { return filepath.Join(j.dir, id+".log") }

func newID() string {
	b := make([]byte, 4)
	_, _ = rand.Read(b)
	return hex.EncodeToString(b)
}

// RunJob starts a detached job (its own process group), its output to the job's log file.
func (s *Server) RunJob(start proto.ExecStart) (proto.Job, error) {
	j := s.jobs
	if err := os.MkdirAll(j.dir, 0o755); err != nil {
		return proto.Job{}, err
	}
	id := newID()
	name := start.Name
	if name == "" {
		name = id
	}
	f, err := os.Create(j.LogPath(id))
	if err != nil {
		return proto.Job{}, err
	}
	cmd := s.command(start)
	cmd.Stdout, cmd.Stderr = f, f
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	if err := cmd.Start(); err != nil {
		f.Close()
		return proto.Job{}, err
	}
	jb := &job{Job: proto.Job{ID: id, Name: name, Command: start.Command, Cwd: cmd.Dir, Started: time.Now(), Running: true},
		pgid: cmd.Process.Pid}
	j.mu.Lock()
	j.jobs[id] = jb
	started := jb.Job // the caller's copy, taken before the waiter can update the job
	j.mu.Unlock()
	go func() {
		err := cmd.Wait()
		f.Close()
		j.mu.Lock()
		jb.Running, jb.Ended, jb.ExitCode = false, time.Now(), exitCode(err)
		snapshot, hooks := jb.Job, append([]func(proto.Job){}, j.hooks...)
		j.mu.Unlock()
		for _, h := range hooks {
			h(snapshot)
		}
	}()
	return started, nil
}

func (j *Jobs) Get(idOrName string) (proto.Job, bool) {
	j.mu.Lock()
	defer j.mu.Unlock()
	if jb, ok := j.jobs[idOrName]; ok {
		return j.withSize(jb.Job), true
	}
	var best *job  // by name: the latest
	for _, jb := range j.jobs {
		if jb.Name == idOrName && (best == nil || jb.Started.After(best.Started)) {
			best = jb
		}
	}
	if best == nil {
		return proto.Job{}, false
	}
	return j.withSize(best.Job), true
}

func (j *Jobs) withSize(x proto.Job) proto.Job {
	if fi, err := os.Stat(j.LogPath(x.ID)); err == nil {
		x.LogBytes = fi.Size()
	}
	return x
}

func (j *Jobs) List() []proto.Job {
	j.mu.Lock()
	defer j.mu.Unlock()
	out := make([]proto.Job, 0, len(j.jobs))
	for _, jb := range j.jobs {
		out = append(out, j.withSize(jb.Job))
	}
	sort.Slice(out, func(a, b int) bool { return out[a].Started.Before(out[b].Started) })
	return out
}

func (j *Jobs) Running() int {
	j.mu.Lock()
	defer j.mu.Unlock()
	n := 0
	for _, jb := range j.jobs {
		if jb.Running {
			n++
		}
	}
	return n
}

func (j *Jobs) Signal(id string, sig syscall.Signal) error {
	j.mu.Lock()
	defer j.mu.Unlock()
	jb, ok := j.jobs[id]
	if !ok {
		return fmt.Errorf("no job %s", id)
	}
	if !jb.Running {
		return fmt.Errorf("job %s has ended", id)
	}
	return syscall.Kill(-jb.pgid, sig)
}

func (s *Server) startJob(w http.ResponseWriter, r *http.Request) {
	var start proto.ExecStart
	if err := json.NewDecoder(r.Body).Decode(&start); err != nil {
		fail(w, 400, err)
		return
	}
	jb, err := s.RunJob(start)
	if err != nil {
		fail(w, 500, err)
		return
	}
	writeJSON(w, 200, jb)
}

func (s *Server) listJobs(w http.ResponseWriter, r *http.Request) { writeJSON(w, 200, s.jobs.List()) }

func (s *Server) getJob(w http.ResponseWriter, r *http.Request) {
	jb, ok := s.jobs.Get(r.PathValue("id"))
	if !ok {
		fail(w, 404, fmt.Errorf("no job %s", r.PathValue("id")))
		return
	}
	writeJSON(w, 200, jb)
}

func (s *Server) signalJob(w http.ResponseWriter, r *http.Request) {
	jb, ok := s.jobs.Get(r.PathValue("id"))
	if !ok {
		fail(w, 404, fmt.Errorf("no job %s", r.PathValue("id")))
		return
	}
	sig, ok := signals[r.URL.Query().Get("signal")]
	if !ok {
		sig = signals["INT"]
	}
	if err := s.jobs.Signal(jb.ID, sig); err != nil {
		fail(w, 409, err)
		return
	}
	writeJSON(w, 200, map[string]bool{"ok": true})
}

// jobLog streams the job's log from ?offset=N, following it until the job has ended and everything was sent.
func (s *Server) jobLog(w http.ResponseWriter, r *http.Request) {
	jb, ok := s.jobs.Get(r.PathValue("id"))
	if !ok {
		fail(w, 404, fmt.Errorf("no job %s", r.PathValue("id")))
		return
	}
	offset, _ := strconv.ParseInt(r.URL.Query().Get("offset"), 10, 64)
	follow := r.URL.Query().Get("follow") != "0"
	c, err := s.upgrader.Upgrade(w, r, nil)
	if err != nil {
		return
	}
	defer c.Close()
	out := &wsWriter{c: c}
	go func() { // drain the client's frames: a closed socket ends the stream
		for {
			if _, _, err := c.ReadMessage(); err != nil {
				return
			}
		}
	}()
	f, err := os.Open(s.jobs.LogPath(jb.ID))
	if err != nil {
		_ = out.control(proto.Control{Type: "error", Error: err.Error()})
		return
	}
	defer f.Close()
	if _, err := f.Seek(offset, io.SeekStart); err != nil {
		_ = out.control(proto.Control{Type: "error", Error: err.Error()})
		return
	}
	buf := make([]byte, 64<<10)
	idle := 0
	for {
		n, err := f.Read(buf)
		if n > 0 {
			if out.binary(proto.ChanStdout, buf[:n]) != nil {
				return
			}
			idle = 0
			continue
		}
		if err != nil && err != io.EOF {
			_ = out.control(proto.Control{Type: "error", Error: err.Error()})
			return
		}
		cur, _ := s.jobs.Get(jb.ID)
		if !cur.Running || !follow {
			// one more read: output written just before the end
			if n, _ := f.Read(buf); n > 0 {
				if out.binary(proto.ChanStdout, buf[:n]) != nil {
					return
				}
				continue
			}
			_ = out.control(proto.Control{Type: "end", Code: cur.ExitCode, Error: map[bool]string{true: "running"}[cur.Running]})
			_ = c.WriteMessage(websocket.CloseMessage, websocket.FormatCloseMessage(websocket.CloseNormalClosure, ""))
			return
		}
		idle++
		if idle%150 == 0 { // keep the connection alive through proxies (every ~30 s of silence)
			out.mu.Lock()
			err := c.WriteControl(websocket.PingMessage, nil, time.Now().Add(5*time.Second))
			out.mu.Unlock()
			if err != nil {
				return
			}
		}
		time.Sleep(200 * time.Millisecond)
	}
}
