// Package client is the CLI's side of the agent protocol (proto): exec, files, sync, jobs and their logs.
package client

import (
	"bytes"
	"crypto/tls"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/gorilla/websocket"

	"github.com/SamuelHaidu/rnn-research/tools/remote-pod/internal/pin"
	"github.com/SamuelHaidu/rnn-research/tools/remote-pod/internal/proto"
	"github.com/SamuelHaidu/rnn-research/tools/remote-pod/internal/syncx"
	"github.com/SamuelHaidu/rnn-research/tools/remote-pod/internal/tlsroots"
)

type Client struct {
	Base  string // http(s)://host:port
	Token string
	HTTP  *http.Client
	tls   *tls.Config // nil: the system's CAs (RunPod's proxy); set by Pin
}

func New(base, token string) *Client {
	// RunPod's proxy and any other agent behind a public certificate: the embedded roots only (tlsroots)
	return &Client{Base: strings.TrimRight(base, "/"), Token: token, HTTP: tlsroots.Client(0), tls: tlsroots.Config()}
}

// CheckBase refuses an agent address that would carry the token in clear: https, or http to this machine only.
func CheckBase(raw string) error {
	u, err := url.Parse(raw)
	if err != nil {
		return fmt.Errorf("agent address %q: %w", raw, err)
	}
	if u.Scheme == "https" && u.Host != "" {
		return nil
	}
	h := u.Hostname()
	if ip := net.ParseIP(h); u.Scheme == "http" && (h == "localhost" || (ip != nil && ip.IsLoopback())) {
		return nil
	}
	return fmt.Errorf("agent address %q: only https (or http to this machine) may carry the token", raw)
}

// Pin trusts only this certificate (PEM) for the agent's TLS: a pod reached on its public IP (vast.ai), see pin.
func (c *Client) Pin(certPEM []byte) error {
	cfg, err := pin.ClientConfig(certPEM)
	if err != nil {
		return err
	}
	c.tls = cfg
	c.HTTP = &http.Client{Transport: &http.Transport{TLSClientConfig: cfg, Proxy: http.ProxyFromEnvironment}}
	return nil
}

func (c *Client) header() http.Header {
	return http.Header{"Authorization": {"Bearer " + c.Token}}
}

func (c *Client) do(method, path string, q url.Values, body io.Reader, out any) error {
	u := c.Base + path
	if len(q) > 0 {
		u += "?" + q.Encode()
	}
	req, err := http.NewRequest(method, u, body)
	if err != nil {
		return err
	}
	req.Header = c.header()
	resp, err := c.HTTP.Do(req)
	if err != nil {
		return fmt.Errorf("agent unreachable at %s: %w", c.Base, tlsroots.Explain(c.Base, err))
	}
	defer resp.Body.Close()
	if resp.StatusCode == http.StatusUnauthorized {
		return errors.New("the agent refused the token (another pod's state? run remote-pod status)")
	}
	if resp.StatusCode >= 300 {
		var e struct {
			Error string `json:"error"`
		}
		b, _ := io.ReadAll(resp.Body)
		if json.Unmarshal(b, &e) == nil && e.Error != "" {
			return errors.New(e.Error)
		}
		return fmt.Errorf("agent: HTTP %d: %s", resp.StatusCode, strings.TrimSpace(string(b)))
	}
	if out != nil {
		return json.NewDecoder(resp.Body).Decode(out)
	}
	return nil
}

func (c *Client) dial(path string, q url.Values) (*websocket.Conn, error) {
	u := c.Base + path
	u = "ws" + strings.TrimPrefix(u, "http")
	if len(q) > 0 {
		u += "?" + q.Encode()
	}
	d := websocket.Dialer{HandshakeTimeout: 30 * time.Second, ReadBufferSize: 64 << 10, WriteBufferSize: 64 << 10,
		TLSClientConfig: c.tls}
	conn, resp, err := d.Dial(u, c.header())
	if err != nil {
		if resp != nil && resp.StatusCode == http.StatusUnauthorized {
			return nil, errors.New("the agent refused the token (another pod's state? run remote-pod status)")
		}
		return nil, fmt.Errorf("agent unreachable at %s: %w", c.Base, tlsroots.Explain(c.Base, err))
	}
	return conn, nil
}

func (c *Client) Health() (proto.Health, error) {
	var h proto.Health
	return h, c.do("GET", "/v1/health", nil, nil, &h)
}

// ExecIO is what an exec connects to locally.
type ExecIO struct {
	Stdin   io.Reader // nil: none (EOF sent at once)
	Stdout  io.Writer
	Stderr  io.Writer
	Signals <-chan string // "INT", "TERM", ... forwarded to the remote process group
	Resize  <-chan [2]uint16
}

// Exec runs a command and returns its exit code.
func (c *Client) Exec(start proto.ExecStart, io_ ExecIO) (int, error) {
	conn, err := c.dial("/v1/exec", nil)
	if err != nil {
		return 0, err
	}
	defer conn.Close()
	if err := conn.WriteJSON(start); err != nil {
		return 0, err
	}
	send := make(chan any, 16) // one writer goroutine: binary []byte or proto.Control
	stop := make(chan struct{})
	defer close(stop)
	go func() {
		for {
			select {
			case m := <-send:
				switch v := m.(type) {
				case []byte:
					_ = conn.WriteMessage(websocket.BinaryMessage, v)
				case proto.Control:
					_ = conn.WriteJSON(v)
				}
			case <-stop:
				return
			}
		}
	}()
	if io_.Stdin == nil {
		send <- proto.Control{Type: "stdin_eof"}
	} else {
		go func() {
			buf := make([]byte, 32<<10)
			for {
				n, err := io_.Stdin.Read(buf)
				if n > 0 {
					select {
					case send <- append([]byte{proto.ChanStdin}, buf[:n]...):
					case <-stop:
						return
					}
				}
				if err != nil {
					select {
					case send <- proto.Control{Type: "stdin_eof"}:
					case <-stop:
					}
					return
				}
			}
		}()
	}
	go func() {
		for {
			select {
			case s, ok := <-io_.Signals:
				if !ok {
					return
				}
				send <- proto.Control{Type: "signal", Signal: s}
			case rc, ok := <-io_.Resize:
				if !ok {
					return
				}
				send <- proto.Control{Type: "resize", Rows: rc[0], Cols: rc[1]}
			case <-stop:
				return
			}
		}
	}()
	for {
		kind, msg, err := conn.ReadMessage()
		if err != nil {
			return 0, fmt.Errorf("the connection to the pod closed before the command ended: %w", err)
		}
		if kind == websocket.BinaryMessage && len(msg) > 0 {
			switch msg[0] {
			case proto.ChanStdout:
				_, _ = io_.Stdout.Write(msg[1:])
			case proto.ChanStderr:
				_, _ = io_.Stderr.Write(msg[1:])
			}
			continue
		}
		var ctl proto.Control
		if json.Unmarshal(msg, &ctl) == nil && ctl.Type == "exit" {
			if ctl.Error != "" {
				return ctl.Code, errors.New(ctl.Error)
			}
			return ctl.Code, nil
		}
	}
}

// Upload copies a local file or directory to dest on the pod (cp semantics); returns where it landed.
func (c *Client) Upload(local, dest string, excludes []string, progress func(int64)) (string, error) {
	pr, pw := io.Pipe()
	go func() { pw.CloseWithError(syncx.WriteTar(&countWriter{w: pw, f: progress}, local, excludes)) }()
	var r struct {
		Target string `json:"target"`
	}
	q := url.Values{"dest": {dest}, "base": {filepath.Base(filepath.Clean(local))}, "mode": {"cp"}}
	err := c.do("POST", "/v1/files", q, pr, &r)
	return r.Target, err
}

// Download copies a file or directory from the pod to local (cp semantics); returns where it landed.
func (c *Client) Download(remote, local string, progress func(int64)) (string, error) {
	req, err := http.NewRequest("GET", c.Base+"/v1/files?"+url.Values{"path": {remote}}.Encode(), nil)
	if err != nil {
		return "", err
	}
	req.Header = c.header()
	resp, err := c.HTTP.Do(req)
	if err != nil {
		return "", fmt.Errorf("agent unreachable at %s: %w", c.Base, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		b, _ := io.ReadAll(resp.Body)
		var e struct {
			Error string `json:"error"`
		}
		if json.Unmarshal(b, &e) == nil && e.Error != "" {
			return "", errors.New(e.Error)
		}
		return "", fmt.Errorf("agent: HTTP %d", resp.StatusCode)
	}
	fi, err := os.Stat(local)
	target := syncx.CpTarget(local, resp.Header.Get("X-Base"), err == nil && fi.IsDir())
	_, _, err = syncx.Extract(&countReader{r: resp.Body, f: progress}, target)
	return target, err
}

// Sync makes remoteRoot match localRoot for the files that differ (and deletes remote extras when del).
func (c *Client) Sync(localRoot, remoteRoot string, excludes []string, del bool) (sent, deleted []string, bytes int64, err error) {
	local, err := syncx.Manifest(localRoot, excludes)
	if err != nil {
		return nil, nil, 0, err
	}
	var rm struct {
		Files map[string]string `json:"files"`
	}
	body, _ := json.Marshal(map[string]any{"root": remoteRoot, "excludes": excludes})
	if err := c.do("POST", "/v1/manifest", nil, bytes_(body), &rm); err != nil {
		return nil, nil, 0, err
	}
	sent, deleted = syncx.Diff(local, rm.Files)
	if len(sent) > 0 {
		pr, pw := io.Pipe()
		cw := &countWriter{w: pw}
		go func() { pw.CloseWithError(syncx.WriteTarFiles(cw, localRoot, sent)) }()
		if err := c.do("POST", "/v1/files", url.Values{"dest": {remoteRoot}, "mode": {"merge"}}, pr, nil); err != nil {
			return nil, nil, 0, err
		}
		bytes = cw.n
	}
	if del && len(deleted) > 0 {
		body, _ := json.Marshal(map[string]any{"root": remoteRoot, "paths": deleted})
		if err := c.do("POST", "/v1/delete", nil, bytes_(body), nil); err != nil {
			return sent, nil, bytes, err
		}
	} else {
		deleted = nil
	}
	return sent, deleted, bytes, nil
}

func (c *Client) StartJob(start proto.ExecStart) (proto.Job, error) {
	var j proto.Job
	body, _ := json.Marshal(start)
	return j, c.do("POST", "/v1/jobs", nil, bytes_(body), &j)
}

func (c *Client) Jobs() ([]proto.Job, error) {
	var js []proto.Job
	return js, c.do("GET", "/v1/jobs", nil, nil, &js)
}

func (c *Client) Job(id string) (proto.Job, error) {
	var j proto.Job
	return j, c.do("GET", "/v1/jobs/"+url.PathEscape(id), nil, nil, &j)
}

func (c *Client) SignalJob(id, sig string) error {
	return c.do("POST", "/v1/jobs/"+url.PathEscape(id)+"/signal", url.Values{"signal": {sig}}, nil, nil)
}

// FollowLog writes a job's log from offset to w, reconnecting after drops (resuming at the last byte received),
// until the job ends (follow) or the log so far was sent. Returns the job's exit code.
func (c *Client) FollowLog(id string, offset int64, follow bool, w io.Writer, onReconnect func(error)) (int, error) {
	backoff := time.Second
	for {
		code, ended, err := c.followOnce(id, &offset, follow, w)
		if ended {
			return code, err
		}
		if err != nil && !isTransient(err) {
			return 0, err
		}
		if onReconnect != nil {
			onReconnect(err)
		}
		time.Sleep(backoff)
		if backoff < 15*time.Second {
			backoff *= 2
		}
	}
}

func (c *Client) followOnce(id string, offset *int64, follow bool, w io.Writer) (int, bool, error) {
	q := url.Values{"offset": {strconv.FormatInt(*offset, 10)}}
	if !follow {
		q.Set("follow", "0")
	}
	conn, err := c.dial("/v1/jobs/"+url.PathEscape(id)+"/log", q)
	if err != nil {
		return 0, false, err
	}
	defer conn.Close()
	for {
		kind, msg, err := conn.ReadMessage()
		if err != nil {
			return 0, false, err
		}
		if kind == websocket.BinaryMessage && len(msg) > 0 {
			n, _ := w.Write(msg[1:])
			*offset += int64(n)
			continue
		}
		var ctl proto.Control
		if json.Unmarshal(msg, &ctl) != nil {
			continue
		}
		switch ctl.Type {
		case "end":
			return ctl.Code, true, nil
		case "error":
			return 0, true, errors.New(ctl.Error)
		}
	}
}

func isTransient(err error) bool {
	msg := err.Error()
	return !strings.Contains(msg, "refused the token") && !strings.Contains(msg, "no job")
}

func bytes_(b []byte) io.Reader { return bytes.NewReader(b) }

type countWriter struct {
	w io.Writer
	n int64
	f func(int64)
}

func (c *countWriter) Write(p []byte) (int, error) {
	n, err := c.w.Write(p)
	c.n += int64(n)
	if c.f != nil {
		c.f(c.n)
	}
	return n, err
}

type countReader struct {
	r io.Reader
	n int64
	f func(int64)
}

func (c *countReader) Read(p []byte) (int, error) {
	n, err := c.r.Read(p)
	c.n += int64(n)
	if c.f != nil {
		c.f(c.n)
	}
	return n, err
}
