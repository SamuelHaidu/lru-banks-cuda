package agent

import (
	"encoding/json"
	"errors"
	"io"
	"log"
	"net/http"
	"os"
	"os/exec"
	"sync"
	"syscall"

	"github.com/creack/pty"
	"github.com/gorilla/websocket"

	"github.com/SamuelHaidu/rnn-research/tools/remote-pod/internal/proto"
)

var signals = map[string]syscall.Signal{"INT": syscall.SIGINT, "TERM": syscall.SIGTERM, "KILL": syscall.SIGKILL,
	"HUP": syscall.SIGHUP, "QUIT": syscall.SIGQUIT}

// command builds `bash -lc <command>` in the request's directory and environment.
func (s *Server) command(start proto.ExecStart) *exec.Cmd {
	cmd := exec.Command("bash", "-lc", start.Command)
	cmd.Dir = s.Resolve(start.Cwd)
	cmd.Env = os.Environ()
	for k, v := range start.Env {
		cmd.Env = append(cmd.Env, k+"="+v)
	}
	return cmd
}

// exitCode is the process's exit status (128 + the signal when a signal ended it, as a shell reports).
func exitCode(err error) int {
	if err == nil {
		return 0
	}
	var ee *exec.ExitError
	if errors.As(err, &ee) {
		if ws, ok := ee.Sys().(syscall.WaitStatus); ok && ws.Signaled() {
			return 128 + int(ws.Signal())
		}
		return ee.ExitCode()
	}
	return 127
}

// exec runs one command for the websocket's lifetime: output streamed, stdin / signals / resizes from the client.
// The command's process group is killed if the client goes away (as an ssh session's would be).
func (s *Server) exec(w http.ResponseWriter, r *http.Request) {
	c, err := s.upgrader.Upgrade(w, r, nil)
	if err != nil {
		return
	}
	defer c.Close()
	s.active.Add(1)
	defer s.active.Add(-1)
	out := &wsWriter{c: c}

	var start proto.ExecStart
	if err := c.ReadJSON(&start); err != nil {
		return
	}
	cmd := s.command(start)
	var (
		stdin    io.WriteCloser
		ptmx     *os.File
		outputs  sync.WaitGroup
	)
	if start.TTY {
		size := &pty.Winsize{Rows: start.Rows, Cols: start.Cols}
		if size.Rows == 0 {
			size.Rows, size.Cols = 24, 80
		}
		ptmx, err = pty.StartWithSize(cmd, size) // a new session: the shell leads its process group
		if err == nil {
			stdin = ptmx
			outputs.Add(1)
			go func() { defer outputs.Done(); pump(ptmx, out, proto.ChanStdout) }()
		}
	} else {
		cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
		var so, se io.ReadCloser
		if stdin, err = cmd.StdinPipe(); err == nil {
			if so, err = cmd.StdoutPipe(); err == nil {
				if se, err = cmd.StderrPipe(); err == nil {
					err = cmd.Start()
				}
			}
		}
		if err == nil {
			outputs.Add(2)
			go func() { defer outputs.Done(); pump(so, out, proto.ChanStdout) }()
			go func() { defer outputs.Done(); pump(se, out, proto.ChanStderr) }()
		}
	}
	if err != nil {
		_ = out.control(proto.Control{Type: "exit", Code: 127, Error: err.Error()})
		return
	}
	pgid := cmd.Process.Pid
	kill := func(sig syscall.Signal) { _ = syscall.Kill(-pgid, sig) }

	// the client's frames: stdin, stdin EOF, signals, resizes; the websocket closing means the client is gone
	gone := make(chan struct{})
	go func() {
		defer close(gone)
		for {
			kind, msg, err := c.ReadMessage()
			if err != nil {
				return
			}
			if kind == websocket.BinaryMessage && len(msg) > 0 && msg[0] == proto.ChanStdin {
				_, _ = stdin.Write(msg[1:])
				continue
			}
			var ctl proto.Control
			if json.Unmarshal(msg, &ctl) != nil {
				continue
			}
			switch ctl.Type {
			case "stdin_eof":
				if ptmx != nil {
					_, _ = ptmx.Write([]byte{4}) // ^D
				} else {
					_ = stdin.Close()
				}
			case "signal":
				if sig, ok := signals[ctl.Signal]; ok {
					kill(sig)
				}
			case "resize":
				if ptmx != nil {
					_ = pty.Setsize(ptmx, &pty.Winsize{Rows: ctl.Rows, Cols: ctl.Cols})
				}
			}
		}
	}()

	// all output read first (Wait closes the pipes), then the exit status
	done := make(chan error, 1)
	go func() { outputs.Wait(); done <- cmd.Wait() }()
	select {
	case err = <-done:
	case <-gone:
		kill(syscall.SIGHUP)
		kill(syscall.SIGKILL)
		<-done
		return
	}
	if ptmx != nil {
		_ = ptmx.Close()
	}
	if e := out.control(proto.Control{Type: "exit", Code: exitCode(err)}); e != nil {
		log.Printf("exec: reporting the exit: %v", e)
	}
	_ = c.WriteMessage(websocket.CloseMessage, websocket.FormatCloseMessage(websocket.CloseNormalClosure, ""))
}
