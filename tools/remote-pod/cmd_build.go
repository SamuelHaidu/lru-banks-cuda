package main

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/spf13/cobra"

	"github.com/SamuelHaidu/rnn-research/tools/remote-pod/internal/agent"
	"github.com/SamuelHaidu/rnn-research/tools/remote-pod/internal/config"
	"github.com/SamuelHaidu/rnn-research/tools/remote-pod/internal/runpod"
	"github.com/SamuelHaidu/rnn-research/tools/remote-pod/internal/store"
	"github.com/SamuelHaidu/rnn-research/tools/remote-pod/internal/ui"
)

// run streams a local command's output (docker, git).
func run(name string, args ...string) error {
	ui.Note("%s %s", name, strings.Join(args, " "))
	c := exec.Command(name, args...)
	c.Stdout, c.Stderr = os.Stderr, os.Stderr
	return c.Run()
}

func gitSHA() string {
	out, err := exec.Command("git", "-C", config.ProjectRoot(), "rev-parse", "--short", "HEAD").Output()
	if err != nil {
		return "dev"
	}
	return strings.TrimSpace(string(out))
}

func buildCommands() []*cobra.Command {
	var arch, cuda string
	build := &cobra.Command{
		Use:   "build",
		Short: "Compile lbc (the engine) for the pods' GPUs in a local CUDA container (no GPU needed)",
		Long: `Compile the C++/CUDA engine in a local nvidia/cuda devel container (Docker), for the GPU architectures
RunPod rents, into engine/build/dist/lbc. Compiling CUDA needs no GPU. The binary goes into the train image
(remote-pod image build), or straight to a running pod for a quick test (remote-pod cp engine/build/dist/lbc :).`,
		Example: `  remote-pod build
  remote-pod build --arch 89                  only the RTX 4090 (faster)
  remote-pod build && remote-pod cp engine/build/dist/lbc : && remote-pod sh ./lbc --help`,
		GroupID: "build",
		RunE: func(cmd *cobra.Command, args []string) error {
			root := config.ProjectRoot()
			img := "remote-pod-builder:" + cuda
			if err := run("docker", "build", "--network", "host", "-q", "-t", img, "--build-arg", "CUDA="+cuda, "-f",
				filepath.Join(root, "docker", "builder.Dockerfile"), filepath.Join(root, "docker")); err != nil {
				return err
			}
			script := fmt.Sprintf("cmake -S . -B build/dist -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES='%s' >/dev/null && cmake --build build/dist --target lbc_cli", arch)
			t0 := time.Now()
			if err := run("docker", "run", "--rm", "--user", fmt.Sprintf("%d:%d", os.Getuid(), os.Getgid()),
				"-v", filepath.Join(root, "engine")+":/src", "-w", "/src", img, "bash", "-c", script); err != nil {
				return err
			}
			ui.Success("engine/build/dist/lbc (sm %s, %s)", arch, time.Since(t0).Round(time.Second))
			return nil
		},
	}
	build.Flags().StringVar(&arch, "arch", "89;90;120", "CUDA architectures: 89 RTX 4090 / L40, 90 H100, 120 RTX 5090, 80 A100")
	build.Flags().StringVar(&cuda, "cuda", "12.8.1", "CUDA version of the build container (12.8+ for sm_120)")

	var base, tag string
	imageBuild := &cobra.Command{
		Use:   "build [train|dev|agent]",
		Short: "Build the pod image (train: lbc + agent; dev: + nvcc, cmake, ninja, Nsight; agent: the agent only)",
		Long: `Build a pod image. The agent is compiled here first (a static Linux binary), then copied in; train also needs
engine/build/dist/lbc (remote-pod build). "agent" is the train image without lbc, for testing the agent: with
--base ubuntu:24.04 it needs no CUDA download.`,
		Example: `  remote-pod build && remote-pod image build train
  remote-pod image build dev
  remote-pod image build agent --base ubuntu:24.04 --tag remote-pod:agent-test     a local test image`,
		Args: cobra.MaximumNArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			cfg, err := config.Load()
			if err != nil {
				return err
			}
			kind := "train"
			if len(args) == 1 {
				kind = args[0]
			}
			ref := imageRef(cfg, kind)
			if tag != "" {
				ref = tag
			}
			root := config.ProjectRoot()
			if kind == "train" {
				if _, err := os.Stat(filepath.Join(root, "engine", "build", "dist", "lbc")); err != nil {
					return hint(fmt.Errorf("engine/build/dist/lbc is missing"), "remote-pod build")
				}
			}
			// the agent: a static Linux binary, built here
			gb := exec.Command("go", "build", "-trimpath", "-ldflags", "-s -w", "-o", "dist/remote-pod", ".")
			gb.Dir = filepath.Join(root, "tools", "remote-pod")
			gb.Env = append(os.Environ(), "CGO_ENABLED=0", "GOOS=linux", "GOARCH=amd64")
			gb.Stdout, gb.Stderr = os.Stderr, os.Stderr
			if err := gb.Run(); err != nil {
				return fmt.Errorf("building the agent: %w", err)
			}
			file, target := "train.Dockerfile", kind
			if kind == "dev" {
				file, target = "dev.Dockerfile", ""
			}
			dargs := []string{"build", "--network", "host", "-f", filepath.Join(root, "docker", file), "-t", ref}
			tagged := ref + "-" + gitSHA()
			if tag == "" {
				dargs = append(dargs, "-t", tagged)
			}
			if target != "" {
				dargs = append(dargs, "--target", target)
			}
			if base != "" {
				dargs = append(dargs, "--build-arg", "BASE="+base)
			}
			if err := run("docker", append(dargs, root)...); err != nil {
				return err
			}
			ui.Success("%s (also %s)", ref, tagged)
			return nil
		},
	}
	imageBuild.Flags().StringVar(&base, "base", "", "base image (default: nvidia/cuda runtime; ubuntu:24.04 for an agent test)")
	imageBuild.Flags().StringVar(&tag, "tag", "", "image reference (default: the configured image)")
	imagePush := &cobra.Command{
		Use:     "push [train|dev]",
		Short:   "Push the pod image to its registry (docker login first)",
		Example: "  remote-pod image push train",
		Args:    cobra.MaximumNArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			cfg, err := config.Load()
			if err != nil {
				return err
			}
			kind := "train"
			if len(args) == 1 {
				kind = args[0]
			}
			ref := imageRef(cfg, kind)
			for _, r := range []string{ref, ref + "-" + gitSHA()} {
				if err := run("docker", "push", r); err != nil {
					return hint(err, "docker login ghcr.io")
				}
			}
			ui.Success("pushed %s", ref)
			return nil
		},
	}
	image := &cobra.Command{Use: "image", Short: "The pod images: build, push", GroupID: "build"}
	image.AddCommand(imageBuild, imagePush)
	return []*cobra.Command{build, image}
}

func agentCommand() *cobra.Command {
	var addr, workdir string
	cmd := &cobra.Command{
		Use:   "agent",
		Short: "Run the agent (the pod image's entrypoint; settings from the environment)",
		Long: `The server inside the pod that sh, cp, sync, jobs and logs talk to. Settings from the environment, set by
remote-pod up / train: REMOTE_POD_TOKEN (required), REMOTE_POD_IDLE_MIN, REMOTE_POD_JOB (a run to execute),
REMOTE_POD_S3_* (R2), RUNPOD_POD_ID and RUNPOD_API_KEY (to stop the pod itself).`,
		Example: "  REMOTE_POD_TOKEN=secret remote-pod agent --addr 127.0.0.1:8080 --workdir /tmp/work   a local agent",
		GroupID: "internal",
		RunE: func(cmd *cobra.Command, args []string) error {
			idle, _ := strconv.Atoi(os.Getenv("REMOTE_POD_IDLE_MIN"))
			stopPod := func() {
				id, key := os.Getenv("RUNPOD_POD_ID"), os.Getenv("RUNPOD_API_KEY")
				if id == "" || key == "" {
					log.Printf("cannot stop the pod: RUNPOD_POD_ID / RUNPOD_API_KEY not set")
					return
				}
				if err := runpod.New(key).Stop(context.Background(), id); err != nil {
					log.Printf("stopping the pod: %v", err)
				}
			}
			srv, err := agent.New(agent.Options{Addr: addr, Token: os.Getenv("REMOTE_POD_TOKEN"), Workdir: workdir,
				IdleStop: time.Duration(idle) * time.Minute, OnIdle: stopPod})
			if err != nil {
				return err
			}
			if sc, ok := store.FromEnv(); ok {
				st, err := store.New(sc)
				if err != nil {
					return err
				}
				go srv.UploadJobLogs(context.Background(), st, 10*time.Second)
				if b64 := os.Getenv("REMOTE_POD_JOB"); b64 != "" {
					var spec agent.TrainSpec
					raw, err := base64.StdEncoding.DecodeString(b64)
					if err == nil {
						err = json.Unmarshal(raw, &spec)
					}
					if err != nil {
						return fmt.Errorf("REMOTE_POD_JOB: %w", err)
					}
					go func() {
						if err := srv.RunTrain(context.Background(), st, spec); err != nil {
							log.Printf("run %s failed before training: %v", spec.Run, err)
						}
					}()
				}
			}
			return srv.Serve()
		},
		PreRun: func(cmd *cobra.Command, args []string) {
			if workdir == "" {
				workdir = os.Getenv("REMOTE_POD_WORKDIR")
			}
		},
	}
	cmd.Flags().StringVar(&addr, "addr", fmt.Sprintf(":%d", config.AgentPort), "listen address")
	cmd.Flags().StringVar(&workdir, "workdir", "", "the working directory (default $REMOTE_POD_WORKDIR, else /work)")
	return cmd
}
