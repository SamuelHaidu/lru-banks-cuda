// Package store is the S3-compatible object store (Cloudflare R2) for datasets, checkpoints and logs.
//
//	datasets/<name>/data.tar.zst + manifest.json     a dataset (a directory, tar + zstd, streamed)
//	runs/<run>/...                                  a run's config, metrics, log, checkpoints
//	jobs/<id>.log                                   a detached job's output
package store

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/klauspost/compress/zstd"
	"github.com/minio/minio-go/v7"
	"github.com/minio/minio-go/v7/pkg/credentials"

	"github.com/SamuelHaidu/rnn-research/tools/remote-pod/internal/syncx"
	"github.com/SamuelHaidu/rnn-research/tools/remote-pod/internal/tlsroots"
)

type Config struct {
	Endpoint, Bucket, AccessKey, SecretKey, Region string
}

// Env names the store's settings take inside a pod (the CLI passes them at creation).
var Env = struct{ Endpoint, Bucket, AccessKey, SecretKey, Region string }{
	"REMOTE_POD_S3_ENDPOINT", "REMOTE_POD_S3_BUCKET", "REMOTE_POD_S3_ACCESS_KEY", "REMOTE_POD_S3_SECRET_KEY",
	"REMOTE_POD_S3_REGION"}

func FromEnv() (Config, bool) {
	c := Config{os.Getenv(Env.Endpoint), os.Getenv(Env.Bucket), os.Getenv(Env.AccessKey), os.Getenv(Env.SecretKey),
		os.Getenv(Env.Region)}
	return c, c.Endpoint != "" && c.Bucket != ""
}

func (c Config) EnvVars() map[string]string {
	return map[string]string{Env.Endpoint: c.Endpoint, Env.Bucket: c.Bucket, Env.AccessKey: c.AccessKey,
		Env.SecretKey: c.SecretKey, Env.Region: c.Region}
}

type Store struct {
	mc     *minio.Client
	bucket string
}

func New(c Config) (*Store, error) {
	u, err := url.Parse(c.Endpoint)
	if err != nil || u.Host == "" {
		return nil, fmt.Errorf("store: bad endpoint %q (want https://<account>.r2.cloudflarestorage.com)", c.Endpoint)
	}
	region := c.Region
	if region == "" {
		region = "auto"
	}
	// the embedded roots only: a CA injected into this machine's store cannot read the key's requests (tlsroots)
	mc, err := minio.New(u.Host, &minio.Options{Creds: credentials.NewStaticV4(c.AccessKey, c.SecretKey, ""),
		Secure: u.Scheme == "https", Region: region, Transport: tlsroots.Transport()})
	if err != nil {
		return nil, err
	}
	return &Store{mc: mc, bucket: c.Bucket}, nil
}

type Object struct {
	Key      string    `json:"key"`
	Size     int64     `json:"size"`
	Modified time.Time `json:"modified"`
}

func (s *Store) List(ctx context.Context, prefix string, recursive bool) ([]Object, error) {
	var out []Object
	for o := range s.mc.ListObjects(ctx, s.bucket, minio.ListObjectsOptions{Prefix: prefix, Recursive: recursive}) {
		if o.Err != nil {
			return nil, o.Err
		}
		out = append(out, Object{Key: o.Key, Size: o.Size, Modified: o.LastModified})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Key < out[j].Key })
	return out, nil
}

func (s *Store) Exists(ctx context.Context, key string) bool {
	_, err := s.mc.StatObject(ctx, s.bucket, key, minio.StatObjectOptions{})
	return err == nil
}

// Presign is a URL that downloads key without credentials until ttl passes (the URL carries a signature, not the key).
func (s *Store) Presign(ctx context.Context, key string, ttl time.Duration) (string, error) {
	return s.PresignGet(ctx, key, ttl)
}

// PresignGet: a URL that downloads key, and only key, until ttl passes.
func (s *Store) PresignGet(ctx context.Context, key string, ttl time.Duration) (string, error) {
	u, err := s.mc.PresignedGetObject(ctx, s.bucket, key, ttl, nil)
	if err != nil {
		return "", err
	}
	return u.String(), nil
}

// PresignPut: a URL that uploads (HTTP PUT) to key, and only key, until ttl passes. It cannot read, list or delete.
// size > 0 signs the Content-Length as well: the URL then takes exactly size bytes and no other length.
func (s *Store) PresignPut(ctx context.Context, key string, ttl time.Duration, size int64) (string, error) {
	if size > 0 {
		u, err := s.mc.PresignHeader(ctx, http.MethodPut, s.bucket, key, ttl, nil,
			http.Header{"Content-Length": {strconv.FormatInt(size, 10)}})
		if err != nil {
			return "", err
		}
		return u.String(), nil
	}
	u, err := s.mc.PresignedPutObject(ctx, s.bucket, key, ttl)
	if err != nil {
		return "", err
	}
	return u.String(), nil
}

// Remove deletes one object (this machine only: a pod is never granted a delete).
func (s *Store) Remove(ctx context.Context, key string) error {
	return s.mc.RemoveObject(ctx, s.bucket, key, minio.RemoveObjectOptions{})
}

func (s *Store) PutFile(ctx context.Context, key, path string) error {
	_, err := s.mc.FPutObject(ctx, s.bucket, key, path, minio.PutObjectOptions{})
	return err
}

func (s *Store) GetFile(ctx context.Context, key, path string) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	return s.mc.FGetObject(ctx, s.bucket, key, path, minio.GetObjectOptions{})
}

func (s *Store) PutBytes(ctx context.Context, key string, b []byte) error {
	_, err := s.mc.PutObject(ctx, s.bucket, key, strings.NewReader(string(b)), int64(len(b)), minio.PutObjectOptions{})
	return err
}

func (s *Store) GetBytes(ctx context.Context, key string) ([]byte, error) {
	o, err := s.mc.GetObject(ctx, s.bucket, key, minio.GetObjectOptions{})
	if err != nil {
		return nil, err
	}
	defer o.Close()
	return io.ReadAll(o)
}

// PutDir uploads a directory as one tar.zst object, streamed (no temporary file); returns the compressed size.
func (s *Store) PutDir(ctx context.Context, key, dir string, progress func(int64)) (int64, error) {
	pr, pw := io.Pipe()
	go func() {
		zw, err := zstd.NewWriter(pw, zstd.WithEncoderLevel(zstd.SpeedDefault))
		if err != nil {
			pw.CloseWithError(err)
			return
		}
		err = syncx.WriteTar(&progressWriter{w: zw, f: progress}, dir, nil)
		if cerr := zw.Close(); err == nil {
			err = cerr
		}
		pw.CloseWithError(err)
	}()
	// unsigned payload: no aws-chunked framing (which some S3-compatible stores keep as data)
	info, err := s.mc.PutObject(ctx, s.bucket, key, pr, -1, minio.PutObjectOptions{PartSize: 64 << 20, DisableContentSha256: true})
	return info.Size, err
}

// GetDir downloads a tar.zst object into dir, streamed.
func (s *Store) GetDir(ctx context.Context, key, dir string, progress func(int64)) (files int, bytes int64, err error) {
	o, err := s.mc.GetObject(ctx, s.bucket, key, minio.GetObjectOptions{})
	if err != nil {
		return 0, 0, err
	}
	defer o.Close()
	zr, err := zstd.NewReader(&progressReader{r: o, f: progress})
	if err != nil {
		return 0, 0, err
	}
	defer zr.Close()
	return syncx.Extract(zr, dir)
}

// Dataset is a dataset's manifest (datasets/<name>/manifest.json).
type Dataset struct {
	Name       string    `json:"name"`
	Source     string    `json:"source"`
	Created    time.Time `json:"created"`
	Files      int       `json:"files"`
	Bytes      int64     `json:"bytes"`      // uncompressed
	Compressed int64     `json:"compressed"` // in the store
	Key        string    `json:"key"`        // the archive's key (or an external s3:// path)
}

func DatasetKey(name string) string      { return "datasets/" + name + "/data.tar.zst" }
func datasetManifest(name string) string { return "datasets/" + name + "/manifest.json" }

func (s *Store) PutDatasetManifest(ctx context.Context, d Dataset) error {
	b, _ := json.MarshalIndent(d, "", "  ")
	return s.PutBytes(ctx, datasetManifest(d.Name), b)
}

func (s *Store) Dataset(ctx context.Context, name string) (Dataset, error) {
	var d Dataset
	b, err := s.GetBytes(ctx, datasetManifest(name))
	if err != nil {
		return d, fmt.Errorf("no dataset %q in the store (add it: remote-pod dataset add --name %s --from DIR): %w", name, name, err)
	}
	return d, json.Unmarshal(b, &d)
}

func (s *Store) Datasets(ctx context.Context) ([]Dataset, error) {
	objs, err := s.List(ctx, "datasets/", true)
	if err != nil {
		return nil, err
	}
	var out []Dataset
	for _, o := range objs {
		if strings.HasSuffix(o.Key, "/manifest.json") {
			name := strings.TrimSuffix(strings.TrimPrefix(o.Key, "datasets/"), "/manifest.json")
			if d, err := s.Dataset(ctx, name); err == nil {
				out = append(out, d)
			}
		}
	}
	return out, nil
}

type progressWriter struct {
	w io.Writer
	n int64
	f func(int64)
}

func (p *progressWriter) Write(b []byte) (int, error) {
	n, err := p.w.Write(b)
	p.n += int64(n)
	if p.f != nil {
		p.f(p.n)
	}
	return n, err
}

type progressReader struct {
	r io.Reader
	n int64
	f func(int64)
}

func (p *progressReader) Read(b []byte) (int, error) {
	n, err := p.r.Read(b)
	p.n += int64(n)
	if p.f != nil {
		p.f(p.n)
	}
	return n, err
}
