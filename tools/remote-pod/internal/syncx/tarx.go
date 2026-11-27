// Package syncx moves files as tar streams (both directions of cp, and sync's changed files) and builds the hash
// manifests sync compares.
//
// A cp stream holds one file as the entry "." or a directory as "." plus entries relative to it; the receiver maps
// "." to the target path. A merge stream (sync) holds paths relative to the target directory. Entries that are
// absolute or climb out with ".." are refused.
package syncx

import (
	"archive/tar"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path"
	"path/filepath"
	"strings"
)

// DefaultExcludes are skipped by sync (and by cp of a directory): VCS data, caches, builds, environments.
var DefaultExcludes = []string{".git", "node_modules", "__pycache__", ".venv", ".mypy_cache", ".pytest_cache",
	"build", "dist", ".remote-pod", ".webchat", "runs"}

// Excluded reports whether a relative path has a component in excludes.
func Excluded(rel string, excludes []string) bool {
	for _, part := range strings.Split(filepath.ToSlash(rel), "/") {
		for _, e := range excludes {
			if ok, _ := path.Match(e, part); ok {
				return true
			}
		}
	}
	return false
}

// WriteTar writes src (a file or a directory) as a cp stream.
func WriteTar(w io.Writer, src string, excludes []string) error {
	tw := tar.NewWriter(w)
	info, err := os.Stat(src)
	if err != nil {
		return err
	}
	if !info.IsDir() {
		if err := addFile(tw, src, ".", info); err != nil {
			return err
		}
		return tw.Close()
	}
	err = filepath.WalkDir(src, func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		rel, _ := filepath.Rel(src, p)
		if rel != "." && Excluded(rel, excludes) {
			if d.IsDir() {
				return filepath.SkipDir
			}
			return nil
		}
		fi, err := d.Info()
		if err != nil {
			return err
		}
		if d.IsDir() {
			return tw.WriteHeader(&tar.Header{Typeflag: tar.TypeDir, Name: filepath.ToSlash(rel) + "/", Mode: int64(fi.Mode().Perm()), ModTime: fi.ModTime()})
		}
		if !fi.Mode().IsRegular() {
			return nil // links, sockets: skipped
		}
		return addFile(tw, p, filepath.ToSlash(rel), fi)
	})
	if err != nil {
		return err
	}
	return tw.Close()
}

// WriteTarFiles writes the given relative paths under root as a merge stream.
func WriteTarFiles(w io.Writer, root string, rels []string) error {
	tw := tar.NewWriter(w)
	for _, rel := range rels {
		p := filepath.Join(root, rel)
		fi, err := os.Stat(p)
		if err != nil {
			return err
		}
		if err := addFile(tw, p, filepath.ToSlash(rel), fi); err != nil {
			return err
		}
	}
	return tw.Close()
}

func addFile(tw *tar.Writer, p, name string, fi os.FileInfo) error {
	f, err := os.Open(p)
	if err != nil {
		return err
	}
	defer f.Close()
	if err := tw.WriteHeader(&tar.Header{Typeflag: tar.TypeReg, Name: name, Size: fi.Size(), Mode: int64(fi.Mode().Perm()), ModTime: fi.ModTime()}); err != nil {
		return err
	}
	_, err = io.Copy(tw, f)
	return err
}

// SafeRelPath reports whether a name from elsewhere (a store key's tail, a stream entry) is a plain relative path:
// not empty, not absolute, no ".." that climbs out, no backslash (a separator on Windows). Only such names may become
// local files: a pod with write access could otherwise name an object "runs/x/../../../.bashrc".
func SafeRelPath(rel string) bool {
	return rel != "" && !strings.Contains(rel, "\\") && filepath.IsLocal(filepath.FromSlash(rel))
}

// safeJoin joins a stream entry onto target, refusing entries that escape it.
func safeJoin(target, name string) (string, error) {
	name = strings.TrimSuffix(name, "/")
	if name == "." || name == "" {
		return target, nil
	}
	clean := path.Clean(name)
	if path.IsAbs(clean) || clean == ".." || strings.HasPrefix(clean, "../") {
		return "", fmt.Errorf("unsafe path in stream: %q", name)
	}
	return filepath.Join(target, filepath.FromSlash(clean)), nil
}

// Extract writes a stream under target ("." = target itself); returns the number of files and bytes written.
func Extract(r io.Reader, target string) (files int, bytes int64, err error) {
	tr := tar.NewReader(r)
	for {
		h, err := tr.Next()
		if errors.Is(err, io.EOF) {
			return files, bytes, nil
		}
		if err != nil {
			return files, bytes, err
		}
		dst, err := safeJoin(target, h.Name)
		if err != nil {
			return files, bytes, err
		}
		switch h.Typeflag {
		case tar.TypeDir:
			if err := os.MkdirAll(dst, 0o755); err != nil {
				return files, bytes, err
			}
		case tar.TypeReg:
			if err := os.MkdirAll(filepath.Dir(dst), 0o755); err != nil {
				return files, bytes, err
			}
			mode := os.FileMode(h.Mode).Perm()
			if mode == 0 {
				mode = 0o644
			}
			f, err := os.OpenFile(dst, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, mode)
			if err != nil {
				return files, bytes, err
			}
			n, err := io.Copy(f, tr)
			if cerr := f.Close(); err == nil {
				err = cerr
			}
			if err != nil {
				return files, bytes, err
			}
			_ = os.Chmod(dst, mode)
			_ = os.Chtimes(dst, h.ModTime, h.ModTime)
			files++
			bytes += n
		}
	}
}

// CpTarget is where a copy of src lands given dest: inside dest when dest is a directory (or ends with "/"),
// else dest itself (as with cp).
func CpTarget(dest, srcBase string, destIsDir bool) string {
	if destIsDir || strings.HasSuffix(dest, "/") {
		return filepath.Join(dest, srcBase)
	}
	return dest
}

// Manifest is the sha256 of every regular file under root (relative paths), minus excludes.
func Manifest(root string, excludes []string) (map[string]string, error) {
	out := map[string]string{}
	if _, err := os.Stat(root); errors.Is(err, os.ErrNotExist) {
		return out, nil
	}
	err := filepath.WalkDir(root, func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		rel, _ := filepath.Rel(root, p)
		if rel != "." && Excluded(rel, excludes) {
			if d.IsDir() {
				return filepath.SkipDir
			}
			return nil
		}
		if !d.Type().IsRegular() {
			return nil
		}
		h, err := hashFile(p)
		if err != nil {
			return err
		}
		out[filepath.ToSlash(rel)] = h
		return nil
	})
	return out, err
}

func hashFile(p string) (string, error) {
	f, err := os.Open(p)
	if err != nil {
		return "", err
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", err
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

// Diff is what to send (local files missing or different remotely) and what to delete (remote files not local).
func Diff(local, remote map[string]string) (send, del []string) {
	for p, h := range local {
		if remote[p] != h {
			send = append(send, p)
		}
	}
	for p := range remote {
		if _, ok := local[p]; !ok {
			del = append(del, p)
		}
	}
	return send, del
}
