package security

// Every secret remote-pod makes comes from internal/entropy: the CPU's thermal-noise generator (RDSEED), the TPM
// (/dev/hwrng, when readable) and the kernel's CSPRNG, hashed together. Statistics cannot prove randomness; these
// tests show the sources are really used, the output has no gross defect, and no code bypasses the mixer.

import (
	"bytes"
	"compress/flate"
	"encoding/hex"
	"go/ast"
	"go/parser"
	"go/token"
	"io/fs"
	"math"
	"math/bits"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"golang.org/x/sys/cpu"

	"github.com/SamuelHaidu/rnn-research/tools/remote-pod/internal/entropy"
)

func TestHardwareSourcesAreUsed(t *testing.T) {
	src := entropy.Sources()
	t.Logf("sources on this machine: %v", src)
	if !slices.Contains(src, "getrandom") {
		t.Fatal("the kernel's CSPRNG is not used")
	}
	if cpu.X86.HasRDSEED && !slices.Contains(src, "rdseed") {
		t.Error("the CPU has RDSEED but it is not used (or failed its health checks)")
	}
}

// No gross defect in 1 MB of output: balanced bits, flat byte histogram, no repeated 32-byte block, incompressible.
func TestOutputHasNoGrossDefect(t *testing.T) {
	const n = 1 << 20
	b := entropy.Bytes(n)
	ones := 0
	var hist [256]int
	for _, x := range b {
		ones += bits.OnesCount8(x)
		hist[x]++
	}
	// monobit: 8n bits, sd = sqrt(8n)/2; 5 sd is a defect, not chance
	if dev := math.Abs(float64(ones) - 4*n); dev > 5*math.Sqrt(8*n)/2 {
		t.Errorf("monobit: %d ones in %d bits (%.0f off)", ones, 8*n, dev)
	}
	chi := 0.0
	exp := float64(n) / 256
	for _, c := range hist {
		chi += (float64(c) - exp) * (float64(c) - exp) / exp
	}
	if chi > 400 { // 255 degrees of freedom: p < 1e-8 beyond ~400
		t.Errorf("byte histogram chi-square %.0f", chi)
	}
	seen := map[string]bool{}
	for i := 0; i+32 <= n; i += 32 {
		k := string(b[i : i+32])
		if seen[k] {
			t.Fatal("a 32-byte block repeated")
		}
		seen[k] = true
	}
	var z bytes.Buffer
	w, _ := flate.NewWriter(&z, flate.BestCompression)
	_, _ = w.Write(b)
	_ = w.Close()
	if ratio := float64(z.Len()) / n; ratio < 0.99 {
		t.Errorf("compressed to %.3f of its size: not random", ratio)
	}
}

func TestTokensNeverCollide(t *testing.T) {
	seen := map[string]bool{}
	for i := 0; i < 200000; i++ {
		tok := hex.EncodeToString(entropy.Bytes(24))
		if seen[tok] {
			t.Fatal("a token repeated")
		}
		seen[tok] = true
	}
}

// Only internal/entropy reads a random source; no code generates a key from a source it passes in (key generation
// may ignore it: keys are derived from entropy seeds instead).
func TestNoCodeBypassesTheMixer(t *testing.T) {
	err := filepath.WalkDir("..", func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() && (d.Name() == "security" || d.Name() == "dist") {
			return filepath.SkipDir
		}
		if d.IsDir() || !strings.HasSuffix(p, ".go") || strings.HasSuffix(p, "_test.go") {
			return nil
		}
		f, err := parser.ParseFile(token.NewFileSet(), p, nil, parser.ImportsOnly)
		if err != nil {
			return err
		}
		inEntropy := strings.Contains(filepath.ToSlash(p), "internal/entropy/")
		for _, im := range f.Imports {
			path := strings.Trim(im.Path.Value, `"`)
			if path == "math/rand" || path == "math/rand/v2" {
				t.Errorf("%s imports %s (not for anything, in this tool)", p, path)
			}
			if path == "crypto/rand" && !inEntropy {
				t.Errorf("%s imports crypto/rand directly: use internal/entropy", p)
			}
		}
		full, err := parser.ParseFile(token.NewFileSet(), p, nil, 0)
		if err != nil {
			return err
		}
		ast.Inspect(full, func(n ast.Node) bool {
			if call, ok := n.(*ast.CallExpr); ok {
				if sel, ok := call.Fun.(*ast.SelectorExpr); ok && sel.Sel.Name == "GenerateKey" {
					t.Errorf("%s: a GenerateKey call (derive the key from an entropy seed instead)", p)
				}
			}
			return true
		})
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
}
