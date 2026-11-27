// Package entropy is where every secret remote-pod makes gets its randomness (tokens, pod tags, job ids, the TLS key
// of a pod's certificate, each connection's SSH key). Each read hashes together:
//
//  1. the CPU's hardware generator, RDSEED (Intel / AMD: a circuit sampling thermal noise, conditioned on-chip),
//     health-checked: a stuck or repeating source (AMD's RDRAND once returned all ones after a resume) is dropped;
//  2. the kernel's hardware RNG device, /dev/hwrng (here: the TPM), when this user may read it;
//  3. the kernel's CSPRNG, getrandom(2) (crypto/rand), which itself mixes RDRAND, the TPM and interrupt timing.
//
// The hardware is read first and the kernel last, and SHA-512 combines them: the output is unpredictable if any one
// source is, and a dishonest hardware generator cannot adapt its bytes to the kernel's (they come after). Hardware is
// never trusted alone. If every hardware source is unavailable, the kernel's CSPRNG still is (that is crypto/rand).
package entropy

import (
	"bytes"
	"crypto/rand"
	"crypto/sha512"
	"encoding/binary"
	"os"
	"sync"
	"sync/atomic"
	"time"
)

// rdseedFn is the CPU source (a variable so the health checks can be tested with a broken one).
var rdseedFn = rdseed64

var (
	mu       sync.Mutex
	counter  atomic.Uint64
	lastSeed uint64
	cpuOK    = hasRDSEED()
	hwrngOK  = true
)

// Read fills b with mixed randomness. It panics only if the kernel's CSPRNG fails (crypto/rand's own contract).
func Read(b []byte) {
	for off := 0; off < len(b); off += sha512.Size {
		block := mix()
		copy(b[off:], block[:])
	}
}

// Bytes returns n mixed random bytes.
func Bytes(n int) []byte {
	b := make([]byte, n)
	Read(b)
	return b
}

// Reader is an io.Reader over Read (for APIs that take one).
var Reader = reader{}

type reader struct{}

func (reader) Read(b []byte) (int, error) {
	Read(b)
	return len(b), nil
}

// Sources reports which sources the last reads used ("rdseed", "hwrng", "getrandom").
func Sources() []string {
	mu.Lock()
	defer mu.Unlock()
	s := []string{}
	if cpuOK {
		s = append(s, "rdseed")
	}
	if hwrngOK && canReadHWRNG() {
		s = append(s, "hwrng")
	}
	return append(s, "getrandom")
}

func mix() [sha512.Size]byte {
	h := sha512.New()
	h.Write([]byte("remote-pod entropy v1\x00"))
	var ctr [16]byte
	binary.LittleEndian.PutUint64(ctr[:], counter.Add(1))
	binary.LittleEndian.PutUint64(ctr[8:], uint64(time.Now().UnixNano()))
	h.Write(ctr[:])
	// 1. the CPU's thermal-noise generator (before the kernel's bytes exist)
	if hw := cpuSeed(8); hw != nil {
		h.Write(hw)
	}
	// 2. the hardware RNG device (TPM), if this process may read it
	if hw := hwrng(32); hw != nil {
		h.Write(hw)
	}
	// 3. the kernel's CSPRNG, last
	k := make([]byte, 64)
	if _, err := rand.Read(k); err != nil {
		panic("entropy: getrandom failed: " + err.Error())
	}
	h.Write(k)
	var out [sha512.Size]byte
	copy(out[:], h.Sum(nil))
	return out
}

// cpuSeed reads n 64-bit RDSEED values, or nil when the CPU has none or the source fails its health checks.
func cpuSeed(n int) []byte {
	mu.Lock()
	defer mu.Unlock()
	if !cpuOK {
		return nil
	}
	out := make([]byte, 0, 8*n)
	for i := 0; i < n; i++ {
		var v uint64
		ok := false
		for try := 0; try < 100 && !ok; try++ { // RDSEED may be momentarily empty: retry, as Intel advises
			v, ok = rdseedFn()
		}
		// health: a failing, stuck (all zeros / all ones) or repeating source is dropped for good
		if !ok || v == 0 || v == ^uint64(0) || v == lastSeed {
			cpuOK = false
			return nil
		}
		lastSeed = v
		out = binary.LittleEndian.AppendUint64(out, v)
	}
	return out
}

func canReadHWRNG() bool {
	f, err := os.Open("/dev/hwrng")
	if err != nil {
		return false
	}
	f.Close()
	return true
}

// hwrng reads n bytes from /dev/hwrng, or nil (no device, no permission: it is root's on most systems, and the kernel
// already feeds it into getrandom).
func hwrng(n int) []byte {
	mu.Lock()
	ok := hwrngOK
	mu.Unlock()
	if !ok {
		return nil
	}
	f, err := os.Open("/dev/hwrng")
	if err != nil {
		mu.Lock()
		hwrngOK = false
		mu.Unlock()
		return nil
	}
	defer f.Close()
	b := make([]byte, n)
	done := make(chan error, 1)
	go func() { _, err := f.Read(b); done <- err }()
	select {
	case err := <-done:
		if err != nil || bytes.Equal(b, make([]byte, n)) {
			return nil
		}
		return b
	case <-time.After(200 * time.Millisecond): // a slow device must not stall key generation
		return nil
	}
}
