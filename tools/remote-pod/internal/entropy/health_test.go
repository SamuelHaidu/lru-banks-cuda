package entropy

import "testing"

// A broken CPU generator is dropped, and Read keeps working on the other sources.
func TestBrokenCPUSourceIsDropped(t *testing.T) {
	for name, src := range map[string]func() (uint64, bool){
		"all ones (AMD's RDRAND bug after resume)": func() (uint64, bool) { return ^uint64(0), true },
		"all zeros":   func() (uint64, bool) { return 0, true },
		"stuck value": func() (uint64, bool) { return 0x1234567890abcdef, true },
		"never ready": func() (uint64, bool) { return 0, false },
	} {
		old := rdseedFn
		rdseedFn, cpuOK, lastSeed = src, true, 0
		_ = Bytes(64)
		if cpuOK {
			t.Errorf("%s: still trusted", name)
		}
		if a, b := Bytes(32), Bytes(32); string(a) == string(b) {
			t.Errorf("%s: Read repeated itself without the CPU source", name)
		}
		rdseedFn, cpuOK, lastSeed = old, hasRDSEED(), 0
	}
}
