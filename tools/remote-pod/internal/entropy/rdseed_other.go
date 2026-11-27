//go:build !amd64

package entropy

func hasRDSEED() bool { return false }

func rdseed64() (uint64, bool) { return 0, false }
