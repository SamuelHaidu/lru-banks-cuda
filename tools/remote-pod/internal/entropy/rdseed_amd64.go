package entropy

import "golang.org/x/sys/cpu"

func hasRDSEED() bool { return cpu.X86.HasRDSEED }

// rdseed64 executes RDSEED RAX: ok reports the carry flag (a value was available).
func rdseed64() (v uint64, ok bool)
