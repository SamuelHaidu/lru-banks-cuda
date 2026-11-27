#include "textflag.h"

// func rdseed64() (v uint64, ok bool)
TEXT ·rdseed64(SB), NOSPLIT, $0-9
	BYTE $0x48; BYTE $0x0f; BYTE $0xc7; BYTE $0xf8 // RDSEED RAX
	SETCS ok+8(FP)
	MOVQ AX, v+0(FP)
	RET
