#!/usr/bin/env python3
"""hgemm_f16: C (f32) = alpha * A B, fp16 products on Xe-LP (tgllp), in assembly. Prints iga64 source.

Ported from tiny-lm's kernels/asm/matmul_f16.py (same scheme: one hardware thread per tile, A broadcast through a
register region, B a row of lanes, accumulators updated in place, a scalar `jmpi` loop), with what training needs:

- f32 accumulation: products are summed in half over KC k steps (the first step a `mul`, so no zeroing), then
  added into f32 accumulators by mixed-mode adds. Half-only sums over k = 768..32768 lose too much.
- a 16 x 32 tile: 32 half + 64 f32 accumulators, 4 A + 8 B registers (a 16 x 64 tile with f32 would need 192).
- any M (rows past `m` are not stored), N % 32 == 0, K % KC == 0 (the packing pads K with zeros).

Packed inputs (see `model/kernels/hgemm.py`): a[M/16][K][16] (one register per k step: lane = row) and
b[N/32][K][32] (two registers per k step). Surfaces: a = BTI 0, b = BTI 1, c = BTI 2, c row-major (ldc = n).

Rules from tiny-lm, measured on this GPU: src0 (accumulator) and src1 (B) of a mad on opposite even/odd registers
(~5%); a NoMask scalar loop (`jmpi`), not the compiled SIMD control flow with its fused-EU workaround.

    python gen_hgemm.py [--kc 16] > hgemm_f16.iga
"""

import argparse

# Payload (from hgemm_stub.cl's zeinfo: payload offset X is r(2 + X / 32), dword (X % 32) / 4).
# scal (arg 3) is stateless: its 64-bit address in r3.6-r3.7.
P_SCAL = "r3.6"
P_M, P_N, P_K, P_ALPHA, P_OFF_A, P_OFF_B = "r4.0", "r4.1", "r4.2", "r4.3", "r4.4", "r4.5"
GROUP = "r0.1"  # thread header: the work-group id (one 16-lane work-group per tile)

S, T = 5, 6  # scalars: S.0 m, S.1 n, S.2 k, S.3 alpha, S.4 row panel, S.5 column panel, S.6 loop count
HDR_A, HDR_B0, HDR_B1, HDR_C = 7, 8, 9, 10
FENCE = 11
HDR_S, SC = 120, 122  # the scales' A64 header, and r122-r123: scal[0..15] (scal[0] = sa, scal[1] = sb)
A = 12  # r12-r15: k step s in r12 + s, lane = row
B = 16  # r16-r23: k step s, column block c in r16 + 2 s + c (even base: column block 0 even, 1 odd)
H = 24  # r24-r55: half accumulators
F = 56  # r56-r119: f32 accumulators, row r in r56 + 4 r .. + 3 (columns 0-15, 16-31), one block write per row
ROWS, COLS, KSTEP_GROUP = 16, 2, 4
BTI_A, BTI_B, BTI_C = 0, 1, 2  # c is written through BTI 2, scal read stateless (A64)


def h_acc(row, col):
    """Opposite parity to B's column block (B + 2 s + col): column 0 odd, column 1 even."""
    return H + 2 * row + (1 - col)


def f_acc(row, col):
    return F + 4 * row + 2 * col


class Asm:
    def __init__(self):
        self.lines = []

    def label(self, name):
        self.lines.append(f"{name}:")

    def w(self, ins, pred=None):
        """NoMask instruction (scalar setup, sends), with an optional flag predicate."""
        prefix = f"(W&{pred})" if pred else "(W)"
        self.lines.append(f"{prefix:<16}{ins}")

    def v(self, ins):
        """A full-width vector instruction (every lane runs: the thread is one whole work-group)."""
        self.lines.append(f"{'':<16}{ins}")

    def read4(self, dst, hdr, bti, token):
        """4 registers (128 bytes) from surface bti, byte offset in hdr.2 (oword aligned)."""
        self.w(f"send.dc0 (16|M0) r{dst} r{hdr} null 0x0 0x{0x02484400 | bti:08X} {{${token}}}")

    def write4(self, hdr, src, bti, token, pred=None):
        """4 registers to surface bti, offset in owords in hdr.2."""
        self.w(f"send.dc0 (16|M0) null r{hdr} r{src} 0x100 0x{0x020A0400 | bti:08X} {{${token}}}", pred)

    def print(self):
        print("\n".join(self.lines))


def generate(kc: int = 16, rb: int = 16) -> Asm:
    if kc % KSTEP_GROUP or kc & (kc - 1):
        raise ValueError("kc must be a power of two, at least 4")
    p = Asm()
    p.w("or (1|M0) cr0.0<1>:ud cr0.0<0;1,0>:ud 0x4C0:uw")  # fp mode (as tiny-lm): keep denormals

    # Scalars out of the payload (r1-r4 are free afterwards).
    p.w(f"mov (8|M0) r{S}.0<1>:ud 0x0:ud")
    p.w(f"mov (8|M0) r{T}.0<1>:ud 0x0:ud")
    p.w(f"mov (1|M0) r{S}.0<1>:ud {P_M}<0;1,0>:ud")
    p.w(f"mov (1|M0) r{S}.1<1>:ud {P_N}<0;1,0>:ud")
    p.w(f"mov (1|M0) r{S}.2<1>:ud {P_K}<0;1,0>:ud")
    p.w(f"mov (1|M0) r{S}.3<1>:f {P_ALPHA}<0;1,0>:f")
    p.w(f"mov (1|M0) r{T}.4<1>:ud {P_OFF_A}<0;1,0>:ud")
    p.w(f"mov (1|M0) r{T}.5<1>:ud {P_OFF_B}<0;1,0>:ud")
    for hdr in (HDR_A, HDR_B0, HDR_B1, HDR_C, HDR_S):
        p.w(f"mov (8|M0) r{hdr}.0<1>:ud 0x0:ud")
    # alpha / (sa sb): the packing's power-of-two scales, read from the device (no host sync); used in the epilogue
    p.w(f"mov (2|M0) r{HDR_S}.0<1>:ud {P_SCAL}<2;2,1>:ud")
    p.w(f"send.dc1 (16|M0) r{SC} r{HDR_S} null 0x0 0x022D0BFF {{$6}}")  # a64 aligned oword block read x4

    # Tile, in bands of rb row panels: within a band, consecutive groups share a column panel (B) across the band's
    # rows, so B is streamed once per band (not once per row panel) and the band's A panels stay in cache.
    #   P = n / 32, R = ceil(m / 16), band = g / (rb P), i = g % (rb P), rows = min(rb, R - band rb),
    #   column panel = i / rows, row panel = band rb + i % rows
    p.w(f"shr (1|M0) r{T}.0<1>:ud r{S}.1<0;1,0>:ud 5:w")  # P
    p.w(f"add (1|M0) r{T}.1<1>:ud r{S}.0<0;1,0>:ud 15:w")
    p.w(f"shr (1|M0) r{T}.1<1>:ud r{T}.1<0;1,0>:ud 4:w")  # R
    p.w(f"mul (1|M0) r{T}.2<1>:d r{T}.0<0;1,0>:d {rb}:w")  # rb P
    p.w(f"math.iqot (1|M0) r{T}.3<1>:ud {GROUP}<0;1,0>:ud r{T}.2<0;1,0>:ud")  # band
    p.w(f"math.irem (1|M0) r{T}.6<1>:ud {GROUP}<0;1,0>:ud r{T}.2<0;1,0>:ud")  # i
    p.w(f"mul (1|M0) r{T}.3<1>:d r{T}.3<0;1,0>:d {rb}:w")  # band rb: the band's first row panel
    p.w(f"add (1|M0) r{T}.7<1>:d r{T}.1<0;1,0>:d -r{T}.3<0;1,0>:d")  # R - band rb
    p.w(f"sel (1|M0) (lt)f0.0 r{T}.7<1>:ud r{T}.7<0;1,0>:ud {rb}:w")  # rows in this band
    p.w(f"math.iqot (1|M0) r{S}.5<1>:ud r{T}.6<0;1,0>:ud r{T}.7<0;1,0>:ud")  # column panel
    p.w(f"math.irem (1|M0) r{S}.4<1>:ud r{T}.6<0;1,0>:ud r{T}.7<0;1,0>:ud")
    p.w(f"add (1|M0) r{S}.4<1>:ud r{S}.4<0;1,0>:ud r{T}.3<0;1,0>:ud")  # row panel
    # Panel byte offsets: a = row panel * k * 16 * 2, b = column panel * k * 32 * 2 (k < 65536: a uw operand).
    k_uw = f"r{S}.4<0;1,0>:uw"  # S.2 (k) as a word: word index 2 * 2
    n_uw = f"r{S}.2<0;1,0>:uw"  # S.1 (n) as a word: word index 2 * 1
    p.w(f"mul (1|M0) r{T}.1<1>:d r{S}.4<0;1,0>:d {k_uw}")
    p.w(f"shl (1|M0) r{T}.1<1>:ud r{T}.1<0;1,0>:ud 5:w")
    p.w(f"add (1|M0) r{HDR_A}.2<1>:ud r{T}.1<0;1,0>:ud r{T}.4<0;1,0>:ud")
    p.w(f"mul (1|M0) r{T}.2<1>:d r{S}.5<0;1,0>:d {k_uw}")
    p.w(f"shl (1|M0) r{T}.2<1>:ud r{T}.2<0;1,0>:ud 6:w")
    p.w(f"add (1|M0) r{HDR_B0}.2<1>:ud r{T}.2<0;1,0>:ud r{T}.5<0;1,0>:ud")
    p.w(f"add (1|M0) r{HDR_B1}.2<1>:ud r{HDR_B0}.2<0;1,0>:ud 128:w")
    # Loop count: k / kc chunks.
    p.w(f"shr (1|M0) r{S}.6<1>:ud r{S}.2<0;1,0>:ud {kc.bit_length() - 1}:w")

    for row in range(ROWS):
        for col in range(COLS):
            p.v(f"mov (16|M0) r{f_acc(row, col)}.0<1>:f 0x0:f")

    p.label("LOOP")
    for g in range(kc // KSTEP_GROUP):
        p.read4(A, HDR_A, BTI_A, token=0)
        p.read4(B, HDR_B0, BTI_B, token=1)
        p.read4(B + 4, HDR_B1, BTI_B, token=2)
        p.w(f"add (1|M0) r{HDR_A}.2<1>:ud r{HDR_A}.2<0;1,0>:ud 128:w")
        p.w(f"add (1|M0) r{HDR_B0}.2<1>:ud r{HDR_B0}.2<0;1,0>:ud 256:w")
        p.w(f"add (1|M0) r{HDR_B1}.2<1>:ud r{HDR_B1}.2<0;1,0>:ud 256:w")
        for step in range(KSTEP_GROUP):
            first = g == 0 and step == 0
            for row in range(ROWS):
                for col in range(COLS):
                    acc, b = h_acc(row, col), B + 2 * step + col
                    assert (acc ^ b) & 1, "accumulator and B on the same parity"
                    if first:  # the chunk's first product: no zeroing of the half accumulators
                        p.v(f"mul (16|M0) r{acc}.0<1>:hf r{b}.0<16;16,1>:hf r{A + step}.{row}<0;1,0>:hf")
                    else:
                        p.v(f"mad (16|M0) r{acc}.0<1>:hf r{acc}.0<8;1>:hf r{b}.0<8;1>:hf r{A + step}.{row}<0>:hf")
    # The chunk's half sums into f32 (mixed mode: two SIMD8 halves).
    for row in range(ROWS):
        for col in range(COLS):
            h, f = h_acc(row, col), f_acc(row, col)
            p.v(f"add (8|M0) r{f}.0<1>:f r{f}.0<8;8,1>:f r{h}.0<8;8,1>:hf")
            p.v(f"add (8|M8) r{f + 1}.0<1>:f r{f + 1}.0<8;8,1>:f r{h}.8<8;8,1>:hf")
    p.w(f"add (1|M0) (ne)f0.0 r{S}.6<1>:d r{S}.6<0;1,0>:d -1:w")
    p.w("jmpi (1|M0) LOOP", pred="f0.0")

    # Epilogue: C = alpha / (sa sb) * acc, row by row, rows past m skipped.
    p.w(f"mul (1|M0) r{SC}.0<1>:f r{SC}.0<0;1,0>:f r{SC}.1<0;1,0>:f")
    p.w(f"math.inv (1|M0) r{SC}.0<1>:f r{SC}.0<0;1,0>:f")
    p.w(f"mul (1|M0) r{S}.3<1>:f r{S}.3<0;1,0>:f r{SC}.0<0;1,0>:f")
    for row in range(ROWS):
        for col in range(COLS):
            p.v(f"mul (16|M0) r{f_acc(row, col)}.0<1>:f r{f_acc(row, col)}.0<8;8,1>:f r{S}.3<0;1,0>:f")
    p.w(f"shl (1|M0) r{T}.0<1>:ud r{S}.4<0;1,0>:ud 4:w")  # first row of the tile
    p.w(f"add (1|M0) r{T}.3<1>:d r{S}.0<0;1,0>:d -r{T}.0<0;1,0>:d")  # rows left: m - first row
    p.w(f"mul (1|M0) r{T}.1<1>:d r{T}.0<0;1,0>:d {n_uw}")  # first row * n
    p.w(f"shl (1|M0) r{T}.2<1>:ud r{S}.5<0;1,0>:ud 5:w")  # first column
    p.w(f"add (1|M0) r{T}.1<1>:ud r{T}.1<0;1,0>:ud r{T}.2<0;1,0>:ud")
    p.w(f"shr (1|M0) r{HDR_C}.2<1>:ud r{T}.1<0;1,0>:ud 2:w")  # element offset (f32) in owords
    p.w(f"shr (1|M0) r{T}.6<1>:ud r{S}.1<0;1,0>:ud 2:w")  # owords per row: n / 4
    for row in range(ROWS):
        # all 16 flag bits alike (a SIMD16 block write is predicated per channel)
        p.w(f"cmp (16|M0) (gt)f1.0 null<1>:d r{T}.3<0;1,0>:d {row}:w")
        p.write4(HDR_C, f_acc(row, 0), BTI_C, token=3 + row % 2, pred="f1.0")
        if row < ROWS - 1:
            p.w(f"add (1|M0) r{HDR_C}.2<1>:ud r{HDR_C}.2<0;1,0>:ud r{T}.6<0;1,0>:ud")
    p.w(f"send.dc0 (8|M0) r{FENCE} r0 null 0x0 0x0219E000 {{$5}}")  # make the writes visible, wait for it
    p.w(f"mov (8|M0) null<1>:ud r{FENCE}.0<8;8,1>:ud")
    p.w("mov (8|M0) r127.0<1>:f r0.0<8;8,1>:f")
    p.w("send.ts (1|M0) null r127 null 0x0 0x02000010 {EOT}")
    return p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kc", type=int, default=16, help="k steps summed in half before the f32 flush")
    ap.add_argument("--rb", type=int, default=16, help="row panels per band (tile order)")
    a = ap.parse_args()
    generate(a.kc, a.rb).print()


if __name__ == "__main__":
    main()
