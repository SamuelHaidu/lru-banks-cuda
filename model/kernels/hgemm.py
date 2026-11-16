"""fp16 matmul on the Iris Xe (Xe-LP): an assembly kernel with fp16 products and f32 sums, about 1.9x torch's fp32
`mm` at the MLP's shapes.

`mm(a, b)` = a @ b for f32 2D tensors (any strides: transposed views cost nothing extra), result f32. Each operand is
packed to fp16 panels by one pass (`hgemm.sycl`); with `scale_a` / `scale_b` it is first scaled by a power of two
that brings its max |value| to 64, chosen on the device (gradients are far below fp16's normal range, 6e-5), and
the kernel divides the scales out. Products are exact in f32 terms up to the fp16 rounding of the inputs; sums are
in fp16 over 16 k steps, then in f32 (~6e-4 relative error at k = 768..3072, vs ~1e-2 for fp16-only sums).

Build (first use, cached in `.kernel_build/hgemm/`): `hgemm/gen_hgemm.py` prints the assembly, iga64 encodes it,
ocloc compiles the OpenCL stub `hgemm/hgemm_stub.cl` (its metadata: arguments, binding table), and the stub's code
section is swapped for the assembly, as tiny-lm does (`zebin_replace_text`). The zebin is loaded through Level Zero
into torch's SYCL context; the driver's loaded code is checked against the assembly. Needs `iga64` ($IGA64, PATH, or
~/projects/tiny-lm/.toolchain) and `ocloc` (the compute runtime's, same version as the driver).
"""

import hashlib
import os
import shutil
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

import torch

from model.kernels import rsm_scan

_HERE = Path(__file__).resolve().parent
_SRC = _HERE / "hgemm"
_ROOT = _HERE.parent.parent
KC, RB = 16, 16  # fp16 k steps per f32 flush, row panels per band
SCALE_TARGET = 64.0
_ext: Any = None
_error: Optional[str] = None


def _iga64() -> str:
    for c in (os.environ.get("IGA64"), shutil.which("iga64"),
              str(Path.home() / "projects/tiny-lm/.toolchain/igc-tools/usr/bin/iga64")):
        if c and Path(c).exists():
            return c
    raise RuntimeError("iga64 not found (set IGA64)")


def _replace_text(zebin: bytes, kernel: str, isa: bytes) -> bytes:
    """The zebin with section `.text.<kernel>` pointing at `isa`, appended (64-byte aligned); the rest untouched."""
    if zebin[:4] != b"\x7fELF":
        raise RuntimeError("ocloc output is not a zebin ELF")
    aligned = (len(zebin) + 63) & ~63
    out = bytearray(zebin) + bytes(aligned - len(zebin)) + isa
    shoff, = struct.unpack_from("<Q", out, 0x28)
    shentsize, shnum, shstrndx = struct.unpack_from("<HHH", out, 0x3A)
    strtab_off, = struct.unpack_from("<Q", out, shoff + shstrndx * shentsize + 0x18)
    want = f".text.{kernel}".encode()
    for i in range(shnum):
        sh = shoff + i * shentsize
        name_off, = struct.unpack_from("<I", out, sh)
        end = out.index(b"\0", strtab_off + name_off)
        if bytes(out[strtab_off + name_off:end]) == want:
            struct.pack_into("<QQ", out, sh + 0x18, aligned, len(isa))
            return bytes(out)
    raise RuntimeError(f"section .text.{kernel} not in the stub")


def build() -> tuple:
    """(zebin, isa) for the current sources, cached by their hash."""
    gen, stub = _SRC / "gen_hgemm.py", _SRC / "hgemm_stub.cl"
    ocloc = shutil.which("ocloc")
    if not ocloc:
        raise RuntimeError("ocloc not found")
    version = subprocess.run([ocloc, "--version"], capture_output=True, text=True).stdout.strip()
    key = hashlib.sha256(gen.read_bytes() + stub.read_bytes() + f"{KC} {RB} {version}".encode()).hexdigest()[:16]
    out = _ROOT / ".kernel_build" / "hgemm" / key
    if not (out / "hgemm_f16.zebin").exists():
        out.mkdir(parents=True, exist_ok=True)
        asm = subprocess.run([sys.executable, str(gen), "--kc", str(KC), "--rb", str(RB)], check=True,
                             capture_output=True, text=True).stdout
        (out / "hgemm_f16.iga").write_text(asm)
        subprocess.run([_iga64(), "-p=12p1", "-a", "-Xauto-deps", str(out / "hgemm_f16.iga"), "-o",
                        str(out / "hgemm_f16.bin")], check=True, capture_output=True)
        subprocess.run([ocloc, "compile", "-file", str(stub), "-device", "tgllp", "-options", "-cl-std=CL3.0",
                        "-output", "stub", "-output_no_suffix", "-out_dir", str(out)], check=True, capture_output=True)
        isa = (out / "hgemm_f16.bin").read_bytes()
        (out / "hgemm_f16.zebin").write_bytes(_replace_text((out / "stub.bin").read_bytes(), "hgemm_f16", isa))
    return (out / "hgemm_f16.zebin").read_bytes(), (out / "hgemm_f16.bin").read_bytes()


def load() -> Any:
    """The extension with the kernel loaded, or None (no XPU, no toolchain, `RSM_HGEMM=0`, a failed build)."""
    global _ext, _error
    if _ext is not None or _error is not None:
        return _ext
    try:
        if os.environ.get("RSM_HGEMM", "1") == "0":
            raise RuntimeError("disabled by RSM_HGEMM=0")
        if rsm_scan.load() is None:  # same toolchain setup (icpx on PATH, AOT target)
            raise RuntimeError(f"rsm_scan unavailable: {rsm_scan.error()}")
        if "Iris(R) Xe" not in torch.xpu.get_device_name(0):
            raise RuntimeError("the assembly is for Xe-LP (Iris Xe) only")
        zebin, isa = build()
        from torch.utils.cpp_extension import load as _load

        build_dir = _ROOT / ".kernel_build" / "hgemm_ext"
        build_dir.mkdir(parents=True, exist_ok=True)
        ext = _load(name="hgemm_ext", sources=[str(_HERE / "hgemm.sycl")], build_directory=str(build_dir),
                    extra_sycl_cflags=["-O3"], extra_ldflags=["-lze_loader"], verbose=False)
        loaded = ext.load_module(zebin, "hgemm_f16")
        if loaded[:len(isa)] != isa:
            raise RuntimeError("the driver did not load the assembly (it recompiled the stub)")
        _ext = ext
    except Exception as e:  # noqa: BLE001  any failure means: use torch's matmul
        _error = f"{type(e).__name__}: {e}"
    return _ext


def error() -> Optional[str]:
    return _error


def available(like: torch.Tensor) -> bool:
    return like.device.type == "xpu" and like.dtype == torch.float32 and load() is not None


def _pack(x: torch.Tensor, rows: int, cols: int, s_r: int, s_k: int, group: int, kp: int, scal: torch.Tensor,
          slot: int, scale: bool) -> torch.Tensor:
    if scale:
        # the max does not depend on the order: walk the memory order (a transposed view row by row would read
        # with a stride of a whole row, ~30x slower)
        if s_r < s_k:
            _ext.amax_scale(x, cols, rows, s_k, s_r, scal, slot, SCALE_TARGET)
        else:
            _ext.amax_scale(x, rows, cols, s_r, s_k, scal, slot, SCALE_TARGET)
    dst = torch.empty(((rows + group - 1) // group) * kp * group, device=x.device, dtype=torch.float16)
    if s_k == 1:  # contiguous along k (row-major A, B^T): the transposing pack, coalesced both ways
        _ext.pack_rows(x, dst, rows, cols, s_r, group, kp, scal, slot)
    else:  # contiguous along the panel's rows (A^T, row-major B): the element-wise pack reads coalesced
        _ext.pack(x, dst, rows, cols, s_r, s_k, group, kp, scal, slot)
    return dst


def ok(m: int, k: int, n: int) -> bool:
    """Shapes the kernel takes: n a multiple of 32, k and n below 65536."""
    return n % 32 == 0 and 0 < n < 65536 and 0 < k < 65536 - KC and m > 0


def mm(a: torch.Tensor, b: torch.Tensor, scale_a: bool = False, scale_b: bool = False, alpha: float = 1.0,
       out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """alpha a @ b, a `(m, k)`, b `(k, n)` f32 (any strides), into `out` `(m, n)` (new if None). No autograd."""
    (m, k), (k2, n) = a.shape, b.shape
    assert k == k2 and ok(m, k, n), (a.shape, b.shape)
    kp = (k + KC - 1) // KC * KC
    scal = torch.ones(16, device=a.device, dtype=torch.float32)
    pa = _pack(a, m, k, a.stride(0), a.stride(1), 16, kp, scal, 0, scale_a)
    pb = _pack(b, n, k, b.stride(1), b.stride(0), 32, kp, scal, 1, scale_b)
    if out is None:
        out = torch.empty(m, n, device=a.device, dtype=torch.float32)
    _ext.gemm(pa, pb, out, scal, m, n, kp, alpha)
    return out
