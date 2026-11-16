#!/usr/bin/env bash
# AOT-compiles model/kernels/rsm_scan.sycl for Tiger Lake Xe-LP (tgllp) and disassembles every kernel with iga64.
#   bash "ad hoc scripts/disasm_rsm_kernels.sh" [out_dir]     -> <out_dir>/iga/<kernel>.asm + a summary
# iga64: $IGA64, else on PATH, else ~/projects/tiny-lm/.toolchain/igc-tools/usr/bin/iga64. ocloc dumps the zebin's
# raw .text sections (and its ze_info: SIMD width, GRF count, spills); iga64 -p=12p1 disassembles them.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${1:-/tmp/rsm_disasm/aot}"
B=/opt/intel/oneapi/compiler/2025.1
TI="$ROOT/.venv/lib/python3.13/site-packages/torch/include"
PYINC="$("$ROOT/.venv/bin/python" -c "import sysconfig; print(sysconfig.get_paths()['include'])")"
rm -rf "$OUT" && mkdir -p "$OUT" && cd "$OUT"
"$B/bin/icpx" -fsycl -fsycl-targets=spir64_gen -Xs "-device tgllp" -save-temps -O3 -ffast-math -std=c++20 -fPIC \
  -DTORCH_EXTENSION_NAME=rsm_scan -DTORCH_API_INCLUDE_EXTENSION_H -isystem "$TI" -isystem "$TI/torch/csrc/api/include" \
  -isystem "$B/include" -isystem "$B/include/sycl" -isystem "$PYINC" -x c++ "$ROOT/model/kernels/rsm_scan.sycl" \
  -shared -o k.so 2>&1 | grep -E " error" || true
IGA="${IGA64:-$(command -v iga64 || echo "$HOME/projects/tiny-lm/.toolchain/igc-tools/usr/bin/iga64")}"
[ -x "$IGA" ] || { echo "iga64 not found (set IGA64)"; exit 1; }
mkdir -p iga
for f in $(file *.out | grep ELF | cut -d: -f1); do
  ocloc disasm -file "$f" -dump "dis_$f" -device tgllp -skip-asm-translation > /dev/null 2>&1 || true
  for k in "dis_$f"/.text.*; do
    [ -f "$k" ] || continue
    name="$(basename "$k" | sed 's/^\.text\._ZTSN12_GLOBAL__N_1[0-9]*//')"
    "$IGA" -d -p=12p1 "$k" -o "iga/$name.asm"
  done
done
for info in dis_*/.ze_info; do
  awk '/^  - name:/{name=$3} /simd_size|grf_count|slm_size|barrier_count/{printf "%s %s %s\n", name, $1, $2}
       /scratch|spill/{printf "%s SPILL/SCRATCH %s\n", name, $0}' "$info"
done
for asm in iga/*.asm; do
  echo "$(basename "$asm"): $(grep -cE '\bsend' "$asm") sends, $(grep -cE '\b(mad|mul|add)\b.*:f' "$asm") fp ops"
done
