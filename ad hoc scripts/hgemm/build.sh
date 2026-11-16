#!/usr/bin/env bash
# Builds hgemm.cpp AOT for tgllp, disassembles every kernel (iga64 -p=12p1) and prints per kernel: GRFs, spill
# bytes, the math / move counts by width and type, and the memory messages. Then runs the benchmark.
#   bash "ad hoc scripts/hgemm/build.sh" [--no-run]
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
OUT=/tmp/hgemm_build
IGA="${IGA64:-$(command -v iga64 || echo "$HOME/projects/tiny-lm/.toolchain/igc-tools/usr/bin/iga64")}"
rm -rf "$OUT" && mkdir -p "$OUT" && cd "$OUT"
/opt/intel/oneapi/compiler/2025.1/bin/icpx -fsycl -fsycl-targets=spir64_gen -Xs "-device tgllp" -O3 -std=c++20 \
  -Wno-deprecated-declarations -save-temps "$HERE/hgemm.cpp" -o hgemm 2>&1 | grep -E " error" || true
for f in $(file *.out | grep ELF | cut -d: -f1); do
  ocloc disasm -file "$f" -dump "dis_$f" -device tgllp -skip-asm-translation > /dev/null 2>&1 || true
done
for k in dis_*/.text.*; do
  name="$(basename "$k" | grep -oE 'HGemm[A-Za-z0-9]*' | head -1)"
  [ -n "$name" ] || continue
  "$IGA" -d -p=12p1 "$k" -o "$name.asm" > /dev/null 2>&1
  info="$(dirname "$k")/.ze_info"
  grf="$(awk -v n="$name" '/^  - name:/{on=index($0,n)>0} on&&/grf_count/{print $2; exit}' "$info")"
  spill="$(awk -v n="$name" '/^  - name:/{on=index($0,n)>0} on&&/spill_size/{print $2; exit}' "$info")"
  echo "== $name  grf ${grf:-?}  spill ${spill:-0}"
  grep -oE "\b(mad|mul|add|mov)\b \([0-9]+\|M[0-9]+\)[^ ]* +r[0-9.]+<[0-9]>:[a-z]+" "$name.asm" | sed -E 's/ +r[0-9.]+/ /' \
    | sort | uniq -c | sort -rn | head -5 | sed 's/^/   /'
  grep -oE "// wr:[^;]*; [a-z0-9 ]*(read|write)[a-z0-9 ]*" "$name.asm" | sed -E 's|// wr:[^;]*; ||' | sort | uniq -c \
    | sort -rn | head -4 | sed 's/^/   /'
done
[ "${1:-}" = "--no-run" ] || LD_LIBRARY_PATH="$ROOT/.venv/lib" timeout 600 ./hgemm
