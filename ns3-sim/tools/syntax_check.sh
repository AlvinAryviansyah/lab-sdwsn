#!/usr/bin/env bash
# Pemeriksaan sintaks + semantik honeycomb-sdwsn.cc terhadap HEADER ns-3.48 asli,
# tanpa membangun ns-3 (tidak ada linking, tidak menjalankan simulasi).
#
# Pemakaian:  ./ns3-sim/tools/syntax_check.sh            (dari folder lab/)
# Butuh: clang++ (C++20), curl, python3. Unduhan ~34 MB ke $NS3_HDR_CACHE.
#
# Yang DIBANGKITKAN secara manual (di build ns-3 asli dibangkitkan CMake):
#   core-config.h (int64x64 128-bit), *-export.h (makro kosong), *-module.h
#   (dari daftar header di CMakeLists.txt). Header yang hanya butuh dependensi
#   platform/opsional (mis. win32-internet.h) dibuang dari header modul.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
CC="$HERE/../honeycomb-sdwsn.cc"
CACHE="${NS3_HDR_CACHE:-/tmp/ns3-3.48-headers}"
SRC="$CACHE/ns-3-dev-ns-3.48-src/src"

if [ ! -d "$SRC" ]; then
  mkdir -p "$CACHE"
  curl -sSL -o "$CACHE/ns3.tar.gz" \
    "https://gitlab.com/nsnam/ns-3-dev/-/archive/ns-3.48/ns-3-dev-ns-3.48.tar.gz?path=src"
  tar xzf "$CACHE/ns3.tar.gz" -C "$CACHE"
fi

python3 "$HERE/mkinc.py" "$SRC" "$CACHE/include/ns3"
python3 "$HERE/mkinc.py" --check "$CC" "$CACHE/include"
