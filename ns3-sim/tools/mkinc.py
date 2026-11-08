#!/usr/bin/env python3
"""Pembantu syntax_check.sh.

  mkinc.py <src_dir> <include/ns3>      susun direktori include seperti build ns-3
  mkinc.py --check <file.cc> <include>  jalankan clang++ -fsyntax-only; buang
                                         header modul yang hanya gagal karena
                                         dependensi platform/opsional
"""
import os
import re
import sys
import glob
import shutil
import subprocess

EXPORT_MODULES = ("applications", "core", "internet", "lte", "mobility", "wifi")
CORE_CONFIG = """#ifndef NS3_CORE_CONFIG_H
#define NS3_CORE_CONFIG_H
#define HAVE___UINT128_T 1
#define INT64X64_USE_128 1
#define HAVE_STDINT_H 1
#define HAVE_INTTYPES_H 1
#define HAVE_SYS_TYPES_H 1
#define HAVE_SYS_STAT_H 1
#define HAVE_DIRENT_H 1
#define HAVE_STDLIB_H 1
#define HAVE_GETENV 1
#define HAVE_SIGNAL_H 1
#endif
"""


def build_include(src, inc):
    os.makedirs(inc, exist_ok=True)
    mods = sorted(d for d in os.listdir(src) if os.path.isfile(os.path.join(src, d, "CMakeLists.txt")))
    for m in mods:
        for sub in ("model", "helper", "utils"):
            for h in glob.glob(os.path.join(src, m, sub, "**", "*.h"), recursive=True):
                shutil.copy(h, os.path.join(inc, os.path.basename(h)))
    for m in mods:
        txt = open(os.path.join(src, m, "CMakeLists.txt")).read()
        seen, out = set(), []
        for h in re.findall(r"(?:model|helper|utils)/[\w\-/]+\.h", txt):
            b = os.path.basename(h)
            if b not in seen:
                seen.add(b)
                out.append(b)
        guard = f"NS3_{m.upper().replace('-', '_')}_MODULE_H"
        with open(os.path.join(inc, f"{m}-module.h"), "w") as f:
            f.write(f"#ifndef {guard}\n#define {guard}\n")
            f.writelines(f'#include "ns3/{b}"\n' for b in out)
            f.write("#endif\n")
    with open(os.path.join(inc, "core-config.h"), "w") as f:
        f.write(CORE_CONFIG)
    for m in EXPORT_MODULES:
        M = m.upper()
        with open(os.path.join(inc, f"{m}-export.h"), "w") as f:
            f.write(f"#ifndef {M}_EXPORT\n#define {M}_EXPORT\n#endif\n")


def check(cc, inc_root):
    inc = os.path.join(inc_root, "ns3")
    removed = []
    for _ in range(80):
        r = subprocess.run(["clang++", "-std=c++20", "-fsyntax-only", "-I", inc_root,
                            "-DNS3_LOG_ENABLE", "-DNS3_ASSERT_ENABLE", cc],
                           capture_output=True, text=True)
        errs = [l for l in r.stderr.splitlines() if " error: " in l]
        if not errs:
            print(f"OK: {os.path.basename(cc)} lolos pemeriksaan sintaks/semantik (ns-3.48)")
            print("header opsional dibuang:", removed or "-")
            return 0
        m = re.search(r"include/ns3/([\w\-.]+\.h):\d+:\d+: fatal error: '([^']+)' file not found",
                      errs[0])
        if not m:
            print("\n".join(errs[:20]))
            return 1
        line = f'#include "ns3/{m.group(1)}"\n'
        hit = False
        for mod in glob.glob(os.path.join(inc, "*-module.h")):
            t = open(mod).read()
            if line in t:
                open(mod, "w").write(t.replace(line, ""))
                hit = True
        if not hit:
            print("gagal (bukan dependensi opsional dari header modul):", errs[0])
            return 1
        removed.append(f"{m.group(1)} (butuh {m.group(2)})")
    return 1


if __name__ == "__main__":
    if sys.argv[1] == "--check":
        sys.exit(check(sys.argv[2], sys.argv[3]))
    build_include(sys.argv[1], sys.argv[2])
