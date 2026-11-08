#!/usr/bin/env bash
# Sweep jumlah node x banyak seed untuk honeycomb-sdwsn.
#
# Pemakaian (di server ns-3):
#   NS3_DIR=/app/ns-3-dev ./run_sweep.sh [jumlah_seed=3] [simTime=ikuti profil]
#
# Opsi lewat variabel lingkungan:
#   NODES="20 60 100"     daftar jumlah node (default: 20 30 ... 100)
#   PROFILE=paper|code    profil simulasi (default: paper)
#   EXTRA="--wifiStd=a"   argumen tambahan untuk honeycomb-sdwsn
#   ROUTE_DIR=results/routes_greedy OUT_TAG=_greedy   tabel referensi (bukan metode);
#                                     keluaran metrics_<n>_s<seed>_greedy.csv
#   TELEM_DIR=results/telemetry_x   folder telemetry (default results/telemetry${OUT_TAG})
#
# Prasyarat: honeycomb-sdwsn.cc sudah di scratch/ dan di-build, serta CSV
# keputusan AI (bigan_decisions/bigan_windows _<n>_s<seed>.csv, tabel next-hop
# tldqn_nexthop_<n>_s<seed>.csv di $ROUTE_DIR, dan
# ai_timing.csv) ada di $NS3_DIR/results/ (hasil ml/export_decisions.py).
#
# Keluaran: results/metrics_<n>_s<seed>.csv, results/timeseries_<n>_s<seed>.csv
#           results/log_<n>_s<seed>.txt, results/sweep_times.csv (waktu per run)
#           $TELEM_DIR/telemetry{,_links}_<n>_s<seed>.csv, watchdog_audit_*, telemetry_meta_*
set -euo pipefail

NS3_DIR="${NS3_DIR:-/app/ns-3-dev}"
SEEDS="${1:-3}"
SIMTIME="${2:-}"   # kosong => ikuti profil (paper 600 s, code 300 s)
NODES="${NODES:-20 30 40 50 60 70 80 90 100}"
PROFILE="${PROFILE:-paper}"
EXTRA="${EXTRA:-}"
ROUTE_DIR="${ROUTE_DIR:-results}"   # folder tabel next-hop TLDQN (relatif thd NS3_DIR)
OUT_TAG="${OUT_TAG:-}"               # akhiran nama keluaran, mis. _greedy (cegah tertimpa)
TELEM_DIR="${TELEM_DIR:-results/telemetry${OUT_TAG}}"   # telemetry_*.csv (langkah 2)

cd "$NS3_DIR"
mkdir -p results "$TELEM_DIR"

# Cek kelengkapan input dulu agar tidak gagal di tengah sweep.
missing=0
for s in $(seq 1 "$SEEDS"); do
  for n in $NODES; do
    for f in "results/bigan_decisions_${n}_s${s}.csv" \
             "results/bigan_windows_${n}_s${s}.csv" \
             "${ROUTE_DIR}/tldqn_nexthop_${n}_s${s}.csv"; do
      [ -f "$f" ] || { echo "KURANG: $f"; missing=1; }
    done
  done
done
[ -f results/ai_timing.csv ] || { echo "KURANG: results/ai_timing.csv"; missing=1; }
[ "$missing" -eq 0 ] || { echo "Salin CSV keputusan AI dulu (lihat BUILD.md langkah 0-1)."; exit 1; }

SIMARG=""; [ -n "$SIMTIME" ] && SIMARG="--simTime=${SIMTIME}"
nn=$(echo $NODES | wc -w)
total=$(( SEEDS * nn )); i=0
echo "n,seed,detik" > results/sweep_times${OUT_TAG}.csv
for s in $(seq 1 "$SEEDS"); do
  for n in $NODES; do
    i=$(( i + 1 ))
    echo "[$i/$total] node=$n seed=$s profile=$PROFILE"
    t0=$SECONDS
    ./ns3 run "honeycomb-sdwsn --profile=${PROFILE} --numNodes=${n} ${SIMARG} --seed=${s} --outCsv=results/metrics_${n}_s${s}${OUT_TAG}.csv --tsCsv=results/timeseries_${n}_s${s}${OUT_TAG}.csv --routeDir=${ROUTE_DIR} --telemDir=${TELEM_DIR} ${EXTRA}" \
      > "results/log_${n}_s${s}${OUT_TAG}.txt" 2>&1 || { echo "GAGAL: lihat results/log_${n}_s${s}${OUT_TAG}.txt"; exit 1; }
    echo "${n},${s},$(( SECONDS - t0 ))" >> results/sweep_times${OUT_TAG}.csv
    echo "    selesai dalam $(( SECONDS - t0 )) s"
  done
done
echo "Selesai. Hasil: $NS3_DIR/results/metrics_<n>_s<seed>.csv"
