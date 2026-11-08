# Arsip: berkas yang tidak dipakai pipeline aktif

Berkas di sini **tidak dihapus**, hanya dipindah dengan `git mv` (riwayat git tetap utuh).
Tidak ada kode, tes, simulasi NS-3, atau dashboard yang membaca folder ini. Daftar lengkap
berisi path asal dan SHA-256 ada di `MANIFEST.json`.

Strukturnya: `archive/<grup>/<path asal>`. Untuk mengembalikan satu berkas:

```bash
git mv archive/<grup>/<path_asli> <path_asli>
```

| Grup | Berkas | Alasan diarsipkan |
|---|---|---|
| `ns3-results-superseded/` | 118 | Hasil sweep NS-3 lama: satu seed, konfigurasi lama tanpa jangkauan 150 m, detektor reconstruction error, dan IDS sintetis. Semuanya digantikan hasil langkah 1-2. |
| `ns3-inputs-legacy-tldqn-routes/` | 45 | `tldqn_routes_*` (pilihan sink TLDQN lama). Sejak langkah 1 NS-3 membaca `tldqn_nexthop_*`. |
| `ns3-inputs-unused-seeds-4-5/` | 36 | Skenario seed 4-5 dari `export_decisions.py --seeds 5`. Sweep dan analisis hanya memakai seed 1-3. |
| `legacy-first-ai-module/` | 2 | Keluaran modul AI pertama, digantikan evaluasi Bi-GAN refaktor (`ml/reports/bigan_eval_*`). |
| `figures-pre-step1-sweep/` | 12 | Grafik dari sweep sebelum langkah 1. Data sumbernya sudah ditimpa hasil langkah 1, jadi grafik ini tidak lagi mencerminkan `metrics_*.csv`. |
| `figures-reference-table7/` | 9 | Kurva acuan baseline Tabel 7 paper. Ini bukan hasil simulasi, dan perbandingan baseline di luar ruang lingkup. |
| `future-work-feature-selection/` | 4 | Future work #1 paper (feature selection NSL-KDD), bukan bagian model HieMulti-Block. |

Catatan:

- `ml/export_decisions.py` dengan default `--seeds 5` akan membuat ulang `tldqn_routes_*`
  dan berkas seed 4-5 di `ns3-sim/results/`.
- `ml/metrics_plots.py` akan membuat ulang `figures/real/*.png` dan `figures/compare/`
  dari `metrics_*.csv` saat ini.
- Folder `semantic-review/` (dokumen review lokal, tidak di-commit) tidak ikut dipindah.
