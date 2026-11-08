# Reproduksi HieMulti-Block: AI-Based IDS/IPS di Edge-Assisted SDWSN dengan Modified Honeycomb Structure

Reproduksi paper:

> J. Kipongo, T. G. Swart, and E. Esenogho, "Artificial Intelligence-Based
> Intrusion Detection and Prevention in Edge-Assisted SDWSN With Modified
> Honeycomb Structure," *IEEE Access*, vol. 12, pp. 3140-3175, 2024.
> DOI: [10.1109/ACCESS.2023.3347778](https://doi.org/10.1109/ACCESS.2023.3347778)

Proyek tugas kelompok mata kuliah CBNS (Pascasarjana). Dokumen ini menjelaskan
latar belakang, tujuan, ruang lingkup, cara kerja, status reproduksi, kondisi
saat ini, dan rencana pengembangan.

---

## 1. Latar belakang

*Software-Defined Wireless Sensor Network* (SDWSN) memisahkan bidang kontrol dan
bidang data pada jaringan sensor sehingga lebih fleksibel dan mudah dikelola.
Namun SDWSN menghadapi beberapa masalah sekaligus:

- **Keamanan:** penyerang dari dalam jaringan (node yang dikompromikan) tidak
  tertangani oleh secure routing berbasis trust saja.
- **Manajemen jaringan:** penempatan sensor acak dan controller tunggal sulit
  diskalakan.
- **Energi dan latency:** routing tidak efisien menambah hop, energi, dan delay.
- **Throughput:** pemilihan rute hanya memakai sedikit parameter.

Paper di atas mengusulkan sistem terpadu bernama **HieMulti-Block** yang
menggabungkan autentikasi (3D Cube), partisi dan clustering honeycomb, routing
berbasis reinforcement learning (TLDQN), deteksi intrusi hibrida (DTO + Bi-GAN),
pencegahan lewat delegator optimal, dan blockchain hierarkis multi-chain.

Kendala reproduksi: penulis hanya merilis file `Modified_Honeycomb_Structure.cc`
yang bergantung pada modul ns-3 kustom (`ns3::sdn`, `ns3::iot`) yang **tidak
dirilis**. Kelas `TLDQN`, `BiGAN`, `DTO`, `Three_Dimensional_Cube` di kode itu
hanya mengembalikan string. Jadi kode rilis tidak bisa dikompilasi, dan angka
Tabel 7 di paper tidak berasal dari kode yang dirilis.

## 2. Tujuan

1. Merekonstruksi **alur dan metodologi** paper dalam simulasi ns-3 yang bisa
   dijalankan ulang.
2. Mengganti komponen AI yang di kode rilis hanya berupa stub dengan
   **implementasi neural network sungguhan** (Bi-GAN dan TLDQN, PyTorch) yang
   dilatih pada dataset benchmark NSL-KDD.
3. **Menghubungkan** keputusan AI itu ke simulasi jaringan, sehingga deteksi
   dan routing berdampak pada PDR, latency, throughput, dan energi.
4. Mengukur sembilan metrik paper dan **mencatat setiap perbedaan** terhadap
   paper dan kode rilis secara transparan.
5. Mengimplementasikan *future work* #1 paper (feature selection) sebagai
   pelengkap.

## 3. Ruang lingkup

**Termasuk:**
- Simulasi ns-3.48 dengan topologi honeycomb (koordinat dari kode rilis penulis)
  dan entitas sesuai Tabel 5 (100 sensor, 10 user, 4 sink, 4 switch, 3 node
  blockchain, 1 cloud, 1 TA, 1+2 controller).
- Bi-GAN (Generator/Encoder/Discriminator) dan TLDQN (DQN + transfer learning)
  yang dilatih sungguhan.
- DTO tiga fase dengan bukti perilaku yang diukur di simulasi.
- Sembilan metrik, metrik time-based (precision/recall/F1 per window), dan
  confusion matrix, dengan ulangan banyak seed (rata-rata ± simpangan baku).
- Feature selection di NSL-KDD (Mutual Information, Chi-square, RF importance).

**Tidak termasuk (saat ini):**
- Baseline PSO-ABC, OL-RL, RBP-DT dan skenario pembanding (tanpa IDS, oracle,
  routing acak): **di luar ruang lingkup** (keputusan tim: hanya mereplikasi
  model HieMulti-Block).
- 3D Cube berbasis DNN/MCTS dan konsensus blockchain yang sebenarnya.
- Routing multi-hop Cluster Head - Cluster Member di lapisan L3.
- Ekstraksi fitur Bi-GAN dari trafik ns-3 (saat ini memakai sampel NSL-KDD).
- Reproduksi **angka** Tabel 7 dan Fig. 9-17 (lihat bagian 6).

## 4. Cara kerja

### 4.1 Alur sistem (mengikuti Fig. 2 paper)

```
1. Autentikasi (3D Cube)         TA beri private key, simpan di blockchain
        ↓
2. Honeycomb + clustering 1/7    CH dipilih dengan bobot (w = 0.25)
        ↓
3. Routing cerdas (TLDQN)        tiap sensor diarahkan ke edge sink terpilih
        ↓
4. Deteksi hibrida               DTO (perilaku) + Bi-GAN (intrusi)
        ↓                        node terdeteksi diisolasi (IPS)
5. Pencegahan                    delegator optimal kirim alarm ke jaringan
        ↓
6. Evaluasi                      9 metrik + precision/recall/F1 + confusion
```

### 4.2 Pembagian kerja Python dan ns-3

Mengikuti paper (Bi-GAN dan DTO di secondary controller, TLDQN di route agent
pada edge server), keputusan AI diambil di luar node sensor lalu dipakai
jaringan:

```
 Python (ml/)                              ns-3 (ns3-sim/honeycomb-sdwsn.cc)
 ─────────────                             ──────────────────────────────────
 latih Bi-GAN (NSL-KDD)  ─┐
 latih TLDQN (transfer)  ─┼─ export_decisions.py ─► bigan_decisions_<n>_s<seed>.csv  ─► confusion matrix, isolasi
 ukur waktu inferensi    ─┘                         bigan_windows_<n>_s<seed>.csv    ─► metrik time-based
                                                    (tldqn_routes_*: legacy, tidak dibaca; lihat archive/)
                                                    ai_timing.csv                    ─► computation overhead
                                                                 metrics_<n>_s<seed>.csv ─► metrics_plots.py
```

Waktu simulasi (satu run): autentikasi di awal, clustering + routing pada
t = 1.3 s (data mulai mengalir), deteksi hibrida + isolasi pada t = 8.3 s,
monitoring IDS tiap 30 s, delegator dan alarm pada t = 15.3 s.

### 4.3 Dampak keputusan AI ke jaringan

- **Ground truth:** 10% node malicious ditentukan skenario (seed), bukan oleh
  DTO, sehingga confusion matrix tidak circular.
- **Penyerang** melakukan flooding (faktor acak 1.5x sampai 5x laju normal)
  sampai diisolasi.
- **Routing:** hasil TLDQN menentukan edge sink tujuan aliran data tiap sensor.
- **Isolasi:** node yang ditandai DTO atau Bi-GAN berhenti mengirim. False
  positive Bi-GAN ikut mengorbankan node sah, sehingga biayanya terlihat di
  throughput.
- **DTO** memakai bukti yang diukur (laju kirim dan energi terpakai per sensor).

### 4.4 Profil simulasi

| Profil | Isi |
|---|---|
| `--profile=paper` (default) | Tabel 4-5: 600 s, ukuran paket 64-1024 B, 802.11ac 80 MHz. **50 paket/node** (Tabel 5: 1000, tidak muat di memori) dan TCP user -> cloud mati default |
| `--profile=code` | Setara kode rilis penulis: 300 s, 802.11a 6 Mbps, tanpa TCP |

Rincian dan alasan tiap perbedaan ada di
[`docs/PAPER_DEVIATIONS.md`](docs/PAPER_DEVIATIONS.md).

### 4.5 Struktur repositori

```
lab/
├── README.md
├── docs/PAPER_DEVIATIONS.md      # catatan perbedaan paper vs kode rilis vs reproduksi
├── docs/TELEMETRY.md             # definisi telemetry NS-3 + watchdog (langkah 2)
├── data/
│   ├── SDWSNIDS_Dataset/         # dataset + kode asli penulis (figshare, CC BY 4.0)
│   └── nsl-kdd/                  # KDDTrain+/KDDTest+ (tidak di-commit; unduh sendiri)
├── ml/
│   ├── nslkdd_data.py            # preprocessing NSL-KDD (122 fitur)
│   ├── bigan_ids.py              # Bi-GAN sungguhan
│   ├── bigan_model.py            # definisi Bi-GAN (model, kerugian, skor, ambang)
│   ├── bigan_infer.py            # inferensi Bi-GAN dari checkpoint
│   ├── detector_eval.py          # evaluasi Bi-GAN (validasi/test)
│   ├── honeycomb_topology.py     # topologi + ruang aksi bersama Python/NS-3
│   ├── routing_env.py            # lingkungan routing honeycomb hop-by-hop
│   ├── tldqn_routing.py          # TLDQN (DQN + transfer learning) dan evaluasi
│   ├── legacy_tldqn.py           # TLDQN lama (hanya untuk export_decisions.py)
│   ├── archive_experiments.py    # arsip hasil eksperimen + manifest SHA-256
│   ├── tests/                    # tes unit (python -m unittest discover -s tests)
│   ├── reports/                  # laporan evaluasi (JSON, PNG, skema routing)
│   ├── experiments/              # arsip hasil eksperimen per fase
│   ├── export_decisions.py       # jembatan Python -> ns-3
│   ├── export_routes.py          # tabel next-hop TLDQN -> ns-3 (--trust-source)
│   ├── telemetry_schema.py       # skema + validator telemetry NS-3
│   ├── telemetry_trust.py        # trust Beta dari watchdog (offline, tahap-1 -> tahap-2)
│   ├── telemetry_eval.py         # evaluasi telemetry (satu-satunya tempat label dipakai)
│   ├── metrics_plots.py          # grafik dari hasil ns-3
│   └── requirements.txt
├── ns3-sim/
│   ├── honeycomb-sdwsn.cc        # simulasi utama (ns-3.48)
│   ├── run_sweep.sh              # sweep node x seed
│   ├── BUILD.md                  # build dan run di server ns-3
│   └── results/                  # CSV keputusan AI dan hasil metrik
├── figures/                      # grafik aktif (real/: hasil langkah 1-2)
├── dashboard/index.html          # dashboard performa end-to-end
└── archive/                      # berkas tidak terpakai (lihat archive/README.md)
```

### 4.6 Menjalankan (ringkas)

Rincian langkah ada di [`ns3-sim/BUILD.md`](ns3-sim/BUILD.md).

```bash
# 1. Di laptop: siapkan lingkungan dan hasilkan keputusan AI
python3 -m venv .venv && .venv/bin/pip install -r ml/requirements.txt
# letakkan KDDTrain+.txt dan KDDTest+.txt di data/nsl-kdd/
.venv/bin/python ml/export_decisions.py --epochs 20 --seeds 5
#   (ulang tanpa melatih lagi: tambahkan --reuse)

# 2. Di server ns-3.48: salin kode dan CSV, build, ukur waktu satu run
cp ns3-sim/honeycomb-sdwsn.cc /app/ns-3-dev/scratch/
cp ns3-sim/results/{bigan_decisions,bigan_windows,tldqn_nexthop}_*_s*.csv \
   ns3-sim/results/ai_timing.csv /app/ns-3-dev/results/
cd /app/ns-3-dev && ./ns3 build honeycomb-sdwsn
time ./ns3 run "honeycomb-sdwsn --numNodes=100 --seed=1"

# 3. Sweep (hemat compute: NODES="20 60 100")
NS3_DIR=/app/ns-3-dev ./ns3-sim/run_sweep.sh 3

# 4. Di laptop: salin metrics_*_s*.csv dan timeseries_*_s*.csv ke ns3-sim/results/
.venv/bin/python ml/metrics_plots.py
```

### 4.7 Dashboard performa end-to-end

`dashboard/index.html` adalah satu berkas HTML mandiri (tanpa CDN, tanpa dependensi)
yang dibangkitkan dari hasil run di repo: jaringan, routing TLDQN, deteksi Bi-GAN/DTO,
trust & watchdog, IPS, alur end-to-end, dan perbandingan eksperimen dengan filter.
Ground truth hanya lapisan evaluasi (bisa dimatikan); hasil NSL-KDD terpisah dari NS-3.

```bash
.venv/bin/python ml/build_dashboard.py      # bangkitkan ulang setelah ada hasil baru
# di server ns-3 (cukup python3 bawaan):
python3 -m http.server 8050 --bind 127.0.0.1 --directory dashboard
# dari laptop: ssh -L 8050:127.0.0.1:8050 user@server, lalu buka http://localhost:8050
```

## 5. Metrik yang diukur

| Metrik | Cara ukur di reproduksi |
|---|---|
| Energi | total - sisa (BasicEnergySource), arus radio kelas sensor |
| Latency, throughput, PDR, drop | FlowMonitor pada trafik UDP sensor (TCP dilaporkan terpisah) |
| Network lifetime | **proyeksi** = energi awal / daya rata-rata per node |
| Computation overhead | RT/CT: waktu per keputusan (3D Cube + Bi-GAN + TLDQN) terhadap latency |
| Control overhead | hitungan pesan kontrol **logis** per detik |
| Detection accuracy, precision, recall, F1, confusion | dipool lintas window monitoring |

## 6. Status reproduksi

Penilaian di bawah adalah **estimasi subjektif** (bukan hasil pengukuran).

| Aspek | Status | Catatan |
|---|---|---|
| Alur 6 tahap dan urutan | Sesuai | Routing dipindah sebelum data mengalir (sesuai Sec. IV-B) |
| Topologi dan jumlah entitas | Sesuai | Koordinat dari kode rilis |
| Parameter simulasi Tabel 5 | Sebagian | Profil `paper` mendekati; 100 MHz diganti 80 MHz; jangkauan 150 m **dipaksakan** sejak integrasi TLDQN langkah 1 (kode rilis penulis tidak memaksakannya) |
| Bi-GAN | Sebagian | Model asli, skor deteksi = logit Discriminator pada (x, E(x)) (Fig. 6); fitur dari sampel NSL-KDD yang ditugaskan ke node |
| TLDQN | Sebagian | **Python (fase 2):** DQN + transfer learning pada routing honeycomb hop-by-hop (aksi = 6 slot next-hop, jangkauan 150 m). **NS-3 (integrasi langkah 1):** tabel next-hop TLDQN dipasang sebagai host route, jangkauan 150 m, relay malicious grayhole; menunggu verifikasi run di server |
| Telemetry NS-3 (langkah 2) | **Selesai, tervalidasi** | Trace IP/MAC/PHY/energi + watchdog pasif; trust offline dua tahap. Validasi: kirim/terima dan penerusan UDP (delivered) sama persis dengan FlowMonitor, seed ulang identik 3/3, trust telemetry AUC 1.000 pada sensor teramati (27/27 skenario, tetapi hanya 71/162 malicious teramati); rute tahap-2 PDR 97.08 vs 92.73 tahap-1 (p = 0.0005). `docs/TELEMETRY.md`, `docs/PAPER_DEVIATIONS.md` 2d |
| DTO | Sebagian | **NS-3 (langkah 1):** 3 fase asli; bukti = laju kirim dan energi (self-report). **Offline (langkah 3, selesai):** pers. (21)–(31) literal di `ml/dto.py`, bukti PLR watchdog telemetry, observer b ≥ 0.8 dikecualikan (K9), energi Not Available (K10 ditunda). stage2: P 1.000, R 0.191 (cakupan 31/162), ctrl0 0 FP; tidak lebih baik dari ambang trust sederhana. Belum memengaruhi simulasi (langkah 4–5). `docs/STEP3_DTO_DESIGN.md` |
| 3D Cube, blockchain, CH selection, delegator | Disederhanakan | Hash/log, tanpa DNN-MCTS dan konsensus |
| Sembilan metrik | Sebagian | Lifetime dan computation overhead berupa proksi |
| Perbandingan baseline (Fig. 9-17) | **Di luar ruang lingkup** | Keputusan tim; kurva acuan Tabel 7 diarsipkan di `archive/figures-reference-table7/` (bukan hasil simulasi) |
| Angka Tabel 7 | **Tidak dikejar** | Kode rilis tidak menghasilkan angka itu |

Perkiraan kemiripan untuk **model HieMulti-Block** (tanpa baseline): sekitar
65-70% setelah profil `paper` dijalankan. Angka persis Tabel 7 tidak bisa
dicapai dengan jujur karena kode rilis penulis tidak menghasilkannya.

## 7. Kondisi saat ini

### 7.1 Sudah selesai dan terverifikasi

- Modul AI Python berjalan dan menghasilkan angka nyata pada NSL-KDD:

  | Model | Hasil |
  |---|---|
  | Bi-GAN one-class (KDDTest+) | Accuracy 77.9%, Precision 82.6%, Recall 77.4%, F1 79.9% |
  | TLDQN vs DQN (routing honeycomb, 5 seed) | Gain awal 0.71 vs -0.17; rata-rata gain sepanjang pelatihan 0.79 vs 0.66; gain akhir setara (0.84 vs 0.83); PDR akhir 0.66 (oracle informasi penuh 0.72, Bayes informasi setara 0.72, greedy 0.44) |
  | Feature selection | 20 fitur: F1 ~76% (Chi-square) vs ~75% dengan 122 fitur, waktu latih lebih rendah |

- Pipeline Python -> CSV -> ns-3 -> CSV -> grafik berjalan end-to-end pada
  **profil `paper`** (600 s, 802.11ac 80 MHz, ukuran paket 64-1024 B,
  **50 paket/node**, tanpa TCP, flooding heterogen 1.5x-5x, monitoring IDS
  20 window), 9 ukuran jaringan x 3 seed = 27 run (total sekitar 8 menit di
  server uji). Skor deteksi Bi-GAN: **logit Discriminator** (default).
  Ringkasan (rata-rata ± simpangan baku antar seed):

  | Metrik | 20 node | 60 node | 100 node |
  |---|---|---|---|
  | Energi total (J) | 18.1 ± 0.0 | 54.8 ± 0.0 | 92.5 ± 0.1 |
  | Throughput (kbps) | 6.7 ± 0.2 | 23.8 ± 0.7 | 45.3 ± 2.2 |
  | Latency (ms) | 0.23 ± 0.00 | 0.67 ± 0.32 | 0.99 ± 0.39 |
  | PDR (%) | 100.00 ± 0.00 | 99.98 ± 0.02 | 99.87 ± 0.05 |
  | Detection accuracy (%) | 93.5 ± 0.2 | 93.8 ± 0.3 | 92.7 ± 0.2 |
  | Precision Bi-GAN (%) | 75.5 ± 4.3 | 78.2 ± 3.6 | 70.9 ± 2.4 |
  | Recall Bi-GAN (%) | 52.5 ± 3.5 | 52.2 ± 2.4 | 45.0 ± 0.7 |
  | F1-score (%) | 61.7 ± 1.0 | 62.5 ± 1.7 | 55.1 ± 1.2 |
  | Control overhead (mps) | 0.91 ± 0.00 | 2.64 ± 0.00 | 4.37 ± 0.00 |
  | Node terisolasi | 2.0 ± 0.0 | 7.0 ± 0.8 | 13.0 ± 1.4 |
  | Penyerang terisolasi | 2.0 ± 0.0 | 5.7 ± 0.5 | 8.3 ± 1.2 |

  Data mentah: `ns3-sim/results/metrics_<n>_s<seed>.csv` dan
  `timeseries_<n>_s<seed>.csv`.
- **Perbaikan F1 Bi-GAN:** mengganti skor reconstruction error dengan skor
  Discriminator menaikkan F1 rata-rata semua n dari **31.8%** menjadi
  **61.3%** (sapuan lama diarsipkan di
  `archive/ns3-results-superseded/ns3-sim/results/legacy_run3_paper_recon_detector/`). Perbandingan 5 varian
  skor ada di `ml/detector_eval.py`. Hasil sapuan sebelumnya di
  `archive/ns3-results-superseded/.../legacy_run2_5seed_old_config/` memakai konfigurasi lama.

### 7.2 Batas lingkungan uji (temuan penting)
- Server uji (6.2 GiB RAM, 2 CPU) **tidak muat** untuk 1000 paket per node
  (Tabel 5). Memori puncak pada 100 node kira-kira
  `115 MB + 37.5 MB x paket per node` (ekstrapolasi dari dua titik ukur),
  jadi 1000 paket butuh sekitar 37 GB. Karena itu paket per node diturunkan
  ke **50**, dan TCP user -> cloud (membengkakkan memori hingga 4.6 GB pada
  20 node) dimatikan secara default. Penyebab pasti di dalam ns-3 belum
  ditemukan. Rincian: `docs/PAPER_DEVIATIONS.md` bagian 4a.
- Profil `paper` sudah dikompilasi dan dijalankan di ns-3.48 pada server uji.

### 7.3 Temuan dan keterbatasan yang perlu dicatat di laporan
1. **Tren terhadap jumlah node terreproduksi:** throughput, energi, dan
   control overhead naik seiring n, seperti di paper.
2. **Kualitas deteksi:** accuracy sekitar 93%, precision sekitar 71-78%,
   recall sekitar 45-56%, F1 sekitar 53-68% (rata-rata 61%). Paper
   melaporkan precision/recall/F1 sekitar 86/92/89%: **recall kita jauh lebih
   rendah** (sekitar separuh serangan lolos). Penyebabnya Bi-GAN one-class
   di KDDTest+ yang memuat banyak serangan yang tak pernah muncul saat
   pelatihan. Accuracy 93% juga menggembung karena 90% node normal (prevalensi
   10%); gunakan F1 sebagai ukuran utama.
3. **Biaya false positive jauh lebih kecil dari sebelumnya:** pada 100 node
   sekitar 5 node sah ikut terisolasi (sebelumnya sekitar
   20) dari 8 penyerang yang terisolasi.
4. **DTO tidak lagi trivial** (FP/FN 0-3 per run) setelah flooding dibuat
   heterogen.
5. **Energi tidak informatif:** sekitar 0.9 J per node sama dengan arus idle
   x 600 s, jadi ditentukan lantai idle, bukan trafik atau IDS. **Lifetime**
   (sekitar 65.000 s) adalah proyeksi dari simulasi yang hampir seluruhnya
   idle dan tidak sebanding dengan 400-580 s di paper.
6. **Computation overhead** ditentukan oleh latency yang sangat kecil, bukan
   oleh komputasi AI; trennya tidak boleh ditafsirkan. **Control overhead**
   sebagian besar berasal dari laporan monitoring per window dan hanya
   hitungan logis.
7. **PDR hampir 100%** dan latency di bawah 2 ms: dengan 50 paket per node,
   jaringan tidak padat sehingga level PDR/latency paper (PDR 58-66%, latency
   35-55 ms) tidak terulang. Perbandingan nilai absolut tidak adil.
8. **Fitur Bi-GAN ditugaskan** dari sampel NSL-KDD, bukan diekstrak dari
   trafik ns-3.
9. **Hanya 3 seed.** Simpangan baku antar seed kecil untuk sebagian besar
   metrik, tetapi 3 sampel terlalu sedikit untuk interval kepercayaan yang
   andal.

## 8. Future works

**Prioritas (replikasi model HieMulti-Block):**
- Cari penyebab pertumbuhan memori di ns-3 agar 1000 paket per node (Tabel 5)
  dan TCP user -> cloud bisa dijalankan.
- Tambah seed (mis. 10) agar interval kepercayaan lebih rapat.
- Perbaiki definisi lifetime dan computation overhead agar bermakna.

*(Baseline PSO-ABC/OL-RL/RBP-DT dan skenario pembanding sengaja tidak
dikerjakan: di luar ruang lingkup.)*

**Peningkatan fidelitas:**
- Routing multi-hop CH - CM di L3 (neighbor graph 6 tetangga heksagon, TLDQN
  memilih next forwarder lalu rute), dengan jangkauan 150 m.
- Ekstraksi fitur flow dan packet dari trafik ns-3 (SrcIP, DstIP, port, tipe
  paket), lalu latih Bi-GAN pada fitur tersebut, bukan sampel NSL-KDD.
- Perbaikan Bi-GAN: kalibrasi ambang yang lebih baik, pendekatan semi-supervised,
  dan evaluasi dengan prevalensi serangan rendah.
- DTO dengan bukti PDR dan packet loss (bukan hanya laju kirim dan energi).
- 3D Cube dengan DNN/MCTS dan blockchain dengan validasi/konsensus.
- Definisi lifetime sebagai first-node-death dengan beban trafik berkelanjutan.
- Lebih banyak seed dan interval kepercayaan; uji signifikansi.

**Future work dari paper:**
- Feature selection (sudah ada, berdiri sendiri; belum terintegrasi ke Bi-GAN).
- Feature extraction (belum).
- Deep learning sebagai anomaly detector ringan di sisi sensor.
- Protokol komunikasi berbasis priority/weighted fair queue untuk QoS.
- Integrasi 5G, signcryption berbasis fog/edge, dan enkripsi kuantum.

**Rekayasa:**
- Bungkus lingkungan (Docker) berisi ns-3.48 dan Python agar reproduksi tidak
  bergantung pada server tertentu.
- Otomatisasi pipeline (satu perintah dari pelatihan sampai grafik).

## 9. Lisensi dan atribusi

Dataset dan kode asli penulis (`data/SDWSNIDS_Dataset/`) berlisensi CC BY 4.0,
Center for Telecommunications, University of Johannesburg. NSL-KDD dari
Canadian Institute for Cybersecurity (UNB). Repositori ini adalah karya
reproduksi untuk tujuan akademik; semua hak atas metode dan desain sistem
milik penulis paper.
