# Catatan Perbedaan: Paper vs Kode Rilis Penulis vs Reproduksi Ini

Dokumen ini mencatat setiap perbedaan antara tiga hal:

- **Paper**: Kipongo et al., IEEE Access vol. 12, 2024 (Tabel 4, 5, Sec. III-IV).
- **Kode rilis**: `data/SDWSNIDS_Dataset/Source Code/src/scratch/Modified_Honeycomb_Structure.cc`
  (apa yang benar-benar dijalankan penulis; dibaca langsung dari file).
- **Reproduksi**: `ns3-sim/honeycomb-sdwsn.cc` + `ml/` (repo ini).

Tujuan reproduksi: **setup dan metodologi mengikuti paper**, hasil dilaporkan
apa adanya. Angka Tabel 7 **tidak** dikejar (lihat bagian 5).

Profil simulasi:
- `--profile=paper` (default): mengikuti Tabel 4-5 sedekat mungkin.
- `--profile=code`: setara kode rilis penulis (802.11a 6 Mbps, 300 s, tanpa TCP).

---

## 1. Parameter simulasi (Tabel 4 dan 5)

| Parameter | Paper | Kode rilis penulis | Reproduksi (`paper`) | Catatan |
|---|---|---|---|---|
| Simulator | NS-3.26 | ns-allinone-3.26 | ns-3.48 | API berbeda; modul `sdn`/`iot` penulis tidak rilis |
| Sensor / user / sink / switch | 100 / 10 / 4 / 4 | sama | sama | |
| Controller primer / sekunder | 1 / 2 | (ada di modul tak rilis) | node ada, fungsi disimulasikan di C++/Python | |
| Node blockchain / cloud / TA | 3 / 1 / 1 | sama | sama | |
| Area | 1000 x 1000 m | koordinat honeycomb tetap | sama (koordinat diambil dari kode rilis) | |
| Energi awal | 100 J | n/a di kode rilis | 100 J | |
| Waktu simulasi | 600 s | **300 s** | 600 s | profil `code`: 300 s |
| Jangkauan transmisi | 150 m | **tidak dipaksakan** (Nakagami tanpa path loss) | **150 m** (`RangePropagationLossModel`) sejak integrasi langkah 1 | Multi-hop ke edge sink lewat tabel next-hop TLDQN (bagian 2c). `--maxRange 0` HANYA mencabut batas jangkauan; host route TLDQN tetap terpasang sehingga paket tetap diteruskan multi-hop (bukan perilaku kode rilis; terverifikasi: hop 2.32, drop grayhole identik) |
| Ukuran paket | 64, 128, ..., 1024 B | **1024 B** tetap | acak 64..1024 B per aliran | profil `code`: 1024 B |
| Jumlah paket | 1000 | **100** per node | **50** per node | **Batas memori**, lihat bagian 4a. Interval kirim diskalakan 20/n detik |
| Tipe trafik | TCP/IP dan UDP | UDP broadcast (sensor) + TCP OnOff IPv6 (user, indeks genap) | UDP unicast (sensor). TCP user -> cloud **tersedia tetapi mati default** | TCP membengkakkan memori (lihat 4a). Aktifkan dengan `--tcpUsers=true` |
| Data rate paket | 100 Mbps | `dataRate="250Kbps"` (aplikasi), PHY 6 Mbps | PHY 802.11ac VhtMcs2 (~88 Mbps) | 100 Mbps tepat tidak tersedia |
| Bandwidth kanal | 100 MHz | 10 MHz (`OfdmRate6MbpsBW10MHz`) | **80 MHz** | 802.11ac tidak punya lebar 100 MHz; 80 MHz terdekat. Fallback: `--wifiStd=a` (802.11a 54 Mbps) |
| Daya transmisi | 0.005 W | n/a | dipakai di **model energi** (arus Tx = 0.005/3 A) | `TxPower` PHY WiFi **tidak** diubah (default ns-3) |
| Mobilitas | Random Waypoint | user RWP | user RWP, sensor/infrastruktur tetap | |
| Node ilegitimat | 10% | `Attackers=10` | 10%, ditentukan skenario (seed) | Ground truth bukan keluaran DTO |

## 2. Komponen dan algoritma

| Komponen | Paper | Reproduksi | Status |
|---|---|---|---|
| 3D Cube + DNN (autentikasi) | DNN + MCTS memecahkan cube, SHA-256 | Hash FNV dengan XOR per langkah (<= 30 langkah) | **Disederhanakan**: tidak ada DNN/MCTS |
| Clone attack detection | Cek ID sama di lokasi berbeda | Hanya dicetak di log | **Tidak dimodelkan** |
| Honeycomb 1/7 | Posisi di tepi dan pusat hexagon | Koordinat dari kode rilis | Posisi **sama** |
| Pemilihan cluster head (pers. 16) | Bobot centrality/distance/energy/trust = 0.25 | Fungsi bobot ada, hasil CH tidak dipakai routing | **Disederhanakan** |
| TLDQN (Alg. 1) | DQN + transfer learning, forwarder lalu rute | DQN + replay + target network + transfer **asli** (PyTorch); aksi = pilih edge sink; lingkungan **sintetis** | Sebagian; bukan routing CH-CM multi-hop |
| DTO (Alg. 2) | 3 fase, OF dari energi/PDR/loss | 3 fase asli; OF = (z - bukti)^2, bukti = laju kirim + energi **terukur**; fase 3 aditif | Sebagian; tanpa PDR/loss sebagai bukti |
| Bi-GAN (Alg. 3) | G/E/D, fitur flow + packet; testing: Encoder -> Concatenate -> Discriminator (Fig. 6) | G/E/D **asli**, one-class di trafik normal NSL-KDD; skor deteksi = **logit Discriminator pada (x, E(x))** (default `--detector disc`) | Sebagian; fitur dari **sampel NSL-KDD yang ditugaskan ke node**, bukan diekstrak dari trafik ns-3 |
| Skor deteksi alternatif | tidak ada | `recon` (reconstruction error) tersedia; `ml/detector_eval.py` membandingkan 5 varian | Reconstruction error memberi F1 ~33% pada prevalensi 10%; skor Discriminator ~60% (dievaluasi di KDDTest+, ambang dari TRAIN). Varian semi-supervised (~70%) **tidak dipakai** karena memakai label serangan saat pelatihan |
| Ambang Bi-GAN | tidak dijelaskan | Dikalibrasi di **train** dengan bobot prevalensi 10% | Pilihan kami; tanpa kebocoran test |
| Prevention (delegator) | Berdasar network active time, trust, throughput, energi | Sensor sehat dengan **energi sisa terbesar** | **Disederhanakan** |
| HieMulti-Block | Main chain + side chain, konsensus | Hash + sharding sederhana | **Disederhanakan**: tanpa konsensus/validasi |
| Perilaku penyerang | tidak dirinci | Flooding dengan faktor acak 1.5x-5x laju normal sampai diisolasi | Asumsi kami |
| Window IDS | "sepanjang 600 s" | Window 30 s; klasifikasi ulang dengan sampel baru; isolasi hanya pada deteksi awal | Asumsi kami |

## 2a. Bi-GAN (refaktor fase 1)

Berkas: `ml/bigan_model.py` (model, kerugian, skor, ambang, metrik),
`ml/bigan_ids.py` (pelatihan + checkpoint), `ml/bigan_infer.py` (inferensi),
`ml/detector_eval.py` (evaluasi), `ml/nslkdd_data.py` (preprocessing),
`ml/tests/` (42 tes unit).

Legenda status: **V** = terverifikasi dari teks paper; **A** = paper ambigu atau
tidak konsisten; **U** = tidak disebut paper (nilai di sini adalah asumsi).

### 2a.1 Arsitektur dan pelatihan

| Butir | Paper (rujukan) | Sebelum | Sesudah | Status |
|---|---|---|---|---|
| Lapisan tersembunyi G, E, D | Satu lapisan + keluaran (pers. 32-34) | 2 lapisan | Default **1** (`--n-hidden`, dapat diubah) | V |
| Aktivasi | "such as a ReLU" (pers. 32-34) | ReLU | ReLU | V |
| Keluaran G, E | Linear (pers. 33-34) | Linear | Linear | V |
| Keluaran D | sigmoid (pers. 32) | logit + BCE-with-logits | sama (sigmoid di kerugian/inferensi) | V |
| Masukan D | "concatenate()"; Fig. 6 | (x, z) | (x, z); nyata = (x, E(x)), palsu = (G(z), z) | A (pers. 32/35 menulis D(x) satu argumen; Alg. 3 b.14 `concatenate(E(x), G(z))` tidak sejajar dimensi) |
| loss_D | pers. (35), Alg. 3 b.15 | BCE | BCE, diuji setara rumus | V |
| loss_G | -mean log D(G(z)) (pers. 36, Alg. 3 b.8) | non-saturating | sama, diuji | V |
| loss_E | pers. (37) vs Alg. 3 b.11/25 bertanda berlawanan | label dibalik | non-saturating pers. (37): -mean log(1 - D(x, E(x))) | A |
| Jadwal | N iterasi; K langkah D lalu 1 langkah G/E, batch baru (Alg. 3) | 1 langkah D + 1 G/E, **batch sama**, permutasi per epoch | K langkah D (`--d-steps`), lalu G/E dengan **batch baru**; batch diambil acak dengan pengembalian | V (jadwal); U (K, N, cara sampling) |
| Optimizer, lr, batch, epoch, dimensi latent, lebar lapisan, prior z | "gradient descent with learning rate alpha" | Adam 2e-4 (0.5, 0.999), 256, 20, z=32, 128, N(0,I) | sama | U |
| Data pelatihan | Trafik jaringan simulasi (Sec. III-D-2) | NSL-KDD normal | NSL-KDD normal (fold train) | Deviasi: NSL-KDD hanya disebut di Sec. VI (future work) |

### 2a.2 Skor deteksi, arah, dan ambang

| Butir | Paper | Sebelum | Sesudah | Status |
|---|---|---|---|---|
| Mekanisme | Fig. 6 testing: Encoder -> Concatenate -> Discriminator -> Normal/Malicious | -logit D(x, E(x)) (sejak commit F1) | sama | V (struktur); U (skor numerik) |
| Arah | Tidak disebut; rujukan [56] (Xu et al.): anomali dinilai "palsu" oleh D | Diasumsikan | Diasumsikan **dan diverifikasi** di validasi (`validate_direction`: AUC > 0.5); checkpoint gagal tidak disimpan dan ditolak saat dimuat; skor **tidak pernah dibalik otomatis** | Arah sesuai [56] |
| Aturan | Tidak disebut | skor > ambang | skor > ambang (ketat), diuji | U |
| Ambang | Tidak disebut | Dipilih di **seluruh KDDTrain+** | Dipilih di **validasi saja** (`select_threshold`); kebijakan `f1_weighted` (prevalensi 10%) atau `fpr` | U |

### 2a.3 Preprocessing dan pemisahan data

| Butir | Sebelum | Sesudah |
|---|---|---|
| Split | Train = seluruh KDDTrain+; tanpa validasi | KDDTrain+ dipisah stratified 80/20 (train/validasi, seed tetap); KDDTest+ hanya untuk test |
| Scaler | Fit di seluruh KDDTrain+ (normal + serangan) | Fit di baris **normal fold train** saja (`--scaler-fit normal`; opsi `train`) |
| Kategori one-hot | `get_dummies` + `align` (kategori dari train) | Kategori dari fold train, urutan alfabet, kategori asing -> vektor nol |
| Skema | Tidak disimpan | `preprocessor/schema.json` + `scaler.npz` dengan `schema_hash`; dimensi dan urutan fitur dicek saat inferensi |
| API lama | `load()` | Dipertahankan (dipakai `export_decisions.py`; `feature_selection.py` diarsipkan) |

### 2a.4 Hasil pengukuran (NSL-KDD, 20 epoch)

AUC skor -logit D(x, E(x)) pada data **validasi** (tanpa memakai test):

| Pelatihan | seed 1 | seed 2 | seed 3 |
|---|---|---|---|
| Baru, 1 lapisan (paper) | 0.071 | 0.066 | - |
| Baru, 2 lapisan | **0.765** | 0.183 | 0.161 |
| Lama (kode sebelum refaktor), 2 lapisan | ~0.83 (test) | 0.694 | 0.221 |

Checkpoint satu-satunya yang lolos (2 lapisan, seed 1) pada **KDDTest+**:
AUC = **0.488**, TPR = 20.8%, FPR = 0.98%, F1 (prevalensi 10%) sekitar 32%.
Ablasi reconstruction error (bukan metode paper) pada model yang sama: AUC test 0.855.

**Kesimpulan yang harus dicatat di laporan:**
1. Arah skor Discriminator **bergantung pada seed dan arsitektur**; pada sebagian
   besar run arahnya terbalik. Hal yang sama terjadi pada kode lama. F1 sekitar 61%
   pada sapuan ns-3 sebelumnya bergantung pada satu seed yang kebetulan berhasil.
2. Validasi dari KDDTrain+ **tidak mewakili** KDDTest+ (serangan baru di test):
   AUC 0.765 di validasi menjadi 0.488 di test.
3. Penyebab dugaan (belum dibuktikan): Discriminator ReLU mengekstrapolasi linear di
   luar distribusi latih, sehingga logit sampel asing bisa sangat tinggi. Hal ini
   terbukti pada data sintetis (`tests/test_bigan.py::TestKnownLimitations`).
4. Belum ada konfigurasi yang diverifikasi selaras paper sekaligus andal.

**Keputusan tim (fase 1):**
- **Mainline = skor Discriminator** (-logit D(x, E(x)), Fig. 6), dilaporkan apa
  adanya. Fase berikutnya hanya memakai checkpoint yang **lolos verifikasi arah**
  di validasi; ketidakstabilan arah dan hasil test dicatat sebagai temuan.
- **Pembanding (bukan mainline) = reconstruction error** ||x - G(E(x))||, dijalankan
  dengan `python detector_eval.py --ckpt <dir> --ablation-recon` (ambangnya sendiri
  dipilih di validasi). Hasil pembanding tidak dipakai untuk keputusan IDS.

## 2b. TLDQN (refaktor fase 2)

Berkas: `ml/honeycomb_topology.py` (topologi dan ruang aksi bersama),
`ml/routing_env.py` (lingkungan), `ml/tldqn_routing.py` (DQN, transfer,
evaluasi), `ml/tests/test_tldqn.py` (32 tes). TLDQN lama (lingkungan acak)
dipindah ke `ml/legacy_tldqn.py` hanya agar `export_decisions.py` tetap jalan
sampai fase integrasi NS-3.

### 2b.1 Celah yang ditutup

| Celah sebelum | Sesudah |
|---|---|
| Lingkungan acak: 6 jalur kandidat dengan fitur acak, satu "terbaik" ditanam; tidak terkait topologi | Lingkungan honeycomb dari koordinat `honeycomb-sdwsn.cc`, routing hop-by-hop dari sensor ke edge sink |
| Aksi tidak berhubungan dengan node nyata; `export_decisions.py` memakai 4 aksi (pilih sink) | Aksi = slot next-hop (K = 6), node id nyata; skema ditulis ke `checkpoints/tldqn/routing_schema.json` |
| Transisi tidak bergantung aksi (keadaan baru diundi ulang) | Paket berpindah ke node terpilih; hilang karena tautan atau relay |
| Transfer: dua lingkungan acak berbeda seed tanpa makna | Sumber 50 sensor -> target 100 sensor, skenario serangan dan stabilitas tautan berbeda |
| Target network diperbarui per episode | Tiap C langkah gradien (Alg. 1 b.24), diuji |
| Optimizer Adam | SGD (pers. 19), diuji |
| Evaluasi = reward pelatihan (bising, termasuk eksplorasi) | PDR dan imbalan harapan eksak kebijakan greedy pada himpunan evaluasi tetap; referensi random, greedy, oracle |

### 2b.2 Definisi dan status

| Butir | Paper | Implementasi | Status |
|---|---|---|---|
| Lokasi | Route agent di edge server; routing CH - CM | Rute sensor -> edge sink terdekat; CH tidak dimodelkan sebagai tujuan | Deviasi (A: paper tidak merinci peran CH dalam jalur ke sink) |
| Jangkauan | 150 m (Tabel 5) | 150 m untuk graf kandidat | V |
| Tetangga | Enam sel tetangga (Sec. III-B) | K = 6 slot; kandidat maksimum terjangkau = 6 pada semua n 20..100 (diuji) | V (jumlah); U (definisi kandidat) |
| Kandidat next-hop | Tidak disebut | Syarat progres ke sink (bebas loop) + sink selalu valid | U |
| Koordinat | Kode rilis | Sensor 23 dan 37 berimpit dengan sink 3 dan 2 (dari kode rilis) | Temuan |
| Keadaan | trust, distance, link quality, energy; hops, link stability, PDR, throughput | 7 fitur x 6 slot + 2 fitur node (lihat `routing_env.py`) | V (kriteria); U (bentuk) |
| Throughput | Kriteria rute | Tidak dimodelkan terpisah (laju tautan tetap) | Deviasi |
| Model kanal | Tidak disebut | PRR logistik (d50 = 120 m, lebar 10 m) x stabilitas tautan U(0.8, 1) | U |
| Trust | Tidak dirinci | Reputasi Beta dari 10 observasi; malicious meneruskan 30%, normal 98% | U |
| Reward | Tidak diberi rumus | +1 sampai, -1 hilang, -0.05 per hop, -0.1 x (1 - energi relay) | U |
| Bellman, MSE, SGD | pers. (17)-(20) | Target network, terminal -> y = r, max hanya atas aksi valid | V + U (terminal/masking) |
| Hiperparameter | Tidak disebut | gamma 0.9, eta 0.05, batch 64, |D| 20000, C 200, epsilon 1 -> 0.05 linear 1500 episode, MLP 2 x 128 | U |
| Pemilihan hiperparameter | - | Dari konvergensi DQN **tanpa** transfer saja (uji 1 seed, skenario 2001), tidak dari hasil transfer. Skenario uji ini sama dengan skenario target seed 1, sehingga bila ada bias, bias itu menguntungkan DQN tanpa transfer (konservatif terhadap klaim transfer). Uji ini tidak diarsipkan | - |
| Evaluasi | - | **Dalam skenario**: node malicious dan stabilitas tautan sama dengan saat pelatihan target; hanya energi dan observasi trust yang diundi ulang. Bukan evaluasi pada skenario yang belum pernah dilihat | - |
| TTL | - | DP referensi tidak memodelkan TTL; berlaku karena rute terpanjang = 12 hop pada n = 100, tepat di batas `max_hops` = 12 (dijaga tes `test_longest_route_within_ttl`) | - |

### 2b.3 Hasil (5 seed; target 100 sensor; 3000 episode sumber dan target)

Referensi PDR target: random 0.403, greedy progres 0.438, Bayes 0.720, oracle 0.721.
Gain ternormalisasi = (E[R] - E[R]_random) / (E[R]_oracle - E[R]_random).

**Validitas referensi (diverifikasi oleh tes):**
- Tidak ada referensi yang memakai informasi masa depan: semuanya menghitung nilai
  harapan dengan program dinamis, tidak memakai hasil acak yang terjadi, dan tidak
  membaca RNG lingkungan (`test_references_use_no_future_information`).
- **Oracle memakai informasi penuh** (label malicious sebenarnya), yang tidak
  dimiliki agen. Karena itu ditambahkan **referensi Bayes informasi setara**:
  keputusan dari E[p_fwd | trust teramati] (prior 10% malicious), dievaluasi pada
  dinamika sebenarnya. Keputusannya tidak bergantung pada label malicious, diuji di
  tingkat keputusan (`test_bayes_decisions_independent_of_malicious_labels`: kebijakan
  identik setelah label diacak, sedangkan kebijakan oracle berubah).
- Hasil: Bayes hampir identik dengan oracle (PDR 0.720 vs 0.721; gain akhir terhadap
  Bayes 0.831 / 0.839 vs terhadap oracle 0.831 / 0.838). Artinya celah ke 1.0
  **bukan** akibat asimetri informasi, melainkan suboptimalitas agen.
- Catatan desain: dengan 10 observasi, trust hampir sempurna memisahkan node
  malicious (30%) dan normal (98%). Asumsi ini (U) membuat identifikasi node
  malicious di lingkungan routing relatif mudah.
- Bayes bukan batas atas ketat: stabilitas tautan tetap selama pelatihan, sehingga
  agen secara prinsip dapat belajar lintas episode.

| Metrik | DQN tanpa transfer | TLDQN |
|---|---|---|
| Gain awal (episode 0) | -0.17 ± 0.11 | 0.71 ± 0.04 |
| Rata-rata gain sepanjang pelatihan | 0.66 ± 0.01 | 0.79 ± 0.02 |
| Gain akhir (3 evaluasi terakhir) | 0.83 ± 0.03 | 0.84 ± 0.01 |
| Gain akhir terhadap Bayes informasi setara | 0.83 ± 0.03 | 0.84 ± 0.01 |
| PDR akhir | 0.658 ± 0.016 | 0.664 ± 0.012 |
| Episode mencapai gain 0.9 (ambang utama, pra-ditetapkan) | tidak tercapai | tidak tercapai |
| Episode mencapai gain 0.8 (post hoc) | 1200, 1400, 1000, 1200, 1600 | 800, 700, 600, 1100, 1000 |

Perbandingan memakai anggaran episode **target** yang sama. TLDQN juga menerima 3000
episode pelatihan sumber, jadi total komputasi **tidak** sama; klaim yang didukung
adalah "lebih cepat di tugas target", bukan "lebih hemat komputasi total". TLDQN
juga memulai eksplorasi dari epsilon 1.0 seperti DQN (bobot sumber tidak dipakai
untuk eksplorasi awal).

Kesimpulan: transfer learning **mempercepat** pembelajaran dan memberi performa
awal yang jauh lebih baik, tetapi **tidak** menaikkan performa akhir. Hal ini
sejalan dengan klaim kualitatif paper (transfer mempercepat), tanpa klaim
peningkatan akhir. Oracle memakai ground truth malicious yang tidak dimiliki
agen, sehingga gain 1.0 tidak dapat dicapai.

### 2b.4 Belum terpenuhi (fase integrasi NS-3)

- ~~NS-3 masih memakai aksi lama~~ dan ~~belum membatasi jangkauan 150 m~~:
  ditangani di integrasi langkah 1 (bagian 2c).
- Model kanal Python adalah proksi pelatihan; perlu dikalibrasi dengan
  pengukuran NS-3.
- Rencana: NS-3 memakai **tabel kandidat hasil ekspor Python** (bukan menghitung
  ulang), karena jarak berseri banyak di kisi honeycomb dan sqrt C++ dapat berbeda
  1 ulp dari numpy. Indeks sink pada skema (n + k) adalah indeks abstrak; C++
  memetakannya lewat `sinks.Get(k)`, bukan NodeId langsung. Skema lengkap
  (normalisasi fitur, model kanal) ada di `ml/reports/routing_schema.json`.

## 2c. Integrasi TLDQN - NS-3 (langkah 1)

Berkas: `ml/export_routes.py` (ekspor tabel next-hop), `ml/tests/test_export_routes.py`
(10 tes), `ns3-sim/honeycomb-sdwsn.cc` (kelas `Tldqn`, `GrayholeRouting`, host route),
`ns3-sim/tools/syntax_check.sh` (pemeriksaan C++ terhadap header ns-3.48 asli).

### 2c.1 Mekanisme

| Butir | Sebelum | Sesudah |
|---|---|---|
| Keputusan routing di NS-3 | Pilih 1 dari 4 sink (TLDQN lama, lingkungan acak); paket dikirim langsung | Tabel next-hop TLDQN per (sensor, sink tujuan) dipasang sebagai host route /32; paket diteruskan **multi-hop** |
| Jangkauan radio | Tidak terbatas (semua node saling dengar) | 150 m (`RangePropagationLossModel`, Tabel 5); `--maxRange 0` mencabut batas jangkauan saja (rute multi-hop tetap) |
| Tujuan aliran sensor | Sink pilihan TLDQN lama | Sink terdekat (aturan topologi bersama Python) |
| Kandidat next-hop | - | Dihitung **hanya di Python**; C++ memvalidasi (rute bebas loop, setiap sensor mencapai sinknya, next-hop dalam jangkauan) dan berhenti dgn error bila tidak konsisten |
| Ground truth malicious | `bigan_decisions_<n>_s<seed>.csv` | Sama, dipakai juga oleh ekspor rute (satu sumber) |
| Relay malicious | Tidak berperilaku jahat saat meneruskan (hanya flooding sebagai sumber) | **Grayhole**: menjatuhkan paket transit dgn peluang 0.70 (= 1 - p_fwd_malicious lingkungan pelatihan); dicatat FlowMonitor sebagai drop |
| Alarm delegator | Ke 4 sink | Ke sink tujuannya sendiri (sink lain tak terjangkau dgn jangkauan 150 m) |
| Metrik baru | - | `planned_hops_mean`, `planned_hops_benign_mean`, `benign_hops_per_rx_est` (= 1 + penerusan/diterima pada aliran sensor normal; sedikit di atas hop nyata karena penerusan paket yang kemudian hilang ikut terhitung), `grayhole_drops`, `grayhole_not_dropped`, `route_entries`, `max_range_m` |

### 2c.2 Asumsi dan deviasi

| Butir | Keterangan | Status |
|---|---|---|
| Trust saat ekspor | Reputasi pra-deploy **tersimulasi** dari ground truth (model observasi lingkungan), bukan telemetry NS-3 | U, sementara (langkah 2) |
| Energi saat ekspor | 1.0 untuk semua sensor (keputusan di awal simulasi) | U |
| Stabilitas tautan | Proksi Python (scenario_seed = 10000 seed + n); NS-3 memakai fading Nakagami tanpa path loss di dalam 150 m | Deviasi: model kanal Python vs NS-3 belum dikalibrasi |
| Relay normal | Lingkungan pelatihan: meneruskan 98%; NS-3: 100% + kehilangan alami MAC/kanal | Deviasi kecil |
| Rute statis | Ditentukan sekali di awal; tidak diperbarui saat node diisolasi | Langkah 5 |
| Node terisolasi IPS | Berhenti mengirim data sendiri, tetapi **masih meneruskan** paket transit. Khususnya penyerang yang sudah diblokir **tetap menjatuhkan** paket transit (grayhole dipasang sejak awal, tidak membaca status blokir). Jadi pada langkah ini kualitas IDS hanya memengaruhi beban flooding, **tidak** mengurangi drop grayhole; tren PDR vs akurasi deteksi belum boleh ditafsirkan sebagai manfaat IPS | Langkah 5 |
| Bukti energi DTO | `evidence = 0.6 x laju kirim + 0.4 x energi terpakai` (dinormalisasi). Dgn jangkauan 150 m, relay dekat sink menerima/meneruskan jauh lebih banyak sehingga energinya tinggi, sedangkan grayhole (menjatuhkan 70% transit) tampak hemat. Bukti energi kini **tercampur beban relay**; hasil `dto_*` dapat bergeser dibanding sweep sebelumnya | Diperbaiki di langkah 3 (DTO dgn telemetry); bandingkan dgn run kontrol `--maxRange=0` |
| Trafik TCP user -> cloud | Dgn jangkauan 150 m tidak ada rute user -> cloud; aliran TCP sebagian besar gagal. Default `tcpUsers=false`; bila dinyalakan, NS-3 mencetak peringatan dan metrik `tcp_*` tidak bermakna | Belum ditangani |
| Kanal di dalam 150 m | Tanpa path loss (hanya Nakagami): semua tautan dalam jangkauan punya daya rata-rata sama. Proksi PRR Python (d50 = 120 m) jauh lebih pesimis untuk tautan panjang. Jangkauan interferensi = jangkauan komunikasi (hidden terminal, tanpa RTS/CTS). PDR, latency, dan energi **tidak** sebanding langsung dgn sweep sebelum langkah 1 | Deviasi |
| Konsistensi tabel | Kolom `node_malicious` di tabel; NS-3 berhenti dgn error bila berbeda dari `bigan_decisions_<n>_s<seed>.csv` (cegah tabel basi) | - |
| Model TLDQN | `ml/experiments/phase2_tldqn/tldqn/checkpoints/tldqn_seed1.pt` (dilatih pada skenario Python 100 sensor), dipakai tanpa fine-tune untuk semua skenario NS-3 | U |
| Referensi pembanding | Tabel greedy progres terbesar (`ns3-sim/results/routes_greedy/`), bukan metode | - |

### 2c.3 Hasil yang sudah diverifikasi (tanpa menjalankan NS-3)

- Tabel 27 skenario (n 20..100, seed 1..3): semua sensor mencapai sinknya, bebas loop.
- TLDQN berbeda dari greedy pada sekitar 32-85% entri (tergantung n); pada 100 node
  rata-rata hop terencana 2.3 (greedy 2.1).
- Sumber yang rutenya melewati relay malicious (100 node, seed 1/2/3): TLDQN 3/7/11,
  greedy 9/31/21.
- `honeycomb-sdwsn.cc` lolos `clang++ -fsyntax-only` terhadap header ns-3.48 asli
  (uji kontrol: signature `RouteInput` yang sengaja salah terdeteksi).

### 2c.4 Hasil run NS-3 (sweep 9 ukuran x 3 seed, berpasangan TLDQN vs greedy)

Diverifikasi di server: 54 run, sekuensial, tanpa duplikat; run ulang identik dgn run
tunggal sebelumnya (deterministik per seed). Analisis: `ml/analyze_routing.py`
(`ml/reports/routing_ns3_step1.json`, `figures/real/routing_tldqn_vs_greedy.png`).

| Metrik (rata-rata 27 skenario) | TLDQN | Greedy | Skenario TLDQN lebih baik / lebih buruk | Uji tanda |
|---|---|---|---|---|
| PDR (%) | 98.09 | 92.04 | 21 / 5 | p = 0.0025 |
| Paket dijatuhkan grayhole | 63.5 | 241.1 | 21 / 4 | p = 0.0009 |
| Throughput (kbps) | 22.80 | 21.49 | 23 / 3 | p < 0.001 |
| Latency (ms) | 0.754 | 0.666 | 9 / 18 | p = 0.12 (tidak signifikan) |
| Energi (J) | 54.27 | 54.25 | 2 / 25 | p < 0.001; selisih +0.02 J (~0.04%) |
| Hop per paket (sensor normal) | 2.02 | 1.91 | - | - |

Cek kewajaran: rasio drop grayhole gabungan 0.699 (target 0.70); metrik Bi-GAN identik
antar kebijakan routing (berasal dari berkas keputusan).

Tafsiran dan batasan:
- Keputusan TLDQN **memengaruhi** penerusan paket NS-3 secara terukur (kriteria 5):
  menghindari relay malicious sehingga PDR dan throughput naik, dibayar sekitar 0.1 hop
  lebih banyak dan energi yang sedikit lebih tinggi secara konsisten.
- Pada n = 20 greedy lebih baik (PDR 98.0 vs 95.6); keunggulan TLDQN tidak seragam di
  semua ukuran jaringan.
- **Keunggulan ini sebagian berasal dari trust tersimulasi dari ground truth** saat
  ekspor (hampir mengungkap node malicious, lihat 2b.3), sedangkan greedy tidak memakai
  trust. Ini batas atas kasar untuk routing berbasis trust, BUKAN bukti kinerja dgn trust
  terukur. Setelah langkah 2 (telemetry), perbandingan ini harus diulang.
- Hanya 3 seed per ukuran; grayhole memakai asumsi perilaku yang sama dengan lingkungan
  pelatihan TLDQN.
- DTO dgn jangkauan 150 m: TLDQN 136/162 penyerang terdeteksi (FP 26), greedy 128/162
  (FP 34); pada run kontrol tanpa batas jangkauan DTO sempurna (10/10, FP 0) -> bukti
  energi DTO tercampur beban relay (langkah 3).

### 2c.5 Belum diverifikasi

- Apakah PDR NS-3 dgn tabel TLDQN lebih tinggi dari tabel greedy (hipotesis, belum diukur).
- Pemeriksaan run pertama di server: rasio `grayhole_drops / (drops + not_dropped)` sekitar
  0.70; `benign_hops_per_rx_est` mendekati `planned_hops_benign_mean`; run kontrol
  `--maxRange=0`; perubahan `dto_*` dibanding sweep sebelumnya; drop ARP di detik pertama
  (antrean ARP bawaan = 3 paket).

### 2c.6 Syarat langkah 2 (telemetry) -- ditetapkan tim

Telemetry per node HARUS berasal dari aktivitas aktual NS-3, **bukan** dari ground
truth atau label malicious yang ditentukan sebelumnya. Label hanya boleh dipakai
untuk (a) mengatur perilaku penyerang di simulator dan (b) evaluasi. Data per node:
paket dikirim dan diterima, jumlah penerusan, rasio drop, laju paket rata-rata,
konsumsi energi, perilaku tetangga, dan kualitas tautan. Konsekuensi: trust
tersimulasi pada `export_routes.py` (bagian 2c.2) harus diganti dgn nilai dari
telemetry ini.

## 2d. Telemetry NS-3 (langkah 2)

Definisi lengkap ada di `docs/TELEMETRY.md`. Kode: `namespace telem` di
`honeycomb-sdwsn.cc`, `ml/telemetry_schema.py`, `ml/telemetry_trust.py`,
`ml/telemetry_eval.py`, `ml/tests/test_telemetry.py`. TLDQN, Bi-GAN, DTO, IPS,
jangkauan radio, dan model trafik **tidak diubah**.

### 2d.1 Asumsi dan deviasi

| Butir | Paper | Implementasi | Status |
|---|---|---|---|
| Sumber perilaku tetangga | "investigate agent memantau perilaku node" (Sec. III-D-1); mekanisme tidak disebut | Watchdog pasif: observer mendengar penerusan oleh tetangga lewat PHY (frame di-decode benar) setelah handoff MAC ber-ACK | U |
| Kategori observasi | - | forwarded / forwarded_late / unobserved_rxfail (ambigu) / unobserved_clean; tidak terdengar **tidak** otomatis malicious | U |
| Batas waktu | - | `wdTimeout` = 0.1 s, `wdLate` = 1.0 s (di atas `WifiMacQueue::MaxDelay` 0.5 s) | U |
| Ambigu karena kegagalan terima | - | Ditentukan dari waktu saja (observer tidak tahu isi frame gagal); diaudit dengan kebenaran simulator | U |
| Mode promiscuous | - | Tidak dipakai (di ns-3.48 mengubah jalur MAC); diganti trace RxOutcome PHY | Pilihan teknis |
| Window | 600 s (Fig. 18); window tidak disebut | 30 s mulai t = 0, terpisah dari window Bi-GAN (mulai 8.3 s) | U |
| Trust | Disebut sebagai kriteria forwarder; rumus tidak disebut | Reputasi Beta(1,1) dari watchdog, offline dua tahap (tahap-1 trust 0.5); kategori ambigu dikoreksi dengan laju kegagalan kebetulan yang diukur per observer (`corrected`); `discard` sebagai pembanding | U |
| Validasi FlowMonitor | - | Kesamaan eksak hanya untuk UDP; TCP dilaporkan terpisah (FlowMonitor mengabaikan penerusan paket yang tidak dilacak lagi) | - |
| Energi per window | - | Dari TracedValue model radio, dibagi linear per interval keadaan PHY (eksak) | - |
| Counter mandiri (`self_*`) | - | Ditulis, tetapi **dilarang** sebagai bukti trust karena ns-3 melaporkannya jujur | Keputusan kami |

### 2d.2 Konsekuensi yang sudah terlihat (tanpa run NS-3)

- Model trafik (50 paket/node, interval `20/n` s) membuat hampir semua trafik
  sensor terjadi di 10 s sampai 50 s pertama. Pada n besar, telemetry per window
  praktis hanya terisi di window 0-1.
- Tabel tahap-1 (`ns3-sim/results/routes_stage1/`, trust 0.5) merutekan 1-26
  sumber per skenario lewat relay malicious; dengan trust tersimulasi langkah 1
  hanya 0-2. Ini disengaja, karena hanya relay yang dipakai yang bisa diamati.
  Malicious yang bukan relay tetap tidak teramati (trust 0.5).
- Bi-GAN tetap memakai fitur NSL-KDD; pemetaan telemetry ke fitur Bi-GAN belum
  dilakukan (menunggu tinjauan metodologi).

### 2d.3 Hasil run server (rincian di `docs/TELEMETRY.md` bagian 5)

- 39 run valid: skema, agregasi, audit, energi, kirim/terima vs FlowMonitor (sama
  persis). Penerusan vs FlowMonitor baru diperiksa dengan syarat perlu (lihat 2d.4).
  Seed ulang identik 3/3, telemetry pasif identik 3/3.
- Rasio penerusan grayhole: kebenaran audit 0.295, estimasi terkoreksi 0.314. Kontrol
  `grayholeDrop=0`: malicious 0.997 vs normal 0.999 (tanpa kebocoran label).
- Watchdog: tidak ada `forwarded` palsu. Semua `unobserved_clean` benar-benar tidak
  dipancarkan `j`, termasuk 42 gagal penerusan node normal (bukan malicious).
- Trust telemetry: AUC 1.000 pada sensor teramati di 27/27 skenario; hanya 71/162
  node malicious teramati.
- Tabel tahap-2: sumber lewat malicious 169 -> 76. TLDQN masih memilih malicious
  bertrust rendah pada 38 entri. Uji sensitivitas: TLDQN peka trust secara umum (81%
  keputusan berubah bila trust = 0), tetapi pada entri itu kandidat malicious sering
  punya link quality terbaik (alternatif lebih baik hanya 16/38). Hipotesis: kebijakan
  menukar trust dengan link quality secara suboptimal. Belum disimpulkan; TLDQN tidak
  diubah di langkah ini.
- Run NS-3 tahap-2 (27 skenario, berpasangan vs tahap-1): PDR 97.08 vs 92.73
  (lebih baik 22/26, p = 0.0005); drop grayhole 106 vs 233; throughput naik
  (p < 0.001); energi +0.011 J (25/27 lebih buruk); latency +0.17 ms (tidak
  signifikan, p = 0.052). Masih di bawah trust tersimulasi dari label (PDR 98.09), yang
  merupakan batas atas informasi dan dibuat dengan binary sebelum telemetry (diff
  sejak itu hanya kode telemetry).

### 2d.4 Koreksi klaim

- Klaim awal "penerusan telemetry = FlowMonitor `timesForwarded`" salah.
  `timesForwarded` hanya menghitung paket yang sampai (`flow-monitor.cc:259`).
  Counter `telemetry_ip_forward_udp_delivered` ditambahkan dan sama persis dengan
  FlowMonitor di 30 run binary baru (27 `_stage2` + 3 `_stage1v2`).
- Toleransi energi awal (relatif 1e-7) terlalu ketat untuk kuantisasi `Time` di
  `BasicEnergySource`; diganti batas yang diturunkan dari mekanismenya.

## 2e. DTO offline di atas telemetry (langkah 3)

Rincian di `docs/STEP3_DTO_DESIGN.md` (bagian 10 berlaku). Kode: `ml/dto.py`,
`ml/dto_detect.py`, `ml/dto_eval.py` (satu-satunya tempat label),
`ml/run_dto_step3.py`, `ml/dto_proxy_validation.py`, `ml/tests/test_dto.py`.
`class Dto` C++, TLDQN, Bi-GAN, dan telemetry **tidak diubah**. Tidak ada run NS-3 baru.

### 2e.1 Asumsi dan deviasi

| Butir | Paper | Implementasi | Status |
|---|---|---|---|
| Pers. (21)–(31) | Tertulis | Diterapkan literal, diuji per persamaan | Sesuai |
| Rand pada (22) | [0, 2] | U(0,1) | U (K6) |
| nID (24) | `[0.1n(1 − q/Y)]` | floor, minimal 1; instruktur dihitung ulang per anggota; k_i sama di fase 2 | U |
| Batas posisi | Tidak disebut | Klip ke [lb, ub) | Keputusan kami |
| Pengkodean | "best candidate = malicious node" | 1 kandidat = 1 node; OF = −PLR_j; akibatnya DTO ≈ argmax | U (K1) |
| Kriteria energi, PDR, PLR | Ketiganya disebut | Hanya PLR (PDR = 1 − PLR, redundan). Energi **Not Available**: tidak ada besaran teramati tetangga yang valid sebagai energi (validasi 10.2); `self_*` dilarang | U (K2, K3) |
| Multi-deteksi, aturan berhenti | Tidak disebut; klaim "10% identified" | Isolasi berurutan + uji Beta `P(q_j > q0 + 0.1) ≥ 1 − 0.01/M`, tanpa label dan tanpa k a priori; `paper_k` (k = 10%) hanya pembanding | U (K4) |
| m, Y, N_min | Tidak disebut | 30, 100, 10 (konfigurasi reproduksi) | U (K5) |
| Waktu eksekusi | Tidak disebut | Offline: akhir run dan akhir window 0; integrasi online di langkah 4–5 | Keputusan kami (K8) |
| Keandalan observer | Tidak disebut | Observer dengan laju kegagalan terima kebetulan b_i ≥ 0.8 (diukur pada watch terdengar, kegagalan PHY observer sendiri) tidak memberi bukti N_j; target tanpa bukti cukup = `insufficient_evidence`. Hanya DTO; trust/routing langkah 2 tetap | Keputusan kami (K9, disetujui) |

### 2e.2 Hasil pra-K9 (pool skenario; evaluasi utama stage2 + ctrl0; riwayat)

- stage2/main: TP 33, FP 0, FN 129 (P 1.000, R 0.204; R pada node berbukti 0.971).
  Recall dibatasi cakupan: hanya 34/162 malicious pernah menjadi relay teramati.
- ctrl0/main (tanpa grayhole): 0 deteksi, 0 FP.
- `paper_k`: precision 0.204 (FPR 0.088), karena k = 10% memaksa isolasi node normal.
- Ablasi aktivitas TX ("bukan energi"): P 0.186, R 0.623, FPR 0.303. Tidak layak.
- Hampir sama dengan ambang trust telemetry sederhana (P 1.000, R 0.198); DTO tidak
  menambah daya deteksi di atas bukti watchdog.
- Determinisme 3/3; stabil terhadap seed DTO 1–4 dan (m, Y) = (10, 30), (50, 200).
- Flooding tidak terdeteksi karena kriteria energi Not Available.

### 2e.2b Hasil pasca-K9 (berlaku; rincian `docs/STEP3_DTO_DESIGN.md` 10.8)

- stage2/main: TP 31, FP 0, FN 131 (P 1.000, R 0.191; R pada node berbukti 1.000;
  cakupan 31/162). ctrl0/main: 0 deteksi, 0 FP. stage1/main: TP 67, FP 0 (R 0.414).
- Dampak K9 (varian main, stage2 + ctrl0 + stage1), dengan definisi eksplisit:
  8 node malicious berubah keputusan, yaitu 2 `normal` → `insufficient_evidence`
  (stage2 n=80 s=2 node 3 = FN berbukti grayhole; ctrl0 n=100 s=1 node 32 = malicious
  di kontrol tanpa grayhole, tidak men-drop) dan **6 TP hilang** (2 stage2, 4 stage1)
  karena bukti drop-nya berasal dari observer b ≥ 0.8. Node normal: 0 berubah ke/dari
  `malicious`; 30 berubah `normal` → `insufficient_evidence` (TN/FPR tetap, cakupan
  turun).
- Recall DTO stage2 (0.191) kini sedikit di bawah ambang trust sederhana (0.198).
- b mencakup tabrakan di sekitar observer, jadi observer di area padat lebih mungkin
  dikecualikan walau target-nya jujur.
- Sensitivitas (m, Y) = (10, 30) turun menjadi 26/27: DTO dengan populasi/iterasi kecil
  tidak menemukan argmax di langkah pertama dan varian main berhenti (kasus nyata
  aturan berhenti tidak monoton). (30, 100) dan (50, 200) tidak terdampak.

### 2e.3 Keterbatasan terbuka

- Observer dengan laju kegagalan kebetulan b mendekati 1 menghasilkan "tidak ada
  drop" yang tidak informatif. Dari 2 node malicious yang berubah `normal` →
  `insufficient_evidence` karena K9, hanya stage2 n=80 s=2 node 3 yang merupakan drop
  grayhole tersembunyi; ctrl0 n=100 s=1 node 32 tidak men-drop (kontrol tanpa grayhole).
  **K9 disetujui dan diterapkan** (lihat 2e.2b); harga yang dibayar adalah 6 TP dan
  cakupan 30 node normal.
- Kriteria energi baru bisa dipulihkan bila telemetry memisahkan frame milik node
  sendiri dari frame transit (K10, **ditunda** ke langkah 4–5; tidak ada proksi baru).
- Pilihan PLR sebagai skor utama dan penolakan proksi `A_j` juga dipengaruhi tabel AUC
  berlabel `_stage1`, sehingga `_stage1` hanya hasil sekunder.

## 3. Definisi metrik

| Metrik | Paper | Reproduksi | Perbedaan |
|---|---|---|---|
| Energi | total - sisa (BasicEnergySource) | sama, hanya sensor, arus radio kelas sensor (idle 0.5 mA, RX 18.8 mA; Tx dari Tabel 5) | Arus bawaan ns-3 (idle 273 mA) tidak realistis untuk sensor; satuan: total J semua sensor |
| Latency, throughput, PDR, drop | Contoh kode point-to-point sederhana (Code 2-4, 8) | FlowMonitor pada **trafik UDP sensor**; TCP terpisah | Skala tidak sebanding dgn paper |
| Network lifetime | Code 5: sisa energi maks / (1000 x 0.1 W) | **Proyeksi** = energi awal / daya rata-rata per node selama simulasi | Nilai besar karena sebagian besar waktu idle; tidak sebanding dgn 400-580 s |
| Computation overhead | RT/CT x 100% | RT = waktu **per keputusan** (3D Cube + Bi-GAN + TLDQN, terukur setelah warm-up), CT = RT + latency | Nilai sangat kecil; tidak sebanding dgn 52.8% di Tabel 7 |
| Control overhead | pesan kontrol per detik | Hitungan pesan **logis** (auth, join, route req/reply, laporan IDS, flow-rule, alarm) per detik | Bukan paket yang benar-benar dikirim |
| Detection accuracy | (TP+TN)/total | Dipool seluruh window monitoring (sejak t = 8.3 s) | Lebih banyak sampel dari paper |
| Precision/recall/F1 | sepanjang 600 s | Dipool seluruh window; juga time-series per window | |
| Confusion matrix | Fig. 19 | Dipool seluruh window dan seed (100 node) | |

## 4. Hal teknis lain

- Autentikasi, clustering, routing, IDS dijadwalkan pada 0 s, 1.3 s, 8.3 s,
  15.3 s mengikuti kode rilis, dengan **urutan diperbaiki** (routing sebelum data
  mengalir) sesuai alur paper (Sec. IV-B).
- Hasil dilaporkan rata-rata +- simpangan baku lintas seed (default 3-5 seed).
  Paper tidak menyebut ulangan.
- NetAnim dimatikan secara default untuk sweep (`--anim=true` untuk demo).

## 4a. Batas memori (hasil pengukuran di server uji)

Server uji: 6.2 GiB RAM (sekitar 4.4 GiB tersedia), tanpa swap, 2 CPU.

| Pengukuran | Hasil |
|---|---|
| 100 node, profil `code`, 10 paket/node | puncak 490 MB, 7.9 s |
| 100 node, profil `code`, 25 paket/node | puncak 1053 MB, 17.4 s |
| 100 node, 100 paket/node | melewati 2.9 GB dan masih naik (dihentikan) |
| 100 node, 1000 paket/node | 4.5 GB dalam 73 s, dibunuh sistem (SIGKILL) |
| 20 node, TCP user -> cloud menyala | puncak 4.6 GB |

Dari dua titik pertama: memori puncak pada 100 node kira-kira
`115 MB + 37.5 MB x (paket per node)` (ekstrapolasi linier, bukan hasil ukur untuk
nilai lain). Dengan 1000 paket per node kebutuhan sekitar 37 GB. Karena itu paket
per node diturunkan ke 50 dan TCP user -> cloud dimatikan.

Yang sudah dicek dan **bukan** penyebab utama: WiFi 802.11ac (profil 802.11a sama
boros), lama simulasi, ukuran paket variabel. FlowMonitor menyumbang sekitar
seperlima laju kenaikan, bukan penyebab utama. Penyebab pasti di dalam ns-3
belum ditemukan (memori yang tertahan sekitar 280 KB per paket terkirim).

## 5. Yang **tidak** bisa direproduksi dan alasannya

1. **Angka Tabel 7 dan Fig. 9-17.** Kode rilis tidak menghasilkan angka itu:
   kelas TLDQN/BiGAN/DTO hanya mengembalikan string, dan baseline (PSO-ABC, OL-RL,
   RBP-DT) tidak ada. `archive/figures-reference-table7/` hanya kurva acuan dari Tabel 7, **bukan
   hasil simulasi kami**.
2. **Perbandingan dengan baseline.** Baseline tidak disimulasikan di repo ini.
   Ini **keputusan cakupan**: proyek hanya mereplikasi model HieMulti-Block, jadi
   baseline (PSO-ABC, OL-RL, RBP-DT) dan skenario pembanding berada di luar
   ruang lingkup.
3. **3D Cube berbasis DNN/MCTS dan konsensus blockchain.** Tidak ada detail yang
   cukup di paper/kode rilis untuk implementasi setia.

## 6. Yang belum diverifikasi

- Build dan run profil `paper` di ns-3.48 (pengaturan 802.11ac 80 MHz
  `ChannelSettings`, atribut arus radio). Jika gagal, gunakan `--wifiStd=a` atau
  `--profile=code`.
- Kecocokan penuh nilai default ns-3.48 dengan ns-3.26 (mis. model propagasi,
  parameter MAC) tidak dicek.
