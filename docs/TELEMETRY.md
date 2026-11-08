# Telemetry NS-3 (integrasi langkah 2)

Kode: `ns3-sim/honeycomb-sdwsn.cc` (`namespace telem`), skema dan validator
`ml/telemetry_schema.py`, trust `ml/telemetry_trust.py`, evaluasi
`ml/telemetry_eval.py`, uji `ml/tests/test_telemetry.py`.

## 1. Prinsip

1. Semua nilai berasal dari trace ns-3.48: `Ipv4L3Protocol` (SendOutgoing,
   UnicastForward, LocalDeliver, Rx, Drop), `WifiMac` (AckedMpdu, NAckedMpdu,
   DroppedMpdu, MpduResponseTimeout), `WifiPhy` (PhyTxBegin, PhyRxDrop),
   `WifiPhyStateHelper` (RxOutcome), dan `WifiRadioEnergyModel`
   (TracedValue TotalEnergyConsumption).
2. Kode telemetry tidak membaca label malicious (`g_bigan`, `actualMal`) maupun
   keadaan `GrayholeRouting`. Uji `TestCppContract` memeriksa ini secara statis.
3. Telemetry bersifat pasif. Kode ini tidak menjadwalkan event, tidak memakai RNG,
   tidak memanggil `GetRemainingEnergy` selama simulasi (yang memicu
   `UpdateEnergySource`), dan tidak mengaktifkan mode promiscuous MAC. Mode
   promiscuous di ns-3.48 ikut meneruskan subframe A-MPDU yang gagal ke MAC dan
   memakai cache duplikat `MacRxMiddle`. Prediksi: metrik jaringan dengan
   `--telemetry=true` sama persis dengan `--telemetry=false` (diuji di server,
   bagian 5).
4. Label hanya digabungkan di `ml/telemetry_eval.py`, setelah telemetry dan trust
   selesai dihitung.

## 2. Watchdog

Observer `i` dan tetangga `j` keduanya sensor. Kunci paket IPv4 adalah
`(src, dst, identification, protocol)`. Kunci ini tidak berubah antar hop (ns-3
memberi identification per tuple `{src, dst, proto}`).

| Tahap | Definisi |
|---|---|
| Mulai | MAC `i` menerima ACK/BlockAck untuk MPDU data ke `j` (`AckedMpdu`), dan paket itu transit bagi `j` (tujuan IP != `j`). Satu watch per `(i, j, paket)`. |
| Tidak dimulai | Handoff yang gagal di MAC `i` (retry habis, `DroppedMpdu`). Ini kehilangan tautan `i -> j`, bukan kesalahan `j`; dicatat di `mac_handoff_drop_*`. |
| Dengar | PHY `i` men-decode **benar** (RxOutcome, status MPDU = true) frame data dari `j` (Addr2 = MAC `j`) yang berisi paket transit dengan kunci yang sama. Setara radio mode monitor. Retry dari `j` ikut dihitung. |
| Delay | `d = t_dengar_pertama - t_ack`. `d < 0` dibolehkan (sampai `-wdLate`): `j` sudah meneruskan sebelum `i` menerima ACK, misalnya ACK pertama hilang. |

| Kategori | Syarat | Dipakai trust |
|---|---|---|
| `forwarded` | `d <= wdTimeout` (0.1 s) | positif |
| `forwarded_late` | `wdTimeout < d <= wdLate` (1.0 s) | positif |
| `unobserved_rxfail` | tidak terdengar sampai `wdLate`, **dan** PHY `i` mencatat >= 1 kegagalan terima (RxOutcome gagal atau PhyRxDrop) di `[t_ack, t_ack + wdLate]` | ambigu (mungkin tabrakan atau kanal); dikoreksi, lihat bagian 4 |
| `unobserved_clean` | tidak terdengar sampai `wdLate`, tanpa kegagalan terima di observer | negatif (timeout observasi; **bukan** otomatis malicious) |

- `wdLate` = 1.0 s dipilih di atas `WifiMacQueue::MaxDelay` (0.5 s), sehingga MPDU
  yang lebih tua dari itu sudah dibuang antrean MAC `j`. Paket yang tertahan di
  antrean ARP (`WaitReplyTimeout` 1 s x 3) bisa melewati batas ini. Kasus itu
  tercatat di `*_heard_after_timeout` dan di audit sebagai `j_tx_after_wdlate`.
- Klasifikasi dihitung setelah `Simulator::Run` dari stempel waktu. Hasilnya setara
  watchdog online dengan timer `wdLate`, tanpa menjadwalkan event.
- Watch yang dimulai kurang dari `wdLate` sebelum akhir simulasi tidak
  diklasifikasikan. Jumlahnya tercatat di `telemetry_meta.watches_truncated_at_end`.
- Heuristik `unobserved_rxfail` hanya memakai waktu, karena observer nyata tidak
  tahu isi frame yang gagal di-decode. Akibatnya kategori ini bisa menyerap banyak
  watch pada jaringan padat. Kategori ini juga **asimetris**. Paket yang di-drop
  masuk ke sini bila kebetulan ada kegagalan apa pun di window (peluang b),
  sedangkan paket yang diteruskan masuk hanya bila frame penerusan itu sendiri gagal
  (peluang c). Membuang kategori ini membuat rasio penerusan relay malicious terbaca
  terlalu tinggi, makin parah saat beban naik. Untuk itu, aturan window yang sama
  juga dicatat pada watch yang **terdengar** (`wd_heard_rxfail`), sehingga b dapat
  diukur (bagian 4). Audit (bagian 3.3) mengukur ketepatan heuristik ini.
- Frame yang datang saat observer sedang memancarkan tetap terlacak sebagai
  kegagalan (`DropPreambleEvent(TXING)` -> PhyRxDrop). Yang lolos tanpa jejak hanya
  sinyal di bawah sensitivitas, yang praktis tidak terjadi dalam 150 m pada model
  kanal ini; audit `j_tx_no_rx_event_at_observer` mengukurnya.

## 3. Berkas keluaran (`--telemDir`, default `results/telemetry${OUT_TAG}`)

### 3.1 `telemetry_<n>_s<seed>.csv`: per sensor per window

Window `w` mencakup `[30w, 30(w+1))` detik sejak t = 0, ada `ceil(simTime/30)`
window, dan setiap (sensor, window) punya tepat satu baris.

| Kolom | Sumber trace | Definisi |
|---|---|---|
| `self_ip_orig_pkts/bytes` | SendOutgoing | paket IP yang dibuat node (aplikasi dan delegator); byte termasuk header IP |
| `self_ip_rx_pkts` | Rx | paket IP diterima dari device (termasuk paket yang akan diteruskan) |
| `self_ip_fwd_pkts/bytes` | UnicastForward | paket yang diteruskan node (semua protokol dan fragmen; total jaringan UDP pada bagian 3.5 = definisi FlowMonitor) |
| `self_ip_local_pkts` | LocalDeliver | paket untuk node ini |
| `self_ip_drop_<alasan>` | Drop | `ttl`, `no_route`, `bad_checksum`, `iface_down`, `route_error`, `frag_timeout`, `duplicate`, `other` (= `Ipv4L3Protocol::DropReason`) |
| `self_mac_acked/nacked` | AckedMpdu / NAckedMpdu | MPDU yang dikirim node dan di-ACK / di-NACK (BlockAck) |
| `self_mac_drop_retry/other` | DroppedMpdu | retry habis / alasan lain (antrean penuh, lifetime habis) |
| `self_mac_ack_timeout` | MpduResponseTimeout | respons (ACK/CTS/BA) tidak datang |
| `self_phy_tx_mpdu/bytes` | PhyTxBegin | MPDU dipancarkan (termasuk retry dan kontrol) |
| `self_phy_rx_ok_mpdu/fail_mpdu` | RxOutcome | MPDU di-decode benar / gagal (payload) |
| `self_phy_rx_drop` | PhyRxDrop | frame dibuang sebelum payload (preamble, header, RXING, TXING, dll.) |
| `self_energy_j` | TotalEnergyConsumption | energi radio pada window ini. Tiap kenaikan dibagi linear atas interval keadaan PHY (arus konstan), jadi pembagian antar window eksak. Jumlah semua window = `energy_consumption_J`. |
| `wd_started_as_observer` | watchdog | watch yang dimulai node ini sebagai observer |
| `obs_watched` | watchdog | watch dengan node ini sebagai yang **diamati** (agregat semua observer), menurut window `t_ack` |
| `obs_forwarded`, `obs_forwarded_late`, `obs_unobserved_rxfail`, `obs_unobserved_clean` | watchdog | kategori bagian 2; jumlahnya = `obs_watched` |
| `obs_heard_after_timeout` | watchdog | bagian dari `obs_unobserved_*` yang akhirnya terdengar setelah `wdLate` |

Izin pakai:

- `obs_*` adalah satu-satunya masukan trust (`TRUST_INPUT_COLUMNS`).
- `self_*` diukur pada node itu sendiri. ns-3 melaporkannya jujur (tidak ada model
  pemalsuan), sehingga counter transit (`self_ip_fwd_*`, `self_ip_drop_route_error`)
  praktis mengungkap grayhole. Kolom ini **tidak boleh** dipakai sebagai bukti
  trust. Kegunaannya untuk validasi (FlowMonitor, energi) dan untuk bukti
  energi/laju pada langkah 3, setelah ditinjau.
- Tidak ada kolom label. Validator menolak nama kolom yang mengandung
  `malicious`, `label`, `attacker`, `grayhole`, `true_cause`.

### 3.2 `telemetry_links_<n>_s<seed>.csv`: per (window, observer, tetangga)

Baris hanya ditulis untuk pasangan yang punya aktivitas. `neighbor_type` bernilai
`sensor`, `sink`, atau `other` (MAC yang bukan sensor/sink digabung dengan id -1).

| Kolom | Definisi |
|---|---|
| `mac_handoff_acked/nacked/drop_retry/drop_other`, `mac_ack_timeout` | MAC observer -> tetangga (Addr1), per window kejadian |
| `rx_ok_from` | MPDU non-kontrol dari tetangga (Addr2) yang di-decode benar oleh observer |
| `rssi_mean_dbm`, `snr_mean_db` | rata-rata pada `rx_ok_from` (kosong jika 0) |
| `wd_started`, `wd_<kategori>`, `wd_heard_after_timeout` | watchdog observer -> tetangga sensor, per window `t_ack` |
| `wd_heard_rxfail` | watch yang terdengar (`forwarded` + `forwarded_late`) **dan** punya kegagalan terima di window yang sama; dipakai untuk mengukur laju kebetulan b |
| `wd_delay_mean_s` | rata-rata delay `forwarded` + `forwarded_late` (delay negatif dihitung 0) |

### 3.3 `watchdog_audit_<n>_s<seed>.csv`: kebenaran simulator, HANYA evaluasi

Format panjang: `window,observer,neighbor,wd_category,true_cause,count`. Penyebab
dihitung dari trace `j` (PhyTxBegin, Drop, UnicastForward, DroppedMpdu) dan trace
PHY `i` pada frame tersebut:

| `true_cause` | Arti |
|---|---|
| `j_tx_heard` | `j` memancarkan paket dan `i` men-decode-nya (sampai `wdLate`) |
| `j_tx_rxfail_at_observer` | `j` memancarkan, tetapi PHY `i` gagal pada frame itu (tabrakan/interferensi) |
| `j_tx_no_rx_event_at_observer` | `j` memancarkan, tanpa event PHY di `i` (di bawah sensitivitas, atau `i` sedang TX tanpa trace) |
| `j_tx_after_wdlate` | `j` memancarkan setelah `wdLate` (mis. antrean ARP) |
| `j_ip_drop_<alasan>` | `j` tidak memancarkan: dibuang di IP (`route_error` = jalur ErrorCallback; dalam simulasi ini hanya grayhole) |
| `j_ip_forwarded_mac_dropped` / `j_ip_forwarded_not_tx` | `j` meneruskan di IP, tetapi MPDU dibuang di MAC / tidak pernah dipancarkan |
| `j_no_ip_event` | tidak ada event IP di `j` untuk paket itu |

### 3.4 `telemetry_meta_<n>_s<seed>.json`

Berisi parameter (`window_s`, `wd_timeout_s`, `wd_late_s`, `route_dir`, profil),
jumlah watch per kategori, `watches_truncated_at_end`, dan total jaringan
(`net_ip_unicast_forward_udp_tcp`, `net_ip_orig_udp`, `net_ip_local_deliver_udp`).

### 3.5 Tambahan pada `metrics_<n>_s<seed>.csv`

| Telemetry | FlowMonitor | Definisi bersama |
|---|---|---|
| `telemetry_ip_forward_udp_delivered` | `flowmon_udp_times_forwarded` | UnicastForward UDP non-fragmen, dihitung per paket dan **ditambahkan saat paket sampai tujuan** (LocalDeliver pertama). Ini definisi FlowMonitor: `FlowMonitor::ReportLastRx` menambahkan penerusan paket ke `FlowStats.timesForwarded` (`flow-monitor.cc:259`), sedangkan `ReportDrop` membuang hitungannya. |
| `telemetry_ip_forward_udp_total` | - | semua UnicastForward UDP non-fragmen, termasuk paket yang kemudian hilang. Selisih dengan baris atas = penerusan paket hilang. |
| `telemetry_ip_forward_tcp_total` | `flowmon_tcp_times_forwarded` | idem TCP; **tidak** diklaim sama, karena FlowMonitor mengabaikan penerusan paket yang sudah tidak dilacak (mis. retransmisi segmen yang sudah diterima atau di-drop). TCP user mati secara default. |
| `telemetry_ip_orig_udp_total` | `flowmon_udp_tx_packets` | SendOutgoing UDP |
| `telemetry_ip_local_udp_total` | `flowmon_udp_rx_packets` | LocalDeliver UDP |

Ada juga `wd_watches`, `wd_<kategori>`, dan `sensor_energy_consumed_J` (12 digit
signifikan, agar bisa dibandingkan dengan jumlah per window). Selisih pada pasangan
UDP juga dicetak sebagai PERINGATAN di log.

Koreksi setelah run: versi pertama membandingkan `telemetry_ip_forward_udp_total`
dengan `timesForwarded` dan mengklaim keduanya sama. Klaim itu salah, karena
`timesForwarded` hanya memuat penerusan paket yang sampai. Pada 39 run, total
telemetry lebih besar 0 sampai 829. Selisihnya nol tepat pada run tanpa paket hilang,
dan berkorelasi 0.92 dengan drop grayhole. Counter `delivered` ditambahkan dan
verifikasi eksaknya menunggu run berikutnya (bagian 6).

## 4. Trust offline dua tahap

1. Tahap-1: `export_routes.py --trust-source none` (semua sensor 0.5, tanpa label)
   -> `ns3-sim/results/routes_stage1/` -> NS-3 dengan telemetry.
2. `telemetry_trust.py` (data masuk hanya lewat `telemetry_schema.trust_view`):
   - `corrected` (default). Laju kebetulan per observer adalah
     `b_i = sum wd_heard_rxfail / sum (wd_forwarded + wd_forwarded_late)`; observer
     tanpa watch terdengar memakai b gabungan. Estimasi drop:
     `D_j = min(A_j + U_j, sum_i U_ij / (1 - b_i))`, dengan batas diterapkan pada
     agregat node. Trust: `trust_j = (N_j - D_j + 1)/(N_j + 2)`.
     Asumsinya, waktu drop independen dari kegagalan terima di observer. Dengan
     asumsi itu, drop jatuh ke `clean` dengan peluang `1 - b_i`, dan estimator ini
     tak bias meskipun `b != c` (diuji dengan data sintetis).
     Arah bias yang diperkirakan: b diukur pada watch yang terdengar. Di sana window
     juga memuat kegagalan yang dipicu transmisi `j` sendiri (percobaan pertama
     gagal lalu retry terdengar, tabrakan akibat frame `j`), sedangkan drop tidak
     melibatkan transmisi `j`. Maka kemungkinan b >= b_drop, D terlalu besar, dan
     trust terlalu rendah. Arah ini konservatif (relay malicious tampak lebih buruk)
     tetapi menghukum relay normal di sel padat. Besarnya diperiksa lewat
     `est_corrected` kelas normal vs audit `true_tx_ratio` normal. Node dengan
     estimasi jenuh (D dipotong ke A + U) dilaporkan di `cap_hit_nodes` / `cap_hit`.
   - `discard` (pembanding sensitivitas): `trust = (F+1)/(F+U+2)`, dengan A dibuang.
   - Sensor tanpa observasi bernilai 0.5.
3. Tahap-2: `export_routes.py --trust-source telemetry --trust-dir ...` -> NS-3.

Syarat sebelum trust tahap-1 dipakai untuk tahap-2, dicek dari audit run tahap-1 di
`telemetry_eval.py`:

- `est_corrected` malicious mendekati rasio audit `true_tx_ratio` malicious.
- Fraksi `unobserved_clean` yang sebenarnya dipancarkan `j`
  (`unobserved_clean_observer_miss_frac`) kecil, sehingga bukti negatif memang
  berasal dari `j`.

Jika salah satu syarat gagal, kami laporkan dulu sebelum tahap-2.

Batasan:

- Sensor yang tidak pernah menjadi relay di tahap-1 tidak teramati, sehingga
  trust-nya tetap 0.5.
- TLDQN dilatih dengan trust Beta dari 10 observasi (rentang 1/12..11/12). Trust
  telemetry dengan banyak observasi bisa keluar dari rentang itu, dan Q-net
  mengekstrapolasi.

## 5. Validasi

| Validasi | Cara | Status |
|---|---|---|
Hasil run server (`ns3-sim/results/step2/`; laporan `ml/reports/telemetry_step2_*.json`):

| Validasi | Cara | Hasil |
|---|---|---|
| Skema, kelengkapan window, kategori = watched | `telemetry_schema.validate` | lolos, 39 run (27 stage1, 3 ulang, 9 kontrol). `all_valid` di laporan memakai syarat perlu untuk penerusan (`forward_check`) |
| Agregasi: node (diamati) = jumlah link per tetangga; observer = jumlah link; audit = telemetry; total = meta; MAC node = MAC link | idem | lolos, 39 run |
| Paket kirim / terima vs FlowMonitor | metrics CSV | sama persis, 39 run |
| Penerusan vs FlowMonitor | metrics CSV | definisi lama tidak sama (lihat 3.5); syarat perlu lolos di 39 run lama. Counter `delivered` **sama persis** dengan FlowMonitor di 30 run binary baru (27 `_stage2` + 3 `_stage1v2`). Contoh n = 60: delivered 2.562 = FlowMonitor 2.562, total 2.593; n = 100: 6.114 = 6.114, total 6.141 |
| Energi per window = energi sumber | idem | lolos dalam batas kuantisasi `BasicEnergySource` (bagian 6) |
| Rasio penerusan grayhole sekitar 0.30 | `telemetry_eval.py` | audit (kebenaran) 0.295; `est_corrected` 0.314; `est` (ambigu dibuang) 0.352; batas [0.294, 0.460]. Normal: 0.999 (semua metode) |
| Reproduksibilitas seed sama | `--repeat-dir` (SHA-256 node/link/audit) | identik 3/3 (n = 20, 60, 100). Binary baru vs lama (`_stage1v2` vs `_stage1`): identik 3/3, jadi counter baru pasif |
| Tanpa kebocoran label | (a) tanpa kolom label; (b) uji statis C++; (c) kontrol `--grayholeDrop=0` | (a), (b) lolos; (c) malicious 0.997 vs normal 0.999 |
| Telemetry pasif | metrics `--telemetry=false` vs `true` | identik 3/3 (kecuali metrik jam dinding) |

Audit watchdog (stage1, 82.639 watch):

| Kategori | Penyebab sebenarnya |
|---|---|
| `forwarded` 76.109, `forwarded_late` 8 | semuanya `j_tx_heard` (tidak ada positif palsu) |
| `unobserved_clean` 4.853 | 4.811 drop grayhole, 42 gagal penerusan node normal (`j_ip_forwarded_not_tx`); **0 terlewat observer** |
| `unobserved_rxfail` 1.669 | 1.469 drop grayhole, 167 tabrakan (`j_tx_rxfail_at_observer`), 30 gagal penerusan, 3 terlambat |

Syarat tahap-2 (bagian 4): `est_corrected` 0.314 vs audit 0.295 (selisih 0.019; sisa bias
naik karena 16 node malicious terkena batas D <= A + U), dan miss observer = 0. Keduanya
lolos.

Trust tahap-1 (`ns3-sim/results/step2/trust_stage1/`): AUC (1 - trust) vs label pada
sensor teramati = 1.000 di 27/27 skenario (`discard`: min 0.943). Pada semua sensor,
median AUC 0.833, karena hanya 71 dari 162 node malicious yang pernah menjadi relay di
tahap-1 dan sisanya tetap bernilai 0.5.

Tabel tahap-2 (`ns3-sim/results/routes_stage2/`, `ml/analyze_stage2_tables.py`):

- Sumber yang lewat relay malicious turun dari 169 (tahap-1) ke 76. Sebagai
  pembanding, langkah-1 dengan trust tersimulasi dari label: 46.
- Dari 76 itu, 53 lewat malicious yang **teramati** (trust rendah). Pada 37 dari 38
  entri next-hop yang memilih malicious teramati, tersedia kandidat dengan trust
  lebih tinggi. Ini hanya soal ketersediaan; fitur lain belum diperhitungkan.
- Uji sensitivitas pada Q-net (ubah HANYA trust kandidat terpilih menjadi 0, fitur
  lain tetap; TLDQN tidak diubah):
  - Secara umum TLDQN peka trust: 1.193 dari 1.479 keputusan (81%) berubah. Dugaan
    awal "bobot trust kecil" **tidak didukung**.
  - Pada 38 entri malicious teramati, hanya 19 yang berubah. Alternatif yang lebih
    tepercaya punya progres lebih baik di 37/38, tetapi link quality
    (prr x stability) lebih baik hanya di 16/38. Jadi pada sekitar 22 entri, kandidat
    malicious punya link quality terbaik, dan kebijakan menukar trust dengan link
    quality. Menurut reward lingkungan, peluang sukses = lq x p_fwd, jadi pertukaran
    ini kemungkinan suboptimal. Ini hipotesis yang perlu ditinjau, belum disimpulkan.
- Kontrafaktual trust dipetakan ke skala 10 observasi: 92 (lebih buruk). Pemetaan ini
  mempersempit rentang trust, jadi tidak bisa memisahkan hipotesis; hasilnya hanya
  menunjukkan pemetaan itu tidak membantu.
- TLDQN tidak diubah di langkah ini.

Run NS-3 tahap-2 (`_stage2`, 27 skenario; `ml/analyze_stage2_runs.py`,
`ml/reports/telemetry_step2_network.json`, `figures/real/step2_stage1_vs_stage2.png`).
Perbandingan berpasangan per skenario, uji tanda dua sisi:

| Metrik | Tahap-2 (trust telemetry) | Tahap-1 (trust 0.5) | Lebih baik / seri / lebih buruk | p |
|---|---|---|---|---|
| PDR (%) | 97.08 | 92.73 | 22 / 1 / 4 | 0.0005 |
| Paket dijatuhkan grayhole | 106.0 | 232.6 | 22 / 1 / 4 | 0.0005 |
| Throughput (kbps) | 22.47 | 21.50 | 23 / 1 / 3 | 9e-05 |
| Latency (ms) | 0.806 | 0.632 | 8 / 0 / 19 | 0.052 |
| Energi (J) | 54.259 | 54.248 | 2 / 0 / 25 | 6e-06 |

- Pembanding langkah-1 (trust tersimulasi dari label; batas atas informasi, bukan
  metode sah): PDR 98.09, drop grayhole 63.5. Tahap-2 masih di bawahnya (PDR lebih
  buruk di 18/21, p = 0.0015). Dugaan penyebabnya adalah 91/162 node malicious yang
  tidak teramati di tahap-1 dan 38 entri bertrust rendah (bagian di atas). Selisih ini
  belum diurai, jadi ini masih hipotesis.
- Run langkah-1 (trust label dan greedy) dibuat dengan binary sebelum telemetry
  (47300cf). Diff kode sejak itu hanya menambah telemetry dan menyimpan nilai balik
  `radioEnergyHelper.Install`. Sifat pasif diuji langsung hanya di 3 skenario
  (`--telemetry=false` vs `true`) dan 3 skenario (binary baru vs lama). Perbandingan
  utama tahap-2 vs tahap-1 memakai binary yang sama.
- Pembanding greedy langkah-1: PDR 92.04; tahap-2 lebih baik 19/26 (p = 0.029).
- Energi sedikit lebih buruk (+0.011 J, 0.02%, 25/27 skenario). Latency +0.17 ms
  **tidak signifikan** pada 0.05 (p = 0.052). Hipotesis: lebih banyak paket sampai lewat
  rute yang sedikit lebih panjang (1.92 vs 1.87 hop; selisih hop tidak diuji),
  alih-alih dijatuhkan grayhole. Bukti energi masih tercampur beban relay (langkah 3).
- Uji tanda memperlakukan 27 skenario (9 ukuran x 3 seed) sebagai independen.
- Rasio penerusan malicious di `_stage2`: estimasi terkoreksi 0.321, normal 0.999;
  ke-27 run valid.

## 6. Catatan ns-3 yang memengaruhi validasi

- **`FlowStats.timesForwarded`** hanya menghitung penerusan paket yang sampai tujuan
  (lihat 3.5).
- **Kuantisasi energi `BasicEnergySource`.** `CalculateRemainingEnergy` menghitung
  `(I * V * duration)` dengan `duration` bertipe `Time`, dan hasil `Time * double`
  adalah `Time` (resolusi 1 ns). Tiap update sumber (periodik 1 s + tiap perubahan
  keadaan PHY) dibulatkan sampai sekitar 1 nJ. Model radio dan pembagian per window
  menghitung dalam double, jadi lebih teliti. Toleransi validator mengikuti
  mekanisme ini (`telemetry_schema.energy_tolerance`): 1 nJ x jumlah update. Jumlah
  update per event PHY (4) adalah margin empiris, bukan turunan dari model radio.
  Contoh pilot: selisih 4.6 uJ, batas sekitar 104 uJ.
- **Counter `delivered` vs FlowMonitor**, dua kasus tepi yang berbeda perilaku:
  (a) penerusan paket yang hilang tetap tersimpan di peta per paket (memori sebanding
  jumlah paket hilang); (b) paket yang sampai setelah FlowMonitor menganggapnya hilang
  (10 s tanpa terlihat) dihitung telemetry tetapi diabaikan FlowMonitor. Kasus (b)
  tidak terlihat di data saat ini (paket terima sama persis di 39 run). Jika
  pemeriksaan eksak gagal, periksa kedua kasus ini dulu.

Uji (b) adalah pemeriksaan teks atas `namespace telem`. Ia tidak membuktikan tidak
ada aliran data tidak langsung. Bukti perilaku datang dari kontrol (c).
