/*
 * ============================================================================
 *  honeycomb-sdwsn.cc
 *
 *  Reproduksi (rekonstruksi untuk ns-3.48) dari:
 *    J. Kipongo, T. G. Swart, E. Esenogho,
 *    "Artificial Intelligence-Based Intrusion Detection and Prevention in
 *     Edge-Assisted SDWSN With Modified Honeycomb Structure",
 *    IEEE Access, vol. 12, 2024. DOI: 10.1109/ACCESS.2023.3347778
 *
 *  Sistem: HieMulti-Block.
 *
 *  CATATAN REKONSTRUKSI
 *  --------------------
 *  Kode asli penulis (Modified_Honeycomb_Structure.cc, dirilis di figshare)
 *  bergantung pada modul NS-3 kustom "ns3::sdn" dan "ns3::iot" beserta kelas
 *  TLDQN / BiGAN / DTO / Three_Dimensional_Cube / hexagon / SDN. Modul-modul
 *  tersebut TIDAK ikut dirilis, sehingga kode asli tidak dapat dikompilasi.
 *  File ini merekonstruksi kelas-kelas itu secara mandiri (self-contained),
 *  mengikuti pseudocode dan deskripsi paper (Algoritma 1-3, Section III),
 *  serta koordinat topologi honeycomb yang diambil dari kode asli.
 *
 *  Roadmap (persis Fig. 2 paper):
 *    1. Secure Authentication (3D Cube + DNN)  -> awal simulasi
 *    2. Modified Honeycomb Partitioning & Clustering (1/7) -> t=1.3 s
 *    3. RL-based Intelligent Routing (TLDQN)   -> t=1.3 s (sebelum data mengalir)
 *    4. Hybrid IDS: DTO (malicious) + Bi-GAN (intrusion) -> t=8.3 s; IPS -> 15.3 s
 *    5. HieMulti-Blockchain (main chain + side chains)
 *    6. Evaluasi 9 metrik + confusion matrix
 *
 *  INTEGRASI AI (Python -> ns-3)
 *  -----------------------------
 *  Bi-GAN dan TLDQN dilatih/dijalankan SUNGGUHAN di Python. ns-3 membaca:
 *    bigan_decisions_<n>_s<seed>.csv : ground truth (10% node) + prediksi intrusi
 *                                      (ml/export_decisions.py)
 *    tldqn_nexthop_<n>_s<seed>.csv   : tabel NEXT-HOP TLDQN per (node, sink tujuan)
 *                                      (ml/export_routes.py)
 *    ai_timing.csv                   : waktu inferensi (computation overhead)
 *  Routing (integrasi langkah 1): jangkauan radio dibatasi 150 m (Tabel 5);
 *  tabel TLDQN dipasang sebagai host route statis, sehingga paket diteruskan
 *  MULTI-HOP sesuai keputusan TLDQN. Relay malicious berperilaku grayhole
 *  (menjatuhkan paket transit dgn peluang --grayholeDrop, sama dengan asumsi
 *  lingkungan pelatihan TLDQN). Kandidat next-hop TIDAK dihitung ulang di C++
 *  (urutan slot berasal dari Python); C++ hanya memvalidasi tabel.
 *  Batasan: fitur Bi-GAN masih dari sampel NSL-KDD yang DITUGASKAN ke tiap node
 *  (diganti telemetry NS-3 pada langkah 2-4); node yang diisolasi IPS masih
 *  meneruskan paket transit (langkah 5).
 *
 *  Target: ns-3.48 (CMake / ./ns3). Modul: core, network, mobility, wifi,
 *          internet, energy, flow-monitor, netanim, applications.
 * ============================================================================
 */

#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/mobility-module.h"
#include "ns3/wifi-module.h"
#include "ns3/internet-module.h"
#include "ns3/applications-module.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/netanim-module.h"
#include "ns3/energy-module.h"

#include <vector>
#include <array>
#include <map>
#include <string>
#include <sstream>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <random>
#include <algorithm>
#include <functional>
#include <chrono>
#include <memory>
#include <tuple>
#include <set>

using namespace ns3;
using std::string;
using std::vector;

NS_LOG_COMPONENT_DEFINE("HoneycombSdwsn");

// ============================================================================
//  PARAMETER GLOBAL (Tabel 4 & 5 paper, + dari kode asli penulis)
// ============================================================================
namespace cfg {
  uint16_t numNodes   = 100;   // sensor nodes
  uint16_t numUsers   = 10;    // IoT users
  uint16_t numSwitch  = 4;     // switches
  uint16_t numSink    = 4;     // edge-assisted sinks
  uint16_t numBlock   = 3;     // blockchain nodes
  uint16_t numMaster  = 1;     // primary controller
  uint16_t numSlave   = 2;     // secondary controllers
  uint16_t numCloud   = 1;     // cloud node
  uint16_t numTA      = 1;     // trusted authority

  uint32_t packetSize = 1024;  // byte (variatif 64..1024 di paper)
  uint32_t numPackets = 50;    // paket per node. Tabel 5: 1000, TIDAK muat di memori
                               // (~37.5 MB per paket-per-node pada 100 node; lihat
                               // docs/PAPER_DEVIATIONS.md). Kode rilis penulis: 100.
  int      attackers  = -1;    // target DTO; -1 => 10% dari numNodes (paper).
                               // Ground truth malicious dibaca dari CSV Python.
  double   initEnergy = 100.0; // Joule (Tabel 5)
  double   txPower    = 0.005; // Watt  (Tabel 5)
  double   txRange    = 150.0; // meter (Tabel 5)
  double   simTime    = 600.0; // detik (Tabel 5). Profil 'code': 300.
  double   interval   = 1.0;   // antar paket (detik)
  string   dataRate   = "250Kbps";
  uint32_t seed       = 1;     // untuk RngSeedManager
  string   aiDir      = "results"; // folder CSV keputusan AI (hasil Python)
  double   idleCurrentA    = 0.0005; // arus idle radio sensor (A)
  double   rxCurrentA      = 0.0188; // arus RX radio sensor (A)
  double   ccaBusyCurrentA = 0.0005; // arus CCA-busy radio sensor (A)

  // --- Profil simulasi ---------------------------------------------------
  //  "paper": mengikuti Tabel 4-5 paper (default).
  //  "code" : setara kode rilis penulis (802.11a 6 Mbps, 300 s, tanpa TCP).
  string   profile    = "paper";
  string   wifiStd    = "ac";    // "ac" (80 MHz, VhtMcs2) atau "a" (802.11a)
  bool     varPktSize = true;    // ukuran paket 64,128,...,1024 byte (Tabel 5)
  bool     tcpUsers   = false;   // TCP user -> cloud (Tabel 5: TCP & UDP). MATI default:
                               // memori membengkak (4.6 GB pada 20 node). --tcpUsers=true
  string   userRate   = "500Kbps"; // laju aplikasi tiap user (TCP)
  double   windowSec  = 30.0;    // lebar window metrik time-based
  bool     enableAnim = false;   // NetAnim (berat; aktifkan utk demo saja)
  string   tsCsv      = "";      // "" => results/timeseries_<n>_s<seed>.csv

  // --- Routing TLDQN multi-hop (integrasi langkah 1) -------------------------
  string   routeDir     = "";    // "" => aiDir; berisi tldqn_nexthop_<n>_s<seed>.csv
  double   maxRange     = 150.0; // m, Tabel 5 (RangePropagationLossModel); 0 = tanpa batas
  double   grayholeDrop = 0.70;  // peluang relay malicious menjatuhkan paket transit
                                 // (= 1 - EnvConfig.p_fwd_malicious di Python)

  // --- Telemetry NS-3 (langkah 2; lihat docs/TELEMETRY.md) -------------------
  bool     telemetry    = true;      // tulis telemetry_*.csv (observasi pasif, tidak
                                     // mengubah perilaku jaringan)
  string   telemDir     = "results"; // folder keluaran telemetry
  double   telemWindow  = 30.0;      // lebar window telemetry (s), mulai t = 0
  double   wdTimeout    = 0.1;       // watchdog: penerusan terdengar <= tau -> forwarded
  double   wdLate       = 1.0;       // watchdog: tau < delay <= wdLate -> forwarded_late;
                                     // tidak terdengar s.d. wdLate -> unobserved_*
}

// ============================================================================
//  TAHAP 1 - SECURE AUTHENTICATION: kelas ThreeDimensionalCube (3D Cube + DNN)
//
//  Paper Section III-A: TA mem-registrasi node (ID, PUF, lokasi, random),
//  lalu meng-generate private key via algoritma 3D cube yang "diselesaikan"
//  oleh DNN (deep cube) dalam <= 30 langkah, hasilnya di-hash (SHA-256 style)
//  dan disimpan di blockchain. Persamaan (1)-(12) MCTS cube-solving.
//
//  Rekonstruksi: kita modelkan cube-solve sebagai pengacakan terkontrol
//  (<=30 move) dengan XOR per-move (sesuai deskripsi "XOR on direct value for
//  every move") lalu di-hash. Deterministik per-ID sehingga reproducible.
// ============================================================================
class ThreeDimensionalCube
{
public:
  ThreeDimensionalCube() : m_maxMoves(30) {}

  // Generate private key untuk sebuah entitas (seed = ID+PUF-like).
  // Mengembalikan string hex yang merepresentasikan kunci privat ter-hash.
  string PrivateKey(uint32_t entityId)
  {
    // State awal cube: 54 facelet (6 sisi x 9). Diisi dari entityId.
    std::array<uint32_t, 54> cube;
    uint32_t s = entityId * 2654435761u + 0x9E3779B9u; // seed deterministik
    for (auto &c : cube) { s ^= s << 13; s ^= s >> 17; s ^= s << 5; c = s & 0xFF; }

    // "Deep cube" solving: hingga 30 move, XOR per-move pada nilai langsung.
    uint64_t acc = 0xCBF29CE484222325ull; // FNV offset basis (hash akumulatif)
    uint32_t moves = 1 + (entityId % m_maxMoves); // <= 30 move (Section III-A)
    for (uint32_t m = 0; m < moves; ++m)
    {
      uint32_t layer = (s + m) % 54;
      cube[layer] = (cube[layer] + m * 7 + 1) & 0xFF;   // rotasi layer (move)
      acc ^= cube[layer];                                // XOR direct value
      acc *= 0x100000001B3ull;                           // FNV prime (hash)
      s ^= s << 13; s ^= s >> 17; s ^= s << 5;
    }

    // Hashing final (representasi 256-bit disederhanakan ke 64-bit hex).
    std::ostringstream oss;
    oss << std::hex << std::setw(16) << std::setfill('0') << acc;
    return oss.str();
  }

private:
  uint32_t m_maxMoves;
};

// ============================================================================
//  TAHAP 2 - MODIFIED HONEYCOMB PARTITIONING & CLUSTERING: kelas Hexagon
//
//  Paper Section III-B: node ditempatkan pada edge+center hexagon; clustering
//  1/7 (7 sel per cluster, pers. 13-15); CH dipilih via bobot pers. (16):
//    WCH = w1*centrality + w2*distance + w3*energy + w4*trust,  w_i = 0.25.
// ============================================================================
class Hexagon
{
public:
  // Pers. (14): jumlah cluster. Pers. (15): sensor per cluster = c * 7.
  void Clustering(int sensorsPerInner /*=7*/)
  {
    m_sensorsPerCluster = sensorsPerInner * 7; // Sc = c * 7  (pers. 15)
    NS_LOG_UNCOND("  [Honeycomb] Clustering 1/7: "
                  << "7 sel/cluster, Sc = " << m_sensorsPerCluster
                  << " sensor/cluster (pers. 15).");
  }

  // Pers. (16): CH selection berbobot. Bobot sama (0.25) sesuai paper
  // (Section IV-B: w1=w2=w3=w4=0.25, sum=1).
  double ChWeight(double centrality, double distance,
                  double energy, double trust)
  {
    const double w1 = 0.25, w2 = 0.25, w3 = 0.25, w4 = 0.25;
    return w1 * centrality + w2 * distance + w3 * energy + w4 * trust;
  }

  // Meniru 'flow_features' kode asli: cetak fitur cluster-head terpilih.
  void FlowFeatures(int centrality, int distance, int energy, int trust,
                    int network, int cells, int sensors)
  {
    double w = ChWeight(centrality, distance, energy, trust);
    NS_LOG_UNCOND("  [Honeycomb] CH feature => centrality=" << centrality
                  << " distance=" << distance << " energy=" << energy
                  << " trust=" << trust << " | WCH=" << w
                  << " | network=" << network << " cells=" << cells
                  << " sensors=" << sensors);
  }

  int SensorsPerCluster() const { return m_sensorsPerCluster; }

private:
  int m_sensorsPerCluster {49};
};

// ============================================================================
//  Helper: pecah satu baris CSV sederhana (tanpa quoting).
// ============================================================================
static vector<string> SplitCsv(const string &line)
{
  vector<string> out;
  std::stringstream ss(line);
  string tok;
  while (std::getline(ss, tok, ',')) out.push_back(tok);
  return out;
}

// ============================================================================
//  TAHAP 3 - RL-BASED INTELLIGENT ROUTING: kelas Tldqn
//
//  Paper Section III-C + Algoritma 1. Di paper, route agent (edge server)
//  menjalankan TLDQN. Pelatihan DQN + transfer learning dilakukan SUNGGUHAN
//  di Python (ml/export_decisions.py -> ml/tldqn_routing.py). Keputusan
//  routing (sink/forwarder dgn Q tertinggi per node) ditulis ke CSV dan
//  dibaca di sini, lalu DIPAKAI sebagai tujuan aliran data tiap sensor,
//  sehingga routing berdampak nyata pada PDR / latency / energi.
// ============================================================================
class Tldqn
{
public:
  struct Entry
  {
    int    nextHop;   // sensor id, atau indeks sink bila toSink
    bool   toSink;
    int    slot;      // aksi TLDQN (0..K-1)
    int    nCand;
    bool   ownDest;
    double q;
  };

  // Muat tabel next-hop hasil ml/export_routes.py:
  //   node_id,dest_sink,next_hop,next_hop_is_sink,slot,n_candidates,own_dest,q_value,
  //   node_malicious
  void Load(const string &path, int nSensors, int nSinks)
  {
    std::ifstream in(path);
    if (!in)
      NS_FATAL_ERROR("Tabel next-hop TLDQN tidak ditemukan: " << path
                     << " (jalankan ml/export_routes.py lalu salin CSV ke folder ini)");
    string line;
    std::getline(in, line);
    if (line.rfind("node_id,dest_sink,next_hop,next_hop_is_sink,slot", 0) != 0)
      NS_FATAL_ERROR("Header tabel next-hop tidak dikenal: " << line);
    while (std::getline(in, line))
    {
      if (line.empty()) continue;
      vector<string> f = SplitCsv(line);
      if (f.size() < 9) NS_FATAL_ERROR("Baris tabel next-hop rusak (butuh 9 kolom): " << line);
      int c = std::stoi(f[0]), d = std::stoi(f[1]);
      Entry e{std::stoi(f[2]), std::stoi(f[3]) != 0, std::stoi(f[4]), std::stoi(f[5]),
              std::stoi(f[6]) != 0, std::stod(f[7])};
      if (c < 0 || c >= nSensors || d < 0 || d >= nSinks)
        NS_FATAL_ERROR("Indeks di luar rentang pada tabel next-hop: " << line);
      if (e.toSink ? (e.nextHop != d) : (e.nextHop < 0 || e.nextHop >= nSensors))
        NS_FATAL_ERROR("Next-hop tidak valid pada tabel: " << line);
      m_mal[c] = std::stoi(f[8]) != 0;
      if (!m_table.emplace(std::make_pair(c, d), e).second)
        NS_FATAL_ERROR("Duplikat (node, sink) pada tabel next-hop: " << line);
      if (e.ownDest)
      {
        if (m_own.count(c)) NS_FATAL_ERROR("Sensor " << c << " punya >1 own_dest");
        m_own[c] = d;
      }
    }
    if ((int)m_own.size() != nSensors)
      NS_FATAL_ERROR("Tabel next-hop: " << m_own.size() << " sensor punya tujuan, harus "
                     << nSensors);
    // Setiap sensor harus mencapai sink tujuannya, bebas loop.
    for (int s = 0; s < nSensors; ++s)
    {
      int cur = s, d = m_own[s], hops = 0;
      std::vector<bool> seen(nSensors, false);
      while (true)
      {
        auto it = m_table.find({cur, d});
        if (it == m_table.end())
          NS_FATAL_ERROR("Rute dari sensor " << s << " terputus di sensor " << cur);
        ++hops;
        if (it->second.toSink) break;
        seen[cur] = true;
        cur = it->second.nextHop;
        if (seen[cur]) NS_FATAL_ERROR("Loop pada tabel next-hop dari sensor " << s);
      }
      m_hops[s] = hops;
    }
    NS_LOG_UNCOND("  [TLDQN] " << m_table.size() << " entri next-hop dimuat dari " << path
                  << " (" << nSensors << " sensor, rute tervalidasi bebas loop).");
  }

  int OwnSink(int node) const { return m_own.at(node); }
  // Ground truth malicious yang dipakai saat ekspor tabel (untuk cek konsistensi).
  bool ExportMalicious(int node) const { return m_mal.at(node); }
  int PlannedHops(int node) const { return m_hops.at(node); }
  size_t Size() const { return m_table.size(); }
  const std::map<std::pair<int, int>, Entry> &Table() const { return m_table; }

private:
  std::map<std::pair<int, int>, Entry> m_table;
  std::map<int, int> m_own;
  std::map<int, int> m_hops;
  std::map<int, bool> m_mal;
};

// ============================================================================
//  Relay malicious: GRAYHOLE pada lapisan IP.
//  Dipasang di Ipv4ListRouting node malicious dgn prioritas tertinggi.
//  Ipv4ListRouting sudah menyerahkan paket untuk node ini sendiri (local
//  delivery) sebelum memanggil protokol, jadi yang sampai di sini hanya paket
//  TRANSIT. Paket transit dijatuhkan dgn peluang pDrop (lewat ErrorCallback,
//  sehingga FlowMonitor mencatatnya sebagai drop); sisanya diteruskan oleh
//  static routing (return false).
// ============================================================================
static uint64_t g_grayholeDrops = 0, g_grayholePassed = 0;
static std::map<Ipv4Address, int> g_ipToSensor;   // IP sensor -> indeks sensor

class GrayholeRouting : public Ipv4RoutingProtocol
{
public:
  static TypeId GetTypeId()
  {
    static TypeId tid = TypeId("ns3::HoneycombGrayholeRouting")
                            .SetParent<Ipv4RoutingProtocol>()
                            .SetGroupName("Internet")
                            .AddConstructor<GrayholeRouting>();
    return tid;
  }

  void Configure(double pDrop, int64_t stream)
  {
    m_pDrop = pDrop;
    m_rv = CreateObject<UniformRandomVariable>();
    m_rv->SetStream(stream);
  }

  Ptr<Ipv4Route> RouteOutput(Ptr<Packet>, const Ipv4Header &, Ptr<NetDevice>,
                             Socket::SocketErrno &sockerr) override
  {
    sockerr = Socket::ERROR_NOROUTETOHOST;   // paket lokal: serahkan ke static routing
    return nullptr;
  }

  bool RouteInput(Ptr<const Packet> p, const Ipv4Header &header, Ptr<const NetDevice>,
                  const UnicastForwardCallback &, const MulticastForwardCallback &,
                  const LocalDeliverCallback &, const ErrorCallback &ecb) override
  {
    const Ipv4Address dst = header.GetDestination();
    if (dst.IsBroadcast() || dst.IsMulticast() || !m_rv) return false;
    if (m_rv->GetValue() < m_pDrop)
    {
      ++g_grayholeDrops;
      if (!ecb.IsNull()) ecb(p, header, Socket::ERROR_NOROUTETOHOST);
      return true;                           // paket diambil alih (dijatuhkan)
    }
    ++g_grayholePassed;
    return false;                            // diteruskan oleh static routing
  }

  void NotifyInterfaceUp(uint32_t) override {}
  void NotifyInterfaceDown(uint32_t) override {}
  void NotifyAddAddress(uint32_t, Ipv4InterfaceAddress) override {}
  void NotifyRemoveAddress(uint32_t, Ipv4InterfaceAddress) override {}
  void SetIpv4(Ptr<Ipv4> ipv4) override { m_ipv4 = ipv4; }
  void PrintRoutingTable(Ptr<OutputStreamWrapper> stream, Time::Unit) const override
  {
    *stream->GetStream() << "GrayholeRouting pDrop=" << m_pDrop << "\n";
  }

private:
  Ptr<Ipv4> m_ipv4;
  Ptr<UniformRandomVariable> m_rv;
  double m_pDrop {0.0};
};
NS_OBJECT_ENSURE_REGISTERED(GrayholeRouting);

// ============================================================================
//  TAHAP 4a - MALICIOUS NODE DETECTION: kelas Dto (Driver Training Optimization)
//
//  Paper Section III-D-1 + Algoritma 2 + pers. (21)-(31). Secondary controller
//  (investigate agent) memantau perilaku node: konsumsi energi, PDR, packet
//  loss. DTO mencari solusi terbaik = node malicious yang diisolasi.
//
//  Implementasi: tiap node punya variabel keputusan z_i (estimasi tingkat
//  kecurigaan, pers. 21-22). Fungsi objektif OF_i = (z_i - bukti_i)^2 dgn
//  bukti_i DIUKUR dari simulasi (laju kirim paket & energi terpakai, bukan
//  label). Tiga fase DTO dgn penerimaan greedy (pers. 25-31) meminimalkan OF.
//  Node dgn z tertinggi dinyatakan malicious. Catatan: fase-3 memakai langkah
//  aditif (bukan perkalian pers. 30) agar konvergen untuk z_i dekat nol.
// ============================================================================
class Dto
{
public:
  Dto(int maxIter = 100) : m_maxIter(maxIter)
  {
    m_rng.seed(cfg::seed + 11);
  }

  // evidence[i] in [0,1]: makin tinggi makin mencurigakan.
  vector<int> DetectMalicious(const vector<double> &evidence, int numAttackers)
  {
    const int n = (int)evidence.size();
    std::uniform_real_distribution<double> U(0.0, 1.0);
    vector<double> z(n);
    for (int i = 0; i < n; ++i) z[i] = U(m_rng);              // pers. (22)

    auto OF = [&](int i, double zi) {
      double d = zi - evidence[i];
      return d * d;
    };

    for (int q = 1; q <= m_maxIter; ++q)
    {
      double decay = 1.0 - double(q) / m_maxIter;             // (1 - q/Y)
      int nID = std::max(1, (int)std::ceil(0.1 * n * decay)); // pers. (24)
      double P = 0.01 + 0.9 * decay;                          // pers. (29)

      // Instruktur = anggota populasi dgn OF terbaik (terkecil).
      vector<int> idx(n);
      for (int i = 0; i < n; ++i) idx[i] = i;
      std::sort(idx.begin(), idx.end(), [&](int a, int b) {
        return OF(a, z[a]) < OF(b, z[b]);
      });

      for (int i = 0; i < n; ++i)
      {
        int k = idx[int(U(m_rng) * nID) % nID];
        double cand;

        // Fase 1: driving instructor training (25)-(26)
        if (OF(k, z[k]) < OF(i, z[i]))
          cand = z[i] + U(m_rng) * (z[k] - (1.0 + U(m_rng)) * z[i]);
        else
          cand = z[i] + U(m_rng) * (z[i] - z[k]);
        if (OF(i, cand) < OF(i, z[i])) z[i] = cand;

        // Fase 2: learner driver patterning (27)-(28)
        cand = P * z[i] + (1.0 - P) * z[k];
        if (OF(i, cand) < OF(i, z[i])) z[i] = cand;

        // Fase 3: learner driver practice (30)-(31), pencarian lokal
        cand = z[i] + (1.0 - 2.0 * U(m_rng)) * decay * 0.5;
        if (OF(i, cand) < OF(i, z[i])) z[i] = cand;
      }
    }

    // Solusi terbaik: numAttackers node dgn z tertinggi = malicious.
    vector<int> order(n);
    for (int i = 0; i < n; ++i) order[i] = i;
    std::sort(order.begin(), order.end(),
              [&](int a, int b) { return z[a] > z[b]; });
    vector<int> mal(order.begin(), order.begin() + std::min(numAttackers, n));
    std::sort(mal.begin(), mal.end());
    return mal;
  }

private:
  int m_maxIter;
  std::mt19937 m_rng;
};

// ============================================================================
//  TAHAP 4b - INTRUSION DETECTION: kelas BiGan (Bidirectional GAN)
//
//  Paper Section III-D-2 + Algoritma 3 + pers. (32)-(37). Di paper, Bi-GAN
//  berjalan di secondary controller dan mengklasifikasi normal/malicious.
//  Model Bi-GAN (G/E/D) dilatih & dijalankan SUNGGUHAN di Python
//  (ml/export_decisions.py -> ml/bigan_ids.py, dataset NSL-KDD). Keputusannya
//  per node dibaca di sini dari CSV:
//     node_id,actual_mal,anomaly_score,threshold,pred_intrusion
//  actual_mal = ground truth SKENARIO (10% node), bukan hasil DTO, sehingga
//  confusion matrix tidak circular.
// ============================================================================
class BiGan
{
public:
  struct Decision
  {
    bool   actualMal;
    bool   pred;
    double score;
  };

  void Load(const string &path)
  {
    std::ifstream in(path);
    if (!in)
      NS_FATAL_ERROR("File keputusan Bi-GAN tidak ditemukan: " << path
                     << " (jalankan ml/export_decisions.py lalu salin CSV "
                        "ke folder ini)");
    string line;
    std::getline(in, line); // header
    while (std::getline(in, line))
    {
      if (line.empty()) continue;
      vector<string> f = SplitCsv(line);
      if (f.size() < 5) continue;
      Decision d;
      d.actualMal = std::stoi(f[1]) != 0;
      d.score     = std::stod(f[2]);
      d.pred      = std::stoi(f[4]) != 0;
      m_dec[std::stoi(f[0])] = d;
    }
    NS_LOG_UNCOND("  [Bi-GAN] " << m_dec.size() << " keputusan dimuat dari "
                  << path << " (model PyTorch, NSL-KDD).");
  }

  // Prediksi per window utk metrik time-based: node_id,actual_mal,p0..p{W-1}
  void LoadWindows(const string &path)
  {
    std::ifstream in(path);
    if (!in)
      NS_FATAL_ERROR("File window Bi-GAN tidak ditemukan: " << path
                     << " (jalankan ml/export_decisions.py)");
    string line;
    std::getline(in, line); // header
    while (std::getline(in, line))
    {
      if (line.empty()) continue;
      vector<string> f = SplitCsv(line);
      if (f.size() < 3) continue;
      vector<uint8_t> p;
      for (size_t j = 2; j < f.size(); ++j)
        p.push_back((uint8_t)(std::stoi(f[j]) != 0));
      m_win[std::stoi(f[0])] = p;
    }
    NS_LOG_UNCOND("  [Bi-GAN] prediksi " << Windows() << " window dimuat dari "
                  << path);
  }
  int Windows() const
  {
    return m_win.empty() ? 0 : (int)m_win.begin()->second.size();
  }
  bool WindowPred(int k, int nodeId) const
  {
    return m_win.at(nodeId).at((size_t)k) != 0;
  }

  const Decision &Get(int nodeId) const { return m_dec.at(nodeId); }
  size_t Size() const { return m_dec.size(); }

  int CountActualMalicious() const
  {
    int c = 0;
    for (const auto &kv : m_dec) c += kv.second.actualMal ? 1 : 0;
    return c;
  }

private:
  std::map<int, Decision> m_dec;
  std::map<int, vector<uint8_t>> m_win;
};

// ============================================================================
//  TAHAP 5 - HIEMULTI-BLOCKCHAIN: kelas HieMultiBlock
//
//  Paper Section III-E + pers. (38)-(43). Main chain (coordinator) + side
//  chains (sharding) untuk skalabilitas & energi rendah. Transaksi disimpan
//  dalam format hash (pers. 39).
// ============================================================================
class HieMultiBlock
{
public:
  HieMultiBlock(int numShards = 3) : m_numShards(numShards) {}

  // Simpan transaksi (mis. private key node) ke side chain lalu commit ke main.
  void StoreTransaction(const string &tx)
  {
    uint64_t h = Hash(tx, m_nonce++);
    int shard = h % m_numShards;              // pilih shard (sharding)
    m_sideChains[shard].push_back(h);         // append ke side chain
    m_mainChain.push_back(h);                 // ringkasan ke main chain (pers.38)
  }

  void Report()
  {
    NS_LOG_UNCOND("  [HieMulti-Block] main chain len=" << m_mainChain.size()
                  << ", shards=" << m_numShards);
    for (int s = 0; s < m_numShards; ++s)
      NS_LOG_UNCOND("     side chain[" << s << "] len="
                    << m_sideChains[s].size());
  }

private:
  // Pers. (39): hash[n] = H(tr, Nc).
  static uint64_t Hash(const string &tr, uint64_t nonce)
  {
    uint64_t h = 0xCBF29CE484222325ull;
    for (char c : tr) { h ^= uint8_t(c); h *= 0x100000001B3ull; }
    h ^= nonce; h *= 0x100000001B3ull;
    return h;
  }

  int m_numShards;
  uint64_t m_nonce {0};
  vector<uint64_t> m_mainChain;
  std::map<int, vector<uint64_t>> m_sideChains;
};

// ============================================================================
//  KOORDINAT HONEYCOMB (diambil PERSIS dari kode asli penulis)
//  Section III-B, Fig. 4: 100 sensor pada edge+center hexagon.
// ============================================================================
static const double g_sensorPos[100][2] = {
  {600,600},{550,550},{550,500},{600,450},{650,550},{650,500},{500,600},
  {100,600},{450,500},{500,450},{400,600},{350,550},{350,500},{400,450},
  {300,600},{250,550},{250,500},{300,450},{200,600},{150,550},{150,500},
  {200,450},{700,600},{700,450},{750,550},{750,500},{800,600},{800,450},
  {850,550},{850,500},{150,650},{150,700},{250,650},{250,700},{350,650},
  {350,700},{450,650},{450,700},{550,650},{550,700},{650,650},{650,700},
  {750,650},{750,700},{850,650},{850,700},{200,750},{600,750},{500,750},
  {400,750},{300,750},{700,750},{800,750},{150,400},{150,350},{100,450},
  {250,350},{450,400},{450,350},{550,400},{550,350},{900,300},{650,350},
  {750,400},{750,350},{850,400},{850,350},{350,400},{350,350},{200,300},
  {600,300},{500,300},{400,300},{300,300},{700,300},{800,300},{150,200},
  {150,250},{250,200},{250,250},{350,200},{350,250},{450,200},{100,300},
  {550,200},{550,250},{650,200},{650,250},{750,200},{750,250},{850,200},
  {850,250},{200,150},{600,150},{500,150},{400,150},{300,150},{700,150},
  {800,150},{900,450}
};

static const double g_sinkPos[4][2] = {
  {250,450},{450,250},{450,700},{700,450}
};

// Objek global tahap-tahap (persis urutan pemanggilan kode asli).
static Hexagon              g_hex;
static Tldqn                g_tldqn;
static Dto                  g_dto;
static BiGan                g_bigan;
static ThreeDimensionalCube g_cube;
static HieMultiBlock        g_chain;
static vector<int>          g_malicious;  // node yang dinyatakan malicious oleh DTO
static vector<bool>         g_blocked;    // node yang diisolasi (IPS)
static vector<uint32_t>     g_txCount;    // paket terkirim per sensor (bukti DTO)
static vector<double>       g_remain;     // sisa energi per sensor saat IDS
static vector<Ptr<Socket>>  g_sinkSockets;

// Penghitung pesan kontrol (logis): auth, join cluster, route req/reply,
// laporan IDS, flow-rule isolasi, dan alarm. Dipakai utk control overhead.
static uint64_t g_ctrlMsgs = 0;

// Waktu komputasi (ms) per entitas utk computation overhead:
// 3D Cube diukur di sini; Bi-GAN & TLDQN dibaca dari ai_timing.csv (Python).
static double g_authMsPerEntity = 0.0;
static size_t g_authSink = 0;   // penahan agar loop pengukuran tidak dioptimasi
static double g_aiBiganMs = 0.0;
static double g_aiTldqnMs = 0.0;

// IP sink (diisi di main) dipakai untuk aliran UDP UNICAST yang terukur
// oleh FlowMonitor. Penting: broadcast/IPv6 link-local tidak ter-klasifikasi
// oleh Ipv4FlowClassifier, sehingga metrik L4 menjadi 0.
static Ipv4InterfaceContainer g_sinkIf;

// Confusion matrix Bi-GAN (Fig. 19) dan DTO (dilaporkan terpisah).
static uint32_t g_tp = 0, g_tn = 0, g_fp = 0, g_fn = 0;
static uint32_t g_dtoTp = 0, g_dtoTn = 0, g_dtoFp = 0, g_dtoFn = 0;
static uint32_t g_blockedCount = 0, g_blockedAttackers = 0;

// Node malicious mengirim FLOODING: laju kirim dikali faktor ini sampai
// diisolasi oleh IPS. Ini yang membuat deteksi berdampak ke PDR/latency/energi.
// Faktor flooding TIDAK seragam: tiap penyerang acak dalam [kFloodMin, kFloodMax]
// sehingga sebagian sulit dipisahkan DTO dari node normal (deteksi tidak trivial).
static const double kFloodMin = 1.5;
static const double kFloodMax = 5.0;

// Hitungan confusion per window (TP,TN,FP,FN) utk metrik time-based.
static vector<std::array<uint32_t, 4>> g_win;

// ----------------------------------------------------------------------------
//  Muat waktu komputasi AI (ms/node) hasil Python untuk jumlah node n.
// ----------------------------------------------------------------------------
static void LoadAiTiming(const string &path, int n,
                         double &biganMs, double &tldqnMs)
{
  biganMs = tldqnMs = 0.0;
  std::ifstream in(path);
  if (!in)
  {
    NS_LOG_UNCOND("  [warn] " << path << " tidak ada; waktu komputasi AI = 0.");
    return;
  }
  string line;
  std::getline(in, line); // header
  while (std::getline(in, line))
  {
    vector<string> f = SplitCsv(line);
    if (f.size() < 3) continue;
    if (std::stoi(f[0]) == n)
    {
      biganMs = std::stod(f[1]);
      tldqnMs = std::stod(f[2]);
    }
  }
}

// ----------------------------------------------------------------------------
//  Traffic helpers (API ns-3.48)
// ----------------------------------------------------------------------------
static void ReceivePacket(Ptr<Socket> socket)
{
  while (socket->Recv()) { /* paket diterima */ }
}

// nodeIdx >= 0: indeks sensor (utk penghitung & status isolasi); -1: lainnya.
static void GenerateTraffic(Ptr<Socket> socket, uint32_t pktSize,
                            uint32_t pktCount, Time pktInterval, int nodeIdx)
{
  const bool isSensor = nodeIdx >= 0 && nodeIdx < (int)g_blocked.size();
  if (isSensor && g_blocked[nodeIdx])
  {
    socket->Close(); // node diisolasi: berhenti mengirim (intrusion prevention)
    return;
  }
  if (pktCount > 0)
  {
    socket->Send(Create<Packet>(pktSize));
    if (isSensor) ++g_txCount[nodeIdx];
    Simulator::Schedule(pktInterval, &GenerateTraffic, socket,
                        pktSize, pktCount - 1, pktInterval, nodeIdx);
  }
  else
  {
    socket->Close();
  }
}

// Receiver UDP port 80 di tiap edge-sink (dibuat SEKALI).
static void SetupSinkReceivers()
{
  TypeId tid = TypeId::LookupByName("ns3::UdpSocketFactory");
  for (uint32_t s = 0; s < g_sinkIf.GetN(); ++s)
  {
    Ptr<Node> sinkNode = g_sinkIf.Get(s).first->GetObject<Node>();
    Ptr<Socket> r = Socket::CreateSocket(sinkNode, tid);
    r->Bind(InetSocketAddress(Ipv4Address::GetAny(), 80));
    r->SetRecvCallback(MakeCallback(&ReceivePacket));
    g_sinkSockets.push_back(r);
  }
}

// Buat satu aliran UDP UNICAST dari node sumber ke sebuah edge-sink.
// Unicast ke IP sink agar FlowMonitor mencatat tx/rx/delay (metrik L4).
//   rateFactor : pengali laju kirim (flooding penyerang, acak dalam [kFloodMin,kFloodMax])
//   pktCount   : 0 => cfg::numPackets
//   pktSize    : 0 => cfg::packetSize
static void SetupFlow(Ptr<Node> srcNode, uint32_t sinkIdx, int nodeIdx = -1,
                      double rateFactor = 1.0, uint32_t pktCount = 0,
                      uint32_t pktSize = 0)
{
  TypeId tid = TypeId::LookupByName("ns3::UdpSocketFactory");
  if (g_sinkIf.GetN() == 0) return;
  sinkIdx = sinkIdx % g_sinkIf.GetN();
  if (pktCount == 0) pktCount = cfg::numPackets;
  if (pktSize == 0)
  {
    // Tabel 5: ukuran paket 64,128,...,1024 byte (dipilih acak per aliran).
    static Ptr<UniformRandomVariable> sizeRv = CreateObject<UniformRandomVariable>();
    pktSize = cfg::varPktSize ? 64u * sizeRv->GetInteger(1, 16)
                              : cfg::packetSize;
  }

  // Sender: unicast ke IP sink.
  Ipv4Address sinkAddr = g_sinkIf.GetAddress(sinkIdx);
  Ptr<Socket> source = Socket::CreateSocket(srcNode, tid);
  InetSocketAddress remote(sinkAddr, 80);
  source->Connect(remote);
  // Jitter start per-node (0..1 s) agar node tidak mengirim serentak.
  static Ptr<UniformRandomVariable> jitter = CreateObject<UniformRandomVariable>();
  double start = 0.1 + jitter->GetValue(0.0, 1.0);

  // Beban trafik DISKALAKAN dengan jumlah node: makin banyak node, makin rapat
  // interval antar paket -> contention di medium adhoc bersama meningkat ->
  // latency naik & PDR sedikit turun seiring node (tren seperti paper).
  double scaledInterval = cfg::interval * (20.0 / std::max<double>(cfg::numNodes, 1));
  if (scaledInterval < 0.01) scaledInterval = 0.01;
  scaledInterval /= std::max(rateFactor, 1e-6);
  if (scaledInterval < 0.002) scaledInterval = 0.002;

  Simulator::ScheduleWithContext(source->GetNode()->GetId(), Seconds(start),
                                 &GenerateTraffic, source, pktSize, pktCount,
                                 Seconds(scaledInterval), nodeIdx);
}

// ----------------------------------------------------------------------------
//  TAHAP 2 + 3: clustering, CH selection, routing TLDQN, lalu mulai transmisi
//  Urutan mengikuti paper (Sec. IV-B): clustering -> routing -> deteksi.
// ----------------------------------------------------------------------------
void PhaseCluster(NodeContainer sensors)
{
  NS_LOG_UNCOND("\n=== TAHAP 2: Modified Honeycomb Partitioning & Clustering "
                "(1/7) ===");
  g_hex.Clustering(7);
  NS_LOG_UNCOND("  CH dipilih via bobot (centrality, distance, energy, trust), "
                "w_i=0.25 (pers. 16).");
  g_hex.FlowFeatures(20, 10, 47, 63, 78, 91, 7);
  g_ctrlMsgs += sensors.GetN();            // pesan join-cluster

  NS_LOG_UNCOND("\n=== TAHAP 3: RL-based Intelligent Routing (TLDQN) ===");
  g_ctrlMsgs += 2ull * sensors.GetN();     // route request + reply
  uint32_t perSink[4] = {0, 0, 0, 0};
  uint32_t flooders = 0;
  std::mt19937 floodRng(cfg::seed + 99);
  std::uniform_real_distribution<double> floodU(0.0, 1.0);
  for (uint32_t i = 0; i < sensors.GetN(); ++i)
  {
    // Tujuan aliran = sink terdekat; jalurnya mengikuti host route TLDQN.
    uint32_t sink = (uint32_t)g_tldqn.OwnSink((int)i);
    ++perSink[sink % 4];
    bool atk = g_bigan.Get((int)i).actualMal;  // perilaku node (skenario)
    double f = 1.0;
    if (atk)
    {
      ++flooders;
      f = kFloodMin + (kFloodMax - kFloodMin) * floodU(floodRng);
    }
    SetupFlow(sensors.Get(i), sink, (int)i, f,
              atk ? (uint32_t)std::ceil(cfg::numPackets * f) : 0);
  }
  double planned = 0.0;
  for (uint32_t i = 0; i < sensors.GetN(); ++i) planned += g_tldqn.PlannedHops((int)i);
  NS_LOG_UNCOND("  TLDQN: rata-rata hop terencana = "
                << planned / std::max<uint32_t>(sensors.GetN(), 1) << " (multi-hop, jangkauan "
                << cfg::maxRange << " m)");
  NS_LOG_UNCOND("  TLDQN: sensor per sink = [" << perSink[0] << ", " << perSink[1]
                << ", " << perSink[2] << ", " << perSink[3] << "]; "
                << flooders << " node malicious mengirim flooding (faktor "
                << kFloodMin << "-" << kFloodMax << ").");
}

// ----------------------------------------------------------------------------
//  TAHAP 4: hybrid IDS (DTO + Bi-GAN) + isolasi node (intrusion prevention)
//  Bukti DTO DIUKUR: laju kirim paket & energi terpakai tiap sensor.
// ----------------------------------------------------------------------------
static vector<double> Norm01(const vector<double> &v)
{
  vector<double> out(v.size(), 0.0);
  if (v.empty()) return out;
  double lo = *std::min_element(v.begin(), v.end());
  double hi = *std::max_element(v.begin(), v.end());
  if (hi - lo < 1e-12) return out;
  for (size_t i = 0; i < v.size(); ++i) out[i] = (v[i] - lo) / (hi - lo);
  return out;
}

template <typename EnergyContainer>
void PhaseIds(NodeContainer sensors, EnergyContainer energySources)
{
  NS_LOG_UNCOND("\n=== TAHAP 4: Hybrid IDS (DTO malicious + Bi-GAN intrusion) "
                "===");
  const int n = (int)sensors.GetN();

  // Investigate agent: kumpulkan perilaku nyata tiap sensor.
  vector<double> consumed(n, 0.0);
  g_remain.assign(n, 0.0);
  int idx = 0;
  for (auto it = energySources.Begin(); it != energySources.End() && idx < n;
       ++it, ++idx)
  {
    g_remain[idx] = (*it)->GetRemainingEnergy();
    consumed[idx] = cfg::initEnergy - g_remain[idx];
  }
  vector<double> tx(g_txCount.begin(), g_txCount.end());
  vector<double> tN = Norm01(tx), eN = Norm01(consumed);
  vector<double> evidence(n);
  for (int i = 0; i < n; ++i) evidence[i] = 0.6 * tN[i] + 0.4 * eN[i];

  // 4a: malicious node detection via DTO (jumlah target = 10% node, paper).
  g_malicious = g_dto.DetectMalicious(evidence, cfg::attackers);
  std::ostringstream ml;
  for (int m : g_malicious) ml << m << " ";
  NS_LOG_UNCOND("  [DTO] node malicious terdeteksi (" << g_malicious.size()
                << "): " << ml.str());

  // 4b: Bi-GAN (keputusan Python) + hitung confusion matrix vs ground truth.
  for (int i = 0; i < n; ++i)
  {
    const BiGan::Decision &d = g_bigan.Get(i);
    bool dtoFlag = std::find(g_malicious.begin(), g_malicious.end(), i)
                   != g_malicious.end();

    if (d.pred && d.actualMal)        ++g_tp;
    else if (!d.pred && !d.actualMal) ++g_tn;
    else if (d.pred && !d.actualMal)  ++g_fp;
    else                              ++g_fn;

    if (dtoFlag && d.actualMal)        ++g_dtoTp;
    else if (!dtoFlag && !d.actualMal) ++g_dtoTn;
    else if (dtoFlag && !d.actualMal)  ++g_dtoFp;
    else                               ++g_dtoFn;

    ++g_ctrlMsgs;                          // laporan node -> secondary controller
    if (d.pred || dtoFlag)                 // IPS: isolasi node terdeteksi
    {
      g_blocked[i] = true;
      ++g_blockedCount;
      if (d.actualMal) ++g_blockedAttackers;
      ++g_ctrlMsgs;                        // flow-rule drop dipasang di switch
    }
  }
  g_win.push_back({g_tp, g_tn, g_fp, g_fn});   // window 0 (t=8.3 s)
  NS_LOG_UNCOND("  [Bi-GAN] confusion => TP=" << g_tp << " TN=" << g_tn
                << " FP=" << g_fp << " FN=" << g_fn);
  NS_LOG_UNCOND("  [DTO]    confusion => TP=" << g_dtoTp << " TN=" << g_dtoTn
                << " FP=" << g_dtoFp << " FN=" << g_dtoFn);
  NS_LOG_UNCOND("  [IPS] node diisolasi = " << g_blockedCount
                << " (penyerang terisolasi " << g_blockedAttackers << ")");
}

// ----------------------------------------------------------------------------
//  Monitoring IDS periodik (metrik time-based, Fig. 18-19): tiap window, Bi-GAN
//  mengklasifikasi ulang perilaku/trafik SEMUA node dengan sampel baru.
//  Isolasi (IPS) hanya diputuskan pada deteksi awal (t=8.3 s); window berikutnya
//  hanya mengukur kualitas deteksi sepanjang simulasi.
// ----------------------------------------------------------------------------
void PhaseMonitor(int k)
{
  const int n = (int)cfg::numNodes;
  std::array<uint32_t, 4> row = {0, 0, 0, 0};   // TP,TN,FP,FN
  for (int i = 0; i < n; ++i)
  {
    const bool actual = g_bigan.Get(i).actualMal;
    const bool pred = g_bigan.WindowPred(k, i);
    if (pred && actual)        ++row[0];
    else if (!pred && !actual) ++row[1];
    else if (pred && !actual)  ++row[2];
    else                       ++row[3];
  }
  g_tp += row[0]; g_tn += row[1]; g_fp += row[2]; g_fn += row[3];
  g_win.push_back(row);
  g_ctrlMsgs += (uint64_t)n;                    // laporan IDS node -> controller
}

// ----------------------------------------------------------------------------
//  Application plane (Tabel 5: TCP & UDP): tiap IoT user mengirim TCP ke cloud.
//  User bergerak Random Waypoint. Trafik sensor (UDP) = plane data utama.
// ----------------------------------------------------------------------------
static void SetupUserTcp(NodeContainer users, Ptr<Node> cloudNode,
                         Ipv4InterfaceContainer ifCloud)
{
  const uint16_t port = 9000;
  PacketSinkHelper sink("ns3::TcpSocketFactory",
                        InetSocketAddress(Ipv4Address::GetAny(), port));
  ApplicationContainer sinkApp = sink.Install(cloudNode);
  sinkApp.Start(Seconds(0.0));
  sinkApp.Stop(Seconds(cfg::simTime));

  for (uint32_t u = 0; u < users.GetN(); ++u)
  {
    OnOffHelper onoff("ns3::TcpSocketFactory",
                      InetSocketAddress(ifCloud.GetAddress(0), port));
    onoff.SetAttribute("OnTime",
                       StringValue("ns3::ConstantRandomVariable[Constant=1]"));
    onoff.SetAttribute("OffTime",
                       StringValue("ns3::ConstantRandomVariable[Constant=1]"));
    onoff.SetAttribute("DataRate", DataRateValue(DataRate(cfg::userRate)));
    onoff.SetAttribute("PacketSize", UintegerValue(512));
    ApplicationContainer app = onoff.Install(users.Get(u));
    app.Start(Seconds(1.0 + 0.1 * u));
    app.Stop(Seconds(cfg::simTime));
  }
}

// ----------------------------------------------------------------------------
//  TAHAP 4c: intrusion prevention (optimal delegator)
//  Delegator = sensor TIDAK terisolasi dgn sisa energi terbesar (faktor lain
//  di paper: network active time, trust, throughput; belum dimodelkan).
//  Delegator mengirim paket alarm ke edge-sink tujuannya (lewat rute TLDQN).
//  Catatan: dgn jangkauan 150 m sink lain tidak terjangkau langsung dan tidak ada
//  rute ke sana; penyebaran alarm ke seluruh jaringan (control plane) belum
//  dimodelkan (langkah 5).
// ----------------------------------------------------------------------------
void PhasePrevention(NodeContainer sensors)
{
  NS_LOG_UNCOND("\n=== TAHAP 4c: Intrusion Prevention (optimal delegator) ===");
  int best = -1;
  double bestE = -1.0;
  for (int i = 0; i < (int)sensors.GetN(); ++i)
  {
    if (g_blocked[i]) continue;
    double e = (i < (int)g_remain.size()) ? g_remain[i] : 0.0;
    if (e > bestE) { bestE = e; best = i; }
  }
  if (best < 0)
  {
    NS_LOG_UNCOND("  Tidak ada delegator yang tersedia.");
    return;
  }
  const uint32_t own = (uint32_t)g_tldqn.OwnSink(best);
  NS_LOG_UNCOND("  Delegator optimal = sensor " << best
                << " (sisa energi " << bestE << " J); kirim alarm ke edge-sink " << own << ".");
  SetupFlow(sensors.Get(best), own, best, 1.0, 1, 64);
  g_ctrlMsgs += 1;                                     // alarm ke sink
  g_ctrlMsgs += (sensors.GetN() - g_blockedCount);     // sink -> node sah
}

// ============================================================================
//  TELEMETRY NS-3 (integrasi langkah 2)  --  definisi lengkap: docs/TELEMETRY.md
//
//  Semua nilai berasal dari TRACE ns-3 (IP, MAC, PHY, energi). Modul ini TIDAK
//  membaca label malicious (g_bigan / actualMal) maupun status GrayholeRouting;
//  label hanya dipakai Python saat evaluasi. Modul ini juga TIDAK menjadwalkan
//  event, tidak memakai RNG, dan tidak mengaktifkan mode promiscuous MAC, sehingga
//  perilaku jaringan identik dengan --telemetry=false (diuji di server).
//
//  WATCHDOG (observer i = sensor, tetangga j = sensor):
//    mulai  : MAC i menerima ACK/BlockAck untuk MPDU data ke j (AckedMpdu) dan
//             paket IP itu transit bagi j (tujuan IP != j). Kunci paket =
//             (src, dst, identification, protocol) IPv4 -- tidak berubah antar hop.
//             Handoff yang gagal di MAC i (retry habis) TIDAK memulai watch: itu
//             kehilangan tautan i->j, bukan kesalahan j (dicatat mac_handoff_*).
//    dengar : PHY i men-decode BENAR (RxOutcome, status per MPDU = true) frame data
//             dari j (Addr2 = MAC j) yang membawa kunci yang sama. Sumber ini setara
//             radio mode monitor; mode promiscuous MAC sengaja tidak dipakai.
//    kelas  : delay = t_dengar_pertama - t_ack
//             forwarded          delay <= wdTimeout (termasuk delay < 0: j sudah
//                                meneruskan sebelum ACK diterima i, mis. ACK hilang)
//             forwarded_late     wdTimeout < delay <= wdLate
//             unobserved_rxfail  tak terdengar s.d. wdLate DAN PHY i mencatat >= 1
//                                kegagalan terima (RxOutcome gagal / PhyRxDrop) di
//                                [t_ack, t_ack + wdLate] -> mungkin tabrakan/kanal
//             unobserved_clean   tak terdengar dan tanpa kegagalan terima (timeout
//                                observasi). BUKAN otomatis malicious.
//    Klasifikasi dilakukan setelah simulasi dari stempel waktu (setara watchdog
//    online dengan timer wdLate) agar tidak perlu menjadwalkan event.
//  AUDIT (kebenaran simulator; HANYA evaluasi, berkas terpisah): untuk tiap watch
//    dicatat penyebab sebenarnya: j mengirim & terdengar / j mengirim tapi PHY i
//    gagal (tabrakan) / j mengirim tanpa event PHY di i (kanal) / j mengirim setelah
//    wdLate / j tidak mengirim (drop IP per alasan, diteruskan IP tapi hilang di MAC,
//    tanpa event IP).
// ============================================================================
namespace telem {

struct PktKey
{
  uint32_t src, dst;
  uint16_t id;
  uint8_t  proto;
  bool operator<(const PktKey &o) const
  {
    return std::tie(src, dst, id, proto) < std::tie(o.src, o.dst, o.id, o.proto);
  }
};
struct WKey   // (observer, tetangga, paket)
{
  int obs, nbr;
  PktKey k;
  bool operator<(const WKey &o) const
  {
    return std::tie(obs, nbr, k) < std::tie(o.obs, o.nbr, o.k);
  }
};
struct NKey   // (node, paket)
{
  int node;
  PktKey k;
  bool operator<(const NKey &o) const { return std::tie(node, k) < std::tie(o.node, o.k); }
};

enum PeerType : uint8_t { PEER_SENSOR = 0, PEER_SINK = 1, PEER_OTHER = 2 };
static const char *kPeerName[] = {"sensor", "sink", "other"};
struct Peer { PeerType type; int idx; };

struct NodeWin
{
  uint64_t ipOrigPkts = 0, ipOrigBytes = 0, ipRxPkts = 0, ipFwdPkts = 0, ipFwdBytes = 0,
           ipLocalPkts = 0;
  uint64_t ipDrop[8] = {0, 0, 0, 0, 0, 0, 0, 0};   // indeks = Ipv4L3Protocol::DropReason
  uint64_t macAcked = 0, macNacked = 0, macDropRetry = 0, macDropOther = 0, macAckTimeout = 0;
  uint64_t phyTxMpdu = 0, phyTxBytes = 0, phyRxOkMpdu = 0, phyRxFailMpdu = 0, phyRxDrop = 0;
  double   energyJ = 0.0;
  uint64_t wdAsObserver = 0;
  // sebagai node yang DIAMATI (agregat semua observer), per window t_ack
  uint64_t obsWatched = 0, obsFwd = 0, obsLate = 0, obsRxFail = 0, obsClean = 0,
           obsHeardAfter = 0;
};
struct LinkKey
{
  int win, obs;
  uint8_t ptype;
  int pidx;
  bool operator<(const LinkKey &o) const
  {
    return std::tie(win, obs, ptype, pidx) < std::tie(o.win, o.obs, o.ptype, o.pidx);
  }
};
struct LinkWin
{
  uint64_t macAcked = 0, macNacked = 0, macDropRetry = 0, macDropOther = 0, macAckTimeout = 0;
  uint64_t rxOkFrom = 0;
  double   rssiSum = 0.0, snrDbSum = 0.0;
  uint64_t wdStarted = 0, wdFwd = 0, wdLate = 0, wdRxFail = 0, wdClean = 0, wdHeardAfter = 0;
  uint64_t wdHeardRxFail = 0;         // terdengar DAN ada kegagalan terima di window watch
                                      // -> laju kebetulan b (koreksi bias, docs/TELEMETRY.md)
  double   wdDelaySum = 0.0;          // jumlah delay (s) forwarded + forwarded_late
};
struct Watch { WKey key; double t0; };

// ---- keadaan ----
static bool g_on = false;
static int g_n = 0, g_nWin = 1;
static double g_W = 30.0;
static std::map<Mac48Address, Peer> g_mac;
static vector<uint32_t> g_sensorIp;               // IP (uint32) sensor i
static vector<vector<NodeWin>> g_node;            // [sensor][window]
static std::map<LinkKey, LinkWin> g_link;
static vector<Watch> g_watch;
static std::set<WKey> g_watchSeen;
static std::map<WKey, double> g_heard;            // dengar pertama (frame transit dari j)
static vector<vector<double>> g_failT;            // waktu kegagalan terima PHY per observer
// audit (kebenaran simulator)
static std::map<WKey, double> g_failOnFrame;      // PHY i gagal pada frame j berisi paket
static std::map<NKey, double> g_txFirst;          // j memancarkan paket transit (pertama)
static std::map<NKey, int> g_ipFate;              // 1 = UnicastForward; 10+r = Drop alasan r
static std::map<NKey, int> g_macDrop;             // 1 + WifiMacDropReason
// energi
static vector<double> g_eLastT, g_eLastV;
// total jaringan (semua node) utk validasi vs FlowMonitor
// Kesamaan eksak hanya diklaim utk UDP: FlowMonitor mengabaikan penerusan paket yang
// sudah tidak dilacak (mis. retransmisi TCP segmen yang sudah diterima/di-drop).
static uint64_t g_totFwdUdp = 0, g_totFwdTcp = 0, g_totOrigUdp = 0, g_totLocalUdp = 0;
// FlowMonitor menambahkan penerusan sebuah paket ke FlowStats.timesForwarded HANYA saat
// paket itu diterima tujuan (FlowMonitor::ReportLastRx); penerusan paket yang kemudian
// hilang dibuang (ReportDrop / lost). Definisi yang sama di sini: hitung penerusan per
// paket UDP, tambahkan ke g_totFwdUdpDelivered pada LocalDeliver pertama.
// Kasus tepi (lihat docs/TELEMETRY.md bagian 6): entri paket hilang tidak dihapus
// (memori ~ jumlah paket hilang); paket yang sampai setelah batas 10 s FlowMonitor
// tetap dihitung di sini tetapi diabaikan FlowMonitor.
static std::map<PktKey, uint32_t> g_fwdPerPkt;
static uint64_t g_totFwdUdpDelivered = 0;

static int Win(double t)
{
  int w = (int)std::floor(t / g_W);
  return std::max(0, std::min(w, g_nWin - 1));
}
static double Now() { return Simulator::Now().GetSeconds(); }
static NodeWin &NW(int i) { return g_node[i][Win(Now())]; }
static Peer PeerOf(Mac48Address a)
{
  auto it = g_mac.find(a);
  return it == g_mac.end() ? Peer{PEER_OTHER, -1} : it->second;
}
static LinkWin &LW(int win, int obs, Peer p)
{
  return g_link[LinkKey{win, obs, (uint8_t)p.type, p.idx}];
}
static bool Unfragmented(const Ipv4Header &h)
{
  return h.IsLastFragment() && h.GetFragmentOffset() == 0;
}
static PktKey KeyOf(const Ipv4Header &h)
{
  return PktKey{h.GetSource().Get(), h.GetDestination().Get(), h.GetIdentification(),
                h.GetProtocol()};
}
// MSDU (LLC/SNAP + IPv4 ...) -> kunci paket. false jika bukan IPv4.
static bool ParseMsdu(Ptr<const Packet> msdu, PktKey &k)
{
  if (!msdu || msdu->GetSize() < 8 + 20) return false;
  Ptr<Packet> c = msdu->Copy();
  LlcSnapHeader llc;
  c->RemoveHeader(llc);
  if (llc.GetType() != 0x0800) return false;
  Ipv4Header ip;
  if (c->PeekHeader(ip) == 0) return false;
  k = KeyOf(ip);
  return true;
}
static bool DataMsdu(const WifiMacHeader &h)
{
  return h.IsData() && !h.IsQosAmsdu();
}
// MPDU lengkap (header MAC + body + FCS) -> header & kunci paket.
static bool ParsePdu(Ptr<const Packet> pdu, WifiMacHeader &hdr, PktKey &k)
{
  if (!pdu) return false;
  Ptr<Packet> c = pdu->Copy();
  if (c->RemoveHeader(hdr) == 0 || !DataMsdu(hdr)) return false;
  return ParseMsdu(c, k);
}

// ---- trace IP ----
static void OnSendOutgoing(int i, const Ipv4Header &h, Ptr<const Packet> p, uint32_t)
{
  if (h.GetProtocol() == 17) ++g_totOrigUdp;
  if (i < 0) return;
  NodeWin &w = NW(i);
  ++w.ipOrigPkts;
  w.ipOrigBytes += p->GetSize() + h.GetSerializedSize();
}
static void OnUnicastForward(int i, const Ipv4Header &h, Ptr<const Packet> p, uint32_t)
{
  if (Unfragmented(h) && h.GetProtocol() == 17)
  {
    ++g_totFwdUdp;
    ++g_fwdPerPkt[KeyOf(h)];
  }
  if (Unfragmented(h) && h.GetProtocol() == 6) ++g_totFwdTcp;
  if (i < 0) return;
  NodeWin &w = NW(i);
  ++w.ipFwdPkts;
  w.ipFwdBytes += p->GetSize() + h.GetSerializedSize();
  g_ipFate.emplace(NKey{i, KeyOf(h)}, 1);
}
static void OnLocalDeliver(int i, const Ipv4Header &h, Ptr<const Packet>, uint32_t)
{
  if (h.GetProtocol() == 17)
  {
    ++g_totLocalUdp;
    auto it = g_fwdPerPkt.find(KeyOf(h));
    if (it != g_fwdPerPkt.end())        // pengiriman pertama; duplikat berikutnya = 0
    {
      g_totFwdUdpDelivered += it->second;
      g_fwdPerPkt.erase(it);
    }
  }
  if (i >= 0) ++NW(i).ipLocalPkts;
}
static void OnIpRx(int i, Ptr<const Packet>, Ptr<Ipv4>, uint32_t) { ++NW(i).ipRxPkts; }
static void OnIpDrop(int i, const Ipv4Header &h, Ptr<const Packet>,
                     Ipv4L3Protocol::DropReason r, Ptr<Ipv4>, uint32_t)
{
  const int ri = (int)r;
  ++NW(i).ipDrop[(ri >= 1 && ri <= 7) ? ri : 0];
  g_ipFate.emplace(NKey{i, KeyOf(h)}, 10 + ri);
}

// ---- trace MAC (pengirim i) ----
static void OnAcked(int i, Ptr<const WifiMpdu> m)
{
  const WifiMacHeader &h = m->GetHeader();
  ++NW(i).macAcked;
  const Peer p = PeerOf(h.GetAddr1());
  ++LW(Win(Now()), i, p).macAcked;
  if (p.type != PEER_SENSOR || !DataMsdu(h)) return;
  PktKey k;
  if (!ParseMsdu(m->GetPacket(), k)) return;
  if (k.dst == g_sensorIp[p.idx]) return;            // bukan transit bagi j
  WKey wk{i, p.idx, k};
  if (!g_watchSeen.insert(wk).second) return;         // satu watch per (i, j, paket)
  g_watch.push_back(Watch{wk, Now()});
  ++NW(i).wdAsObserver;
}
static void OnNacked(int i, Ptr<const WifiMpdu> m)
{
  ++NW(i).macNacked;
  ++LW(Win(Now()), i, PeerOf(m->GetHeader().GetAddr1())).macNacked;
}
static void OnMacDropped(int i, WifiMacDropReason r, Ptr<const WifiMpdu> m)
{
  const bool retry = (r == WIFI_MAC_DROP_REACHED_RETRY_LIMIT);
  NodeWin &w = NW(i);
  retry ? ++w.macDropRetry : ++w.macDropOther;
  const WifiMacHeader &h = m->GetHeader();
  if (!h.GetAddr1().IsGroup())
  {
    LinkWin &l = LW(Win(Now()), i, PeerOf(h.GetAddr1()));
    retry ? ++l.macDropRetry : ++l.macDropOther;
  }
  PktKey k;
  if (DataMsdu(h) && ParseMsdu(m->GetPacket(), k))
    g_macDrop.emplace(NKey{i, k}, 1 + (int)r);
}
static void OnRespTimeout(int i, uint8_t, Ptr<const WifiMpdu> m, const WifiTxVector &)
{
  ++NW(i).macAckTimeout;
  if (m && !m->GetHeader().GetAddr1().IsGroup())
    ++LW(Win(Now()), i, PeerOf(m->GetHeader().GetAddr1())).macAckTimeout;
}

// ---- trace PHY ----
static void OnPhyTx(int i, Ptr<const Packet> pdu, double)
{
  NodeWin &w = NW(i);
  ++w.phyTxMpdu;
  w.phyTxBytes += pdu->GetSize();
  WifiMacHeader h;
  PktKey k;
  if (ParsePdu(pdu, h, k) && k.src != g_sensorIp[i])      // paket transit milik orang lain
    g_txFirst.emplace(NKey{i, k}, Now());
}
// Frame dari sensor j berisi paket transit bagi j (bukan paket asli j).
static bool TransitFrom(const WifiMacHeader &h, const PktKey &k, int &j)
{
  const Peer p = PeerOf(h.GetAddr2());
  if (p.type != PEER_SENSOR) return false;
  j = p.idx;
  return k.src != g_sensorIp[j];
}
static void OnRxOutcome(int i, Ptr<const WifiPpdu> ppdu, RxSignalInfo sig,
                        const WifiTxVector &, const std::vector<bool> &status)
{
  Ptr<const WifiPsdu> psdu = ppdu->GetPsdu();
  if (!psdu) return;
  const double t = Now();
  NodeWin &w = NW(i);
  const std::size_t n = psdu->GetNMpdus();
  for (std::size_t m = 0; m < n; ++m)
  {
    const bool ok = (m < status.size()) ? status[m] : status.empty();
    const WifiMacHeader &h = psdu->GetHeader(m);
    PktKey k;
    const bool data = DataMsdu(h) && ParseMsdu(psdu->GetPayload(m), k);
    int j = -1;
    if (ok)
    {
      ++w.phyRxOkMpdu;
      if (!h.IsCtl())
      {
        const Peer p = PeerOf(h.GetAddr2());
        LinkWin &l = LW(Win(t), i, p);
        ++l.rxOkFrom;
        l.rssiSum += sig.rssi;
        l.snrDbSum += 10.0 * std::log10(std::max(sig.snr, 1e-12));
      }
      if (data && TransitFrom(h, k, j)) g_heard.emplace(WKey{i, j, k}, t);
    }
    else
    {
      ++w.phyRxFailMpdu;
      g_failT[i].push_back(t);
      if (data && TransitFrom(h, k, j)) g_failOnFrame.emplace(WKey{i, j, k}, t);
    }
  }
}
static void OnPhyRxDrop(int i, Ptr<const Packet> pdu, WifiPhyRxfailureReason)
{
  ++NW(i).phyRxDrop;
  const double t = Now();
  g_failT[i].push_back(t);
  WifiMacHeader h;
  PktKey k;
  int j = -1;
  if (ParsePdu(pdu, h, k) && TransitFrom(h, k, j)) g_failOnFrame.emplace(WKey{i, j, k}, t);
}

// ---- energi: TracedValue TotalEnergyConsumption, dibagi linear antar window ----
static void AddEnergy(int i, double t0, double t1, double e)
{
  if (e == 0.0) return;
  if (t1 <= t0) { g_node[i][Win(t1)].energyJ += e; return; }
  double a = t0;
  while (a < t1)
  {
    const int wi = Win(a);
    const double b = (wi == g_nWin - 1) ? t1 : std::min(t1, (wi + 1) * g_W);
    g_node[i][wi].energyJ += e * (b - a) / (t1 - t0);
    if (b <= a) break;
    a = b;
  }
}
static void OnEnergy(int i, double oldV, double newV)
{
  const double t = Now();
  AddEnergy(i, g_eLastT[i], t, newV - oldV);
  g_eLastT[i] = t;
  g_eLastV[i] = newV;
}

// ---- pemasangan ----
template <typename RadioModels>
void Install(NodeContainer all, NodeContainer sensors, NetDeviceContainer devSensors,
             NetDeviceContainer devSinks, const Ipv4InterfaceContainer &ifSensors,
             RadioModels radioModels)
{
  g_on = true;
  g_n = (int)sensors.GetN();
  g_W = cfg::telemWindow;
  g_nWin = std::max(1, (int)std::ceil(cfg::simTime / g_W - 1e-9));
  g_node.assign(g_n, vector<NodeWin>(g_nWin));
  g_failT.assign(g_n, {});
  g_eLastT.assign(g_n, 0.0);
  g_eLastV.assign(g_n, 0.0);
  g_sensorIp.assign(g_n, 0);
  for (int i = 0; i < g_n; ++i)
  {
    g_sensorIp[i] = ifSensors.GetAddress(i).Get();
    g_mac[Mac48Address::ConvertFrom(devSensors.Get(i)->GetAddress())] = Peer{PEER_SENSOR, i};
  }
  for (uint32_t s = 0; s < devSinks.GetN(); ++s)
    g_mac[Mac48Address::ConvertFrom(devSinks.Get(s)->GetAddress())] = Peer{PEER_SINK, (int)s};

  std::map<uint32_t, int> nodeToSensor;
  for (int i = 0; i < g_n; ++i) nodeToSensor[sensors.Get(i)->GetId()] = i;

  // IP: semua node (total jaringan); counter per node hanya untuk sensor.
  for (uint32_t a = 0; a < all.GetN(); ++a)
  {
    Ptr<Ipv4L3Protocol> ip = all.Get(a)->GetObject<Ipv4L3Protocol>();
    if (!ip) continue;
    auto it = nodeToSensor.find(all.Get(a)->GetId());
    const int i = (it == nodeToSensor.end()) ? -1 : it->second;
    ip->TraceConnectWithoutContext("SendOutgoing", MakeBoundCallback(&OnSendOutgoing, i));
    ip->TraceConnectWithoutContext("UnicastForward", MakeBoundCallback(&OnUnicastForward, i));
    ip->TraceConnectWithoutContext("LocalDeliver", MakeBoundCallback(&OnLocalDeliver, i));
    if (i >= 0)
    {
      ip->TraceConnectWithoutContext("Rx", MakeBoundCallback(&OnIpRx, i));
      ip->TraceConnectWithoutContext("Drop", MakeBoundCallback(&OnIpDrop, i));
    }
  }
  // MAC/PHY/energi: sensor.
  for (int i = 0; i < g_n; ++i)
  {
    Ptr<WifiNetDevice> dev = DynamicCast<WifiNetDevice>(devSensors.Get(i));
    if (!dev) NS_FATAL_ERROR("Telemetry: device sensor " << i << " bukan WifiNetDevice");
    Ptr<WifiMac> mac = dev->GetMac();
    Ptr<WifiPhy> phy = dev->GetPhy();
    bool ok = mac->TraceConnectWithoutContext("AckedMpdu", MakeBoundCallback(&OnAcked, i));
    ok &= mac->TraceConnectWithoutContext("NAckedMpdu", MakeBoundCallback(&OnNacked, i));
    ok &= mac->TraceConnectWithoutContext("DroppedMpdu", MakeBoundCallback(&OnMacDropped, i));
    ok &= mac->TraceConnectWithoutContext("MpduResponseTimeout",
                                          MakeBoundCallback(&OnRespTimeout, i));
    ok &= phy->TraceConnectWithoutContext("PhyTxBegin", MakeBoundCallback(&OnPhyTx, i));
    ok &= phy->TraceConnectWithoutContext("PhyRxDrop", MakeBoundCallback(&OnPhyRxDrop, i));
    ok &= phy->GetState()->TraceConnectWithoutContext("RxOutcome",
                                                       MakeBoundCallback(&OnRxOutcome, i));
    ok &= radioModels.Get(i)->TraceConnectWithoutContext("TotalEnergyConsumption",
                                                          MakeBoundCallback(&OnEnergy, i));
    if (!ok) NS_FATAL_ERROR("Telemetry: gagal memasang trace pada sensor " << i);
  }
}

// ---- klasifikasi watchdog & audit (setelah Simulator::Run) ----
static const char *kCat[] = {"forwarded", "forwarded_late", "unobserved_rxfail",
                             "unobserved_clean"};
struct AuditKey
{
  int win, obs, nbr, cat;
  string cause;
  bool operator<(const AuditKey &o) const
  {
    return std::tie(win, obs, nbr, cat, cause) < std::tie(o.win, o.obs, o.nbr, o.cat, o.cause);
  }
};
static std::map<AuditKey, uint64_t> g_audit;
static uint64_t g_catTot[4] = {0, 0, 0, 0};
static uint64_t g_classified = 0, g_truncated = 0;

static string IpFateName(int f)
{
  static const char *drop[] = {"other", "ttl_expired", "no_route", "bad_checksum",
                               "interface_down", "route_error", "fragment_timeout",
                               "duplicate"};
  if (f == 1) return "ip_forwarded";
  const int r = f - 10;
  return string("ip_drop_") + drop[(r >= 1 && r <= 7) ? r : 0];
}

template <typename RadioModels>
void Finalize(RadioModels radioModels)
{
  const double tEnd = Now();
  for (int i = 0; i < g_n; ++i)      // segmen energi terakhir (setelah Run: aman)
  {
    const double v = radioModels.Get(i)->GetTotalEnergyConsumption();
    AddEnergy(i, g_eLastT[i], tEnd, v - g_eLastV[i]);
    g_eLastT[i] = tEnd;
    g_eLastV[i] = v;
  }
  for (const Watch &w : g_watch)
  {
    const int i = w.key.obs, j = w.key.nbr, win = Win(w.t0);
    const double tEndW = w.t0 + cfg::wdLate;
    if (tEndW > tEnd)            // timer watchdog belum habis saat simulasi berhenti
    {
      ++g_truncated;
      --g_node[i][win].wdAsObserver;
      continue;
    }
    ++g_classified;
    auto h = g_heard.find(w.key);
    const bool heard = h != g_heard.end() && h->second >= w.t0 - cfg::wdLate;
    const double d = heard ? h->second - w.t0 : 0.0;
    // Kegagalan terima PHY di observer pada [t_ack, t_ack + wdLate] (aturan yang sama
    // utk watch terdengar maupun tidak -> laju kebetulan dapat diestimasi).
    const vector<double> &f = g_failT[i];
    auto lo = std::lower_bound(f.begin(), f.end(), w.t0);
    const bool rxFailInWin = lo != f.end() && *lo <= tEndW;
    int cat;
    if (heard && d <= cfg::wdTimeout) cat = 0;
    else if (heard && d <= cfg::wdLate) cat = 1;
    else cat = rxFailInWin ? 2 : 3;
    const bool heardAfter = heard && d > cfg::wdLate;
    LinkWin &l = LW(win, i, Peer{PEER_SENSOR, j});
    NodeWin &nw = g_node[j][win];
    ++l.wdStarted;
    ++nw.obsWatched;
    ++g_catTot[cat];
    switch (cat)
    {
      case 0: ++l.wdFwd; ++nw.obsFwd; l.wdDelaySum += std::max(d, 0.0); break;
      case 1: ++l.wdLate; ++nw.obsLate; l.wdDelaySum += d; break;
      case 2: ++l.wdRxFail; ++nw.obsRxFail; break;
      default: ++l.wdClean; ++nw.obsClean; break;
    }
    if (heardAfter) { ++l.wdHeardAfter; ++nw.obsHeardAfter; }
    if (cat <= 1 && rxFailInWin) ++l.wdHeardRxFail;

    // Penyebab sebenarnya (audit).
    string cause;
    auto tx = g_txFirst.find(NKey{j, w.key.k});
    if (tx != g_txFirst.end() && tx->second <= tEndW)
    {
      if (heard && h->second <= tEndW) cause = "j_tx_heard";
      else
      {
        auto fo = g_failOnFrame.find(w.key);
        cause = (fo != g_failOnFrame.end() && fo->second <= tEndW)
                    ? "j_tx_rxfail_at_observer" : "j_tx_no_rx_event_at_observer";
      }
    }
    else if (tx != g_txFirst.end()) cause = "j_tx_after_wdlate";
    else
    {
      auto f = g_ipFate.find(NKey{j, w.key.k});
      if (f == g_ipFate.end()) cause = "j_no_ip_event";
      else if (f->second == 1)
        cause = g_macDrop.count(NKey{j, w.key.k}) ? "j_ip_forwarded_mac_dropped"
                                                  : "j_ip_forwarded_not_tx";
      else cause = "j_" + IpFateName(f->second);
    }
    ++g_audit[AuditKey{win, i, j, cat, cause}];
  }
}

static string Tag()
{
  return std::to_string(cfg::numNodes) + "_s" + std::to_string(cfg::seed);
}

static void Write()
{
  const string dir = cfg::telemDir;
  {
    std::ofstream o(dir + "/telemetry_" + Tag() + ".csv");
    if (!o) NS_FATAL_ERROR("Telemetry: tidak bisa menulis ke " << dir);
    o << std::setprecision(10);
    o << "node_id,window,t_start,t_end,"
         "self_ip_orig_pkts,self_ip_orig_bytes,self_ip_rx_pkts,self_ip_fwd_pkts,"
         "self_ip_fwd_bytes,self_ip_local_pkts,self_ip_drop_ttl,self_ip_drop_no_route,"
         "self_ip_drop_bad_checksum,self_ip_drop_iface_down,self_ip_drop_route_error,"
         "self_ip_drop_frag_timeout,self_ip_drop_duplicate,self_ip_drop_other,"
         "self_mac_acked,self_mac_nacked,self_mac_drop_retry,self_mac_drop_other,"
         "self_mac_ack_timeout,self_phy_tx_mpdu,self_phy_tx_bytes,self_phy_rx_ok_mpdu,"
         "self_phy_rx_fail_mpdu,self_phy_rx_drop,self_energy_j,wd_started_as_observer,"
         "obs_watched,obs_forwarded,obs_forwarded_late,obs_unobserved_rxfail,"
         "obs_unobserved_clean,obs_heard_after_timeout\n";
    for (int i = 0; i < g_n; ++i)
      for (int w = 0; w < g_nWin; ++w)
      {
        const NodeWin &x = g_node[i][w];
        o << i << "," << w << "," << w * g_W << "," << std::min((w + 1) * g_W, cfg::simTime)
          << "," << x.ipOrigPkts << "," << x.ipOrigBytes << "," << x.ipRxPkts << ","
          << x.ipFwdPkts << "," << x.ipFwdBytes << "," << x.ipLocalPkts;
        for (int r = 1; r <= 7; ++r) o << "," << x.ipDrop[r];
        o << "," << x.ipDrop[0] << "," << x.macAcked << "," << x.macNacked << ","
          << x.macDropRetry << "," << x.macDropOther << "," << x.macAckTimeout << ","
          << x.phyTxMpdu << "," << x.phyTxBytes << "," << x.phyRxOkMpdu << ","
          << x.phyRxFailMpdu << "," << x.phyRxDrop << "," << x.energyJ << ","
          << x.wdAsObserver << "," << x.obsWatched << "," << x.obsFwd << "," << x.obsLate
          << "," << x.obsRxFail << "," << x.obsClean << "," << x.obsHeardAfter << "\n";
      }
  }
  {
    std::ofstream o(dir + "/telemetry_links_" + Tag() + ".csv");
    o << std::setprecision(10);
    o << "window,observer,neighbor_type,neighbor_id,mac_handoff_acked,mac_handoff_nacked,"
         "mac_handoff_drop_retry,mac_handoff_drop_other,mac_ack_timeout,rx_ok_from,"
         "rssi_mean_dbm,snr_mean_db,wd_started,wd_forwarded,wd_forwarded_late,"
         "wd_unobserved_rxfail,wd_unobserved_clean,wd_heard_after_timeout,"
         "wd_heard_rxfail,wd_delay_mean_s\n";
    for (const auto &kv : g_link)
    {
      const LinkKey &k = kv.first;
      const LinkWin &x = kv.second;
      const uint64_t heard = x.wdFwd + x.wdLate;
      o << k.win << "," << k.obs << "," << kPeerName[k.ptype] << "," << k.pidx << ","
        << x.macAcked << "," << x.macNacked << "," << x.macDropRetry << "," << x.macDropOther
        << "," << x.macAckTimeout << "," << x.rxOkFrom << ",";
      if (x.rxOkFrom) o << x.rssiSum / x.rxOkFrom << "," << x.snrDbSum / x.rxOkFrom;
      else o << ",";
      o << "," << x.wdStarted << "," << x.wdFwd << "," << x.wdLate << "," << x.wdRxFail << ","
        << x.wdClean << "," << x.wdHeardAfter << "," << x.wdHeardRxFail << ",";
      if (heard) o << x.wdDelaySum / heard;
      o << "\n";
    }
  }
  {
    // Kebenaran simulator: HANYA untuk evaluasi watchdog. Bukan input trust/model.
    std::ofstream o(dir + "/watchdog_audit_" + Tag() + ".csv");
    o << "window,observer,neighbor,wd_category,true_cause,count\n";
    for (const auto &kv : g_audit)
      o << kv.first.win << "," << kv.first.obs << "," << kv.first.nbr << ","
        << kCat[kv.first.cat] << "," << kv.first.cause << "," << kv.second << "\n";
  }
  {
    std::ofstream o(dir + "/telemetry_meta_" + Tag() + ".json");
    o << std::setprecision(10);
    o << "{\n \"telemetry_version\": 1,\n \"num_nodes\": " << cfg::numNodes
      << ",\n \"seed\": " << cfg::seed << ",\n \"sim_time_s\": " << cfg::simTime
      << ",\n \"window_s\": " << g_W << ",\n \"n_windows\": " << g_nWin
      << ",\n \"wd_timeout_s\": " << cfg::wdTimeout << ",\n \"wd_late_s\": " << cfg::wdLate
      << ",\n \"route_dir\": \"" << cfg::routeDir << "\""
      << ",\n \"max_range_m\": " << cfg::maxRange
      << ",\n \"profile\": \"" << cfg::profile << "\""
      << ",\n \"wifi_std\": \"" << cfg::wifiStd << "\""
      << ",\n \"num_packets\": " << cfg::numPackets
      << ",\n \"watches\": " << g_classified
      << ",\n \"watches_truncated_at_end\": " << g_truncated
      << ",\n \"wd_forwarded\": " << g_catTot[0] << ",\n \"wd_forwarded_late\": " << g_catTot[1]
      << ",\n \"wd_unobserved_rxfail\": " << g_catTot[2]
      << ",\n \"wd_unobserved_clean\": " << g_catTot[3]
      << ",\n \"net_ip_unicast_forward_udp\": " << g_totFwdUdp
      << ",\n \"net_ip_unicast_forward_udp_delivered\": " << g_totFwdUdpDelivered
      << ",\n \"net_ip_unicast_forward_tcp\": " << g_totFwdTcp
      << ",\n \"net_ip_orig_udp\": " << g_totOrigUdp
      << ",\n \"net_ip_local_deliver_udp\": " << g_totLocalUdp << "\n}\n";
  }
  NS_LOG_UNCOND("  [Telemetry] " << g_classified << " watch (+" << g_truncated
                << " terpotong akhir sim; fwd=" << g_catTot[0] << " late=" << g_catTot[1] << " rxfail=" << g_catTot[2] << " clean="
                << g_catTot[3] << ") -> " << dir << "/telemetry_*" << Tag() << "*");
}

}  // namespace telem

// ============================================================================
//  MAIN
// ============================================================================
int main(int argc, char *argv[])
{
  string phyMode = "OfdmRate54Mbps";   // dipakai jika --wifiStd=a
  bool enableFlowMonitor = true;
  string outCsv = "results/metrics.csv";

  // Profil: "paper" (default, Tabel 4-5) atau "code" (setara kode rilis penulis).
  // Dibaca SEBELUM CommandLine agar opsi eksplisit lain tetap menimpa profil.
  for (int a = 1; a < argc; ++a)
  {
    if (string(argv[a]) == "--profile=code")
    {
      cfg::profile = "code";
      cfg::simTime = 300.0;
      cfg::numPackets = 50;
      cfg::varPktSize = false;
      cfg::tcpUsers = false;
      cfg::wifiStd = "a";
      phyMode = "OfdmRate6MbpsBW10MHz";
    }
  }

  CommandLine cmd;
  cmd.AddValue("profile",    "paper (Tabel 4-5) | code (setara kode rilis penulis)",
               cfg::profile);
  cmd.AddValue("wifiStd",    "ac (80 MHz VhtMcs2) | a (802.11a, phyMode)", cfg::wifiStd);
  cmd.AddValue("varPktSize", "ukuran paket 64..1024 byte acak", cfg::varPktSize);
  cmd.AddValue("tcpUsers",   "trafik TCP user -> cloud", cfg::tcpUsers);
  cmd.AddValue("userRate",   "laju aplikasi user (mis. 500Kbps)", cfg::userRate);
  cmd.AddValue("windowSec",  "lebar window metrik time-based (detik)", cfg::windowSec);
  cmd.AddValue("anim",       "tulis XML NetAnim (berat)", cfg::enableAnim);
  cmd.AddValue("tsCsv",      "path CSV time-series (default otomatis)", cfg::tsCsv);
  cmd.AddValue("routeDir",   "folder tldqn_nexthop_<n>_s<seed>.csv (default = aiDir)",
               cfg::routeDir);
  cmd.AddValue("maxRange",   "jangkauan radio maks (m), 0 = tanpa batas", cfg::maxRange);
  cmd.AddValue("grayholeDrop", "peluang relay malicious menjatuhkan paket transit",
               cfg::grayholeDrop);
  cmd.AddValue("telemetry",  "tulis telemetry NS-3 (observasi pasif)", cfg::telemetry);
  cmd.AddValue("telemDir",   "folder keluaran telemetry", cfg::telemDir);
  cmd.AddValue("telemWindow", "lebar window telemetry (s), mulai t=0", cfg::telemWindow);
  cmd.AddValue("wdTimeout",  "watchdog: batas delay 'forwarded' (s)", cfg::wdTimeout);
  cmd.AddValue("wdLate",     "watchdog: batas delay 'forwarded_late' / timeout (s)",
               cfg::wdLate);
  cmd.AddValue("numNodes",   "jumlah sensor node", cfg::numNodes);
  cmd.AddValue("numUsers",   "jumlah IoT user", cfg::numUsers);
  cmd.AddValue("packetSize", "ukuran paket aplikasi (byte)", cfg::packetSize);
  cmd.AddValue("numPackets", "jumlah paket", cfg::numPackets);
  cmd.AddValue("attackers",  "target DTO (-1 = 10% dari numNodes)", cfg::attackers);
  cmd.AddValue("aiDir",      "folder CSV keputusan AI (bigan_decisions_<n>_s<seed>.csv, "
                             "ai_timing.csv)", cfg::aiDir);
  cmd.AddValue("idleCurrentA",    "arus idle radio sensor (A)", cfg::idleCurrentA);
  cmd.AddValue("rxCurrentA",      "arus RX radio sensor (A)", cfg::rxCurrentA);
  cmd.AddValue("ccaBusyCurrentA", "arus CCA-busy radio sensor (A)", cfg::ccaBusyCurrentA);
  cmd.AddValue("simTime",    "durasi simulasi (detik)", cfg::simTime);
  cmd.AddValue("seed",       "RNG seed", cfg::seed);
  cmd.AddValue("phyMode",    "WiFi PHY mode", phyMode);
  cmd.AddValue("flowmon",    "aktifkan FlowMonitor", enableFlowMonitor);
  cmd.AddValue("outCsv",     "path output CSV metrik", outCsv);
  cmd.Parse(argc, argv);

  RngSeedManager::SetSeed(cfg::seed);

  if (cfg::attackers < 0) cfg::attackers = std::max(1, cfg::numNodes / 10);
  g_blocked.assign(cfg::numNodes, false);
  g_txCount.assign(cfg::numNodes, 0);

  // Muat keputusan AI (hasil Python). Berhenti jika file tidak ada.
  // Skenario AI dipilih per (jumlah node, seed): ..._<n>_s<seed>.csv
  const string nTag = std::to_string(cfg::numNodes) + "_s" +
                      std::to_string(cfg::seed);
  g_bigan.Load(cfg::aiDir + "/bigan_decisions_" + nTag + ".csv");
  g_bigan.LoadWindows(cfg::aiDir + "/bigan_windows_" + nTag + ".csv");
  if (cfg::routeDir.empty()) cfg::routeDir = cfg::aiDir;
  g_tldqn.Load(cfg::routeDir + "/tldqn_nexthop_" + nTag + ".csv", cfg::numNodes, cfg::numSink);
  if (g_bigan.Size() != (size_t)cfg::numNodes)
    NS_FATAL_ERROR("Jumlah keputusan Bi-GAN (" << g_bigan.Size() << ") != numNodes ("
                   << cfg::numNodes << "). Jalankan ulang ml/export_decisions.py.");
  // Tabel next-hop harus dibuat dari ground truth malicious yang SAMA dengan skenario.
  for (int i = 0; i < (int)cfg::numNodes; ++i)
    if (g_tldqn.ExportMalicious(i) != g_bigan.Get(i).actualMal)
      NS_FATAL_ERROR("Tabel next-hop basi: status malicious sensor " << i
                     << " berbeda dari bigan_decisions_" << nTag
                     << ".csv. Jalankan ulang ml/export_routes.py.");
  if (cfg::tcpUsers && cfg::maxRange > 0)
    NS_LOG_UNCOND("  [PERINGATAN] tcpUsers=true dgn maxRange=" << cfg::maxRange
                  << " m: tidak ada rute user -> cloud; aliran TCP sebagian besar gagal "
                     "(metrik tcp_* tidak bermakna).");
  LoadAiTiming(cfg::aiDir + "/ai_timing.csv", cfg::numNodes,
               g_aiBiganMs, g_aiTldqnMs);

  NS_LOG_UNCOND("\n######################################################");
  NS_LOG_UNCOND("#  HieMulti-Block: AI-Based IDS/IPS di Edge-Assisted SDWSN");
  NS_LOG_UNCOND("#  Topologi: Modified Honeycomb Structure (ns-3.48)");
  NS_LOG_UNCOND("######################################################\n");
  NS_LOG_UNCOND("Jaringan: " << cfg::numNodes << " sensor, " << cfg::numUsers
                << " user, " << cfg::numSink << " edge-sink, " << cfg::numSwitch
                << " switch, 1 master + 2 slave controller, "
                << cfg::numBlock << " blockchain, 1 cloud, 1 TA.");

  // --- Buat node ---
  NodeContainer sensors;    sensors.Create(cfg::numNodes);
  NodeContainer users;      users.Create(cfg::numUsers);
  NodeContainer blockchain; blockchain.Create(cfg::numBlock);
  NodeContainer ta;         ta.Create(cfg::numTA);
  NodeContainer switches;   switches.Create(cfg::numSwitch);
  NodeContainer sinks;      sinks.Create(cfg::numSink);
  NodeContainer master;     master.Create(cfg::numMaster);
  NodeContainer slaves;     slaves.Create(cfg::numSlave);
  NodeContainer cloud;      cloud.Create(cfg::numCloud);

  // --- WiFi (ns-3.48 API) ---
  WifiHelper wifi;
  YansWifiPhyHelper wifiPhy;
  if (cfg::wifiStd == "ac")
  {
    // Tabel 5: data rate 100 Mbps, bandwidth 100 MHz. 802.11ac tidak punya lebar
    // 100 MHz; yang terdekat 80 MHz + VhtMcs2 (~88 Mbps). Lihat docs/PAPER_DEVIATIONS.md.
    wifi.SetStandard(WIFI_STANDARD_80211ac);
    wifi.SetRemoteStationManager("ns3::ConstantRateWifiManager",
                                 "DataMode", StringValue("VhtMcs2"),
                                 "ControlMode", StringValue("OfdmRate24Mbps"));
    wifiPhy.Set("ChannelSettings", StringValue("{0, 80, BAND_5GHZ, 0}"));
  }
  else
  {
    // Setara kode rilis penulis (802.11a).
    wifi.SetStandard(WIFI_STANDARD_80211a);
    wifi.SetRemoteStationManager("ns3::ConstantRateWifiManager",
                                 "DataMode", StringValue(phyMode),
                                 "ControlMode", StringValue(phyMode));
  }
  wifiPhy.Set("RxGain", DoubleValue(-30));
  wifiPhy.SetPcapDataLinkType(YansWifiPhyHelper::DLT_IEEE802_11_RADIO);

  YansWifiChannelHelper wifiChannel;
  wifiChannel.SetPropagationDelay("ns3::ConstantSpeedPropagationDelayModel");
  wifiChannel.AddPropagationLoss("ns3::NakagamiPropagationLossModel");
  if (cfg::maxRange > 0)     // Tabel 5: jangkauan transmisi 150 m
    wifiChannel.AddPropagationLoss("ns3::RangePropagationLossModel",
                                   "MaxRange", DoubleValue(cfg::maxRange));
  wifiPhy.SetChannel(wifiChannel.Create());

  WifiMacHelper wifiMac;
  wifiMac.SetType("ns3::AdhocWifiMac");

  NetDeviceContainer devSensors = wifi.Install(wifiPhy, wifiMac, sensors);
  NetDeviceContainer devUsers   = wifi.Install(wifiPhy, wifiMac, users);
  NetDeviceContainer devSinks   = wifi.Install(wifiPhy, wifiMac, sinks);
  NetDeviceContainer devSwitch  = wifi.Install(wifiPhy, wifiMac, switches);
  NetDeviceContainer devCloud   = wifi.Install(wifiPhy, wifiMac, cloud);
  NetDeviceContainer devSlaves  = wifi.Install(wifiPhy, wifiMac, slaves);
  NetDeviceContainer devMaster  = wifi.Install(wifiPhy, wifiMac, master);

  // --- Energy model (Tabel 5: initial energy 100 J, tx power 0.005 W) ---
  // Catatan: di ns-3.48 pada build ini, helper energy berada di namespace
  // ns3:: langsung (bukan ns3::energy::).
  BasicEnergySourceHelper energySrcHelper;
  energySrcHelper.Set("BasicEnergySourceInitialEnergyJ",
                      DoubleValue(cfg::initEnergy));
  // Pakai auto: EnergySourceContainer ada di ns3::energy:: pada build ini,
  // sedangkan helper-nya di ns3:: — auto menghindari ketidakcocokan namespace.
  auto energySources = energySrcHelper.Install(sensors);

  // Arus radio kelas sensor (mis. CC2420: idle ~0.4 mA, RX ~19 mA) agar energi
  // tidak didominasi arus idle radio WiFi default (273 mA) yang tak realistis
  // untuk sensor. Tx mengikuti Tabel 5 (0.005 W pada 3 V). Dengan ini energi
  // dan lifetime ikut dipengaruhi volume trafik (flooding vs isolasi IDS).
  WifiRadioEnergyModelHelper radioEnergyHelper;
  radioEnergyHelper.Set("TxCurrentA", DoubleValue(cfg::txPower / 3.0));
  radioEnergyHelper.Set("RxCurrentA", DoubleValue(cfg::rxCurrentA));
  radioEnergyHelper.Set("IdleCurrentA", DoubleValue(cfg::idleCurrentA));
  radioEnergyHelper.Set("CcaBusyCurrentA", DoubleValue(cfg::ccaBusyCurrentA));
  auto radioModels = radioEnergyHelper.Install(devSensors, energySources);

  // --- Mobility ---
  // User: Random Waypoint (Tabel 5). Lainnya: posisi konstan (honeycomb).
  MobilityHelper mobUser;
  ObjectFactory pos;
  pos.SetTypeId("ns3::RandomRectanglePositionAllocator");
  pos.Set("X", StringValue("ns3::UniformRandomVariable[Min=1.0|Max=1000.0]"));
  pos.Set("Y", StringValue("ns3::UniformRandomVariable[Min=1.0|Max=1000.0]"));
  Ptr<PositionAllocator> alloc = pos.Create()->GetObject<PositionAllocator>();
  mobUser.SetPositionAllocator(alloc);
  mobUser.SetMobilityModel("ns3::RandomWaypointMobilityModel",
                           "Speed", StringValue(
                             "ns3::UniformRandomVariable[Min=0.0|Max=25.0]"),
                           "Pause", StringValue(
                             "ns3::ConstantRandomVariable[Constant=0]"),
                           "PositionAllocator", PointerValue(alloc));
  mobUser.Install(users);

  MobilityHelper mobFixed;
  mobFixed.SetMobilityModel("ns3::ConstantPositionMobilityModel");
  mobFixed.Install(sensors);
  mobFixed.Install(switches);
  mobFixed.Install(sinks);
  mobFixed.Install(blockchain);
  mobFixed.Install(ta);
  mobFixed.Install(master);
  mobFixed.Install(slaves);
  mobFixed.Install(cloud);

  // Posisi honeycomb (dari kode asli penulis).
  for (uint32_t i = 0; i < sensors.GetN() && i < 100; ++i)
    AnimationInterface::SetConstantPosition(sensors.Get(i),
                                            g_sensorPos[i][0],
                                            g_sensorPos[i][1]);
  for (uint32_t i = 0; i < sinks.GetN() && i < 4; ++i)
    AnimationInterface::SetConstantPosition(sinks.Get(i),
                                            g_sinkPos[i][0], g_sinkPos[i][1]);

  // Posisi infrastruktur (dari kode asli).
  AnimationInterface::SetConstantPosition(blockchain.Get(0), 200, 8);
  AnimationInterface::SetConstantPosition(blockchain.Get(1), 400, 8);
  AnimationInterface::SetConstantPosition(blockchain.Get(2), 800, 8);
  AnimationInterface::SetConstantPosition(cloud.Get(0), 600, 8);
  AnimationInterface::SetConstantPosition(ta.Get(0), 450, 450);
  AnimationInterface::SetConstantPosition(master.Get(0), 450, 100);
  AnimationInterface::SetConstantPosition(slaves.Get(0), 350, 350);
  AnimationInterface::SetConstantPosition(slaves.Get(1), 600, 350);
  AnimationInterface::SetConstantPosition(switches.Get(0), 225, 225);
  AnimationInterface::SetConstantPosition(switches.Get(1), 225, 675);
  AnimationInterface::SetConstantPosition(switches.Get(2), 675, 225);
  AnimationInterface::SetConstantPosition(switches.Get(3), 675, 675);

  // --- Internet stack + routing ---
  InternetStackHelper internet;
  internet.Install(sensors);
  internet.Install(users);
  internet.Install(switches);
  internet.Install(sinks);
  internet.Install(blockchain);
  internet.Install(ta);
  internet.Install(master);
  internet.Install(slaves);
  internet.Install(cloud);

  // Semua device WiFi berada pada SATU channel adhoc (satu broadcast domain
  // L2), jadi harus SATU subnet (/16) agar bisa saling jangkau langsung tanpa
  // router. Jika dibagi per-subnet, aliran lintas-subnet tidak ter-route dan
  // FlowMonitor mencatat 0.
  Ipv4AddressHelper ipv4;
  ipv4.SetBase("10.1.0.0", "255.255.0.0");
  Ipv4InterfaceContainer ifSensors = ipv4.Assign(devSensors);
  ipv4.Assign(devUsers);
  g_sinkIf = ipv4.Assign(devSinks);   // simpan IP sink untuk aliran unicast
  ipv4.Assign(devSwitch);
  Ipv4InterfaceContainer ifCloud = ipv4.Assign(devCloud);
  ipv4.Assign(devSlaves);
  ipv4.Assign(devMaster);

  // ==========================================================================
  //  ROUTING TLDQN MULTI-HOP: host route /32 ke tiap sink lewat next-hop TLDQN
  //  (mengalahkan rute subnet /16). Validasi: next-hop harus dalam jangkauan.
  // ==========================================================================
  {
    Ipv4StaticRoutingHelper srh;
    uint32_t installed = 0;
    for (uint32_t i = 0; i < sensors.GetN(); ++i) g_ipToSensor[ifSensors.GetAddress(i)] = (int)i;
    for (const auto &kv : g_tldqn.Table())
    {
      const int c = kv.first.first, d = kv.first.second;
      const Tldqn::Entry &e = kv.second;
      Ptr<Node> nh = e.toSink ? sinks.Get(e.nextHop) : sensors.Get(e.nextHop);
      const double dist = CalculateDistance(
          sensors.Get(c)->GetObject<MobilityModel>()->GetPosition(),
          nh->GetObject<MobilityModel>()->GetPosition());
      if (cfg::maxRange > 0 && dist > cfg::maxRange + 1e-6)
        NS_FATAL_ERROR("Next-hop " << (e.toSink ? "sink " : "sensor ") << e.nextHop
                       << " dari sensor " << c << " berjarak " << dist << " m > "
                       << cfg::maxRange << " m (tabel tidak konsisten dengan topologi)");
      Ptr<Ipv4> ip = sensors.Get(c)->GetObject<Ipv4>();
      const uint32_t ifIdx = (uint32_t)ip->GetInterfaceForDevice(devSensors.Get(c));
      const Ipv4Address gw = e.toSink ? g_sinkIf.GetAddress(e.nextHop)
                                      : ifSensors.GetAddress(e.nextHop);
      srh.GetStaticRouting(ip)->AddHostRouteTo(g_sinkIf.GetAddress(d), gw, ifIdx);
      ++installed;
    }
    // Grayhole pada relay malicious (ground truth skenario).
    uint32_t gh = 0;
    for (uint32_t i = 0; i < sensors.GetN(); ++i)
    {
      if (!g_bigan.Get((int)i).actualMal) continue;
      Ptr<Ipv4ListRouting> list =
          DynamicCast<Ipv4ListRouting>(sensors.Get(i)->GetObject<Ipv4>()->GetRoutingProtocol());
      if (!list) NS_FATAL_ERROR("Routing sensor " << i << " bukan Ipv4ListRouting");
      Ptr<GrayholeRouting> g = CreateObject<GrayholeRouting>();
      g->Configure(cfg::grayholeDrop, 100000 + (int64_t)i);
      list->AddRoutingProtocol(g, 100);
      ++gh;
    }
    NS_LOG_UNCOND("  [TLDQN] " << installed << " host route dipasang; " << gh
                  << " relay grayhole (pDrop=" << cfg::grayholeDrop << ").");
  }

  // ==========================================================================
  //  TAHAP 1: SECURE AUTHENTICATION (3D Cube + DNN) -> simpan di blockchain
  // ==========================================================================
  NS_LOG_UNCOND("\n=== TAHAP 1: Secure Authentication (3D Cube + DNN) ===");
  NS_LOG_UNCOND("  TA mem-registrasi node (ID, PUF, lokasi, random) lalu "
                "meng-generate private key via 3D Cube.");
  {
    std::ofstream ofsDev("results/PrivatekeyRegDevce.txt");
    for (uint32_t j = 0; j < sensors.GetN(); ++j)
    {
      string key = g_cube.PrivateKey(j + 2);
      ofsDev << "IoTdevice-" << (j + 1) << " PrivateKey=" << key << "\n";
      g_chain.StoreTransaction("dev:" + key); // simpan ke HieMulti-Block
    }
    std::ofstream ofsUser("results/PrivatekeyRegUser.txt");
    for (uint32_t j = 0; j < users.GetN(); ++j)
    {
      string key = g_cube.PrivateKey(j + 7);
      ofsUser << "User-" << (j + 1) << " PrivateKey=" << key << "\n";
      g_chain.StoreTransaction("user:" + key);
    }
    const uint32_t entities = sensors.GetN() + users.GetN();
    g_ctrlMsgs += 2ull * entities;   // registrasi + penerimaan private key

    // Waktu komputasi 3D Cube per entitas: warm-up lalu rata-rata banyak
    // ulangan (satu kali ukur terlalu kecil & kena efek cold-start).
    size_t acc = 0;
    for (int w = 0; w < 500; ++w) acc += g_cube.PrivateKey(w + 2).size();
    const int reps = 5000;
    const auto tAuth0 = std::chrono::steady_clock::now();
    for (int r = 0; r < reps; ++r) acc += g_cube.PrivateKey((r % 512) + 2).size();
    const auto tAuth1 = std::chrono::steady_clock::now();
    g_authMsPerEntity =
        std::chrono::duration<double, std::milli>(tAuth1 - tAuth0).count() / reps;
    g_authSink = acc;   // cegah optimisasi membuang loop pengukuran
  }
  NS_LOG_UNCOND("  Private key disimpan di HieMulti-Block (lihat results/).");
  NS_LOG_UNCOND("  Clone attack detection dilakukan (validasi random number).");
  g_chain.Report();

  // ==========================================================================
  //  Penjadwalan fase (waktu 1.3s, 8.3s, 15.3s seperti kode asli penulis)
  // ==========================================================================
  //  Urutan paper (Sec. IV-B): clustering -> routing TLDQN -> hybrid IDS ->
  //  prevention. Routing harus sebelum data mengalir, jadi tahap 2+3 di 1.3 s.
  // ==========================================================================
  SetupSinkReceivers();   // receiver UDP di tiap edge-sink (sekali)
  Simulator::Schedule(Seconds(1.3),  &PhaseCluster, sensors);
  Simulator::Schedule(Seconds(8.3),  &PhaseIds<decltype(energySources)>,
                      sensors, energySources);
  Simulator::Schedule(Seconds(15.3), &PhasePrevention, sensors);
  if (cfg::tcpUsers) SetupUserTcp(users, cloud.Get(0), ifCloud);
  for (int k = 1; k < g_bigan.Windows(); ++k)
  {
    const double tk = 8.3 + k * cfg::windowSec;
    if (tk >= cfg::simTime) break;
    Simulator::Schedule(Seconds(tk), &PhaseMonitor, k);
  }

  // --- NetAnim (Fig. 8) --- opsional (--anim=true); berat utk sweep.
  std::unique_ptr<AnimationInterface> animPtr;
  if (cfg::enableAnim)
  {
    animPtr = std::make_unique<AnimationInterface>(
        "results/Modified_Honeycomb_Structure.xml");
    AnimationInterface &anim = *animPtr;
    anim.SetMaxPktsPerTraceFile(5000000);
    for (uint32_t i = 0; i < sensors.GetN(); ++i)
    {
      anim.UpdateNodeDescription(sensors.Get(i), "Sensor");
      anim.UpdateNodeColor(sensors.Get(i), 255, 255, 0);
    }
    for (uint32_t i = 0; i < users.GetN(); ++i)
      anim.UpdateNodeDescription(users.Get(i), "User");
    for (uint32_t i = 0; i < sinks.GetN(); ++i)
      anim.UpdateNodeDescription(sinks.Get(i), "Edge_Sink");
    anim.UpdateNodeDescription(master.Get(0), "Master_Controller");
    anim.UpdateNodeDescription(slaves.Get(0), "Slave_Controller");
    anim.UpdateNodeDescription(slaves.Get(1), "Slave_Controller");
    anim.UpdateNodeDescription(cloud.Get(0), "Cloud");
    anim.UpdateNodeDescription(ta.Get(0), "TA");
  }

  // --- FlowMonitor ---
  // FlowMonitor bisa dimatikan (--flowmon=false) HANYA untuk diagnosis memori;
  // tanpa FlowMonitor metrik jaringan (PDR, latency, throughput) bernilai 0.
  FlowMonitorHelper flowmon;
  Ptr<FlowMonitor> monitor;
  if (enableFlowMonitor) monitor = flowmon.InstallAll();

  // --- Telemetry NS-3 (langkah 2): hanya memasang trace sink ---
  if (cfg::telemetry)
  {
    if (cfg::telemWindow <= 0 || cfg::wdTimeout <= 0 || cfg::wdLate < cfg::wdTimeout)
      NS_FATAL_ERROR("Telemetry: butuh telemWindow > 0 dan 0 < wdTimeout <= wdLate");
    telem::Install(NodeContainer::GetGlobal(), sensors, devSensors, devSinks, ifSensors,
                   radioModels);
  }

  Simulator::Stop(Seconds(cfg::simTime));
  Simulator::Run();
  if (cfg::telemetry) telem::Finalize(radioModels);

  // ==========================================================================
  //  TAHAP 6: EVALUASI METRIK (9 metrik + confusion matrix)
  // ==========================================================================
  Ptr<Ipv4FlowClassifier> classifier;
  FlowMonitor::FlowStatsContainer stats;
  if (enableFlowMonitor)
  {
    monitor->CheckForLostPackets();
    classifier = DynamicCast<Ipv4FlowClassifier>(flowmon.GetClassifier());
    stats = monitor->GetFlowStats();
  }

  uint64_t txPackets = 0, rxPackets = 0, lostPackets = 0;
  uint64_t txBytes = 0, rxBytes = 0;
  double   delaySum = 0.0;
  uint64_t tcpTx = 0, tcpRx = 0, tcpRxBytes = 0;
  uint64_t benignFwd = 0, benignRx = 0;     // aliran data sensor NORMAL -> sink
  uint64_t fmFwdUdp = 0, fmFwdTcp = 0;      // timesForwarded per protokol (validasi telemetry)
  for (auto &kv : stats)
  {
    // Metrik utama = trafik sensor (UDP). TCP user->cloud dilaporkan terpisah.
    Ipv4FlowClassifier::FiveTuple ft = classifier->FindFlow(kv.first);
    if (ft.protocol == 17) fmFwdUdp += kv.second.timesForwarded;
    else if (ft.protocol == 6) fmFwdTcp += kv.second.timesForwarded;
    if (ft.protocol == 6)
    {
      tcpTx += kv.second.txPackets;
      tcpRx += kv.second.rxPackets;
      tcpRxBytes += kv.second.rxBytes;
      continue;
    }
    txPackets += kv.second.txPackets;
    {
      auto it = g_ipToSensor.find(ft.sourceAddress);
      if (it != g_ipToSensor.end() && !g_bigan.Get(it->second).actualMal)
      {
        benignFwd += kv.second.timesForwarded;
        benignRx += kv.second.rxPackets;
      }
    }
    rxPackets += kv.second.rxPackets;
    lostPackets += kv.second.lostPackets;
    txBytes += kv.second.txBytes;
    rxBytes += kv.second.rxBytes;
    delaySum += kv.second.delaySum.GetSeconds();
  }

  double pdr = txPackets ? (100.0 * rxPackets / txPackets) : 0.0;
  double dropRatio = txPackets ? (100.0 * lostPackets / txPackets) : 0.0;
  double avgLatencyMs = rxPackets ? (1000.0 * delaySum / rxPackets) : 0.0;
  double throughputKbps = cfg::simTime > 0
      ? (rxBytes * 8.0 / 1000.0 / cfg::simTime) : 0.0;

  // Energi: total - sisa, pada sumber energi sensor.
  double totalEnergyJ = cfg::numNodes * cfg::initEnergy;
  double remainingJ = 0.0;
  for (auto it = energySources.Begin(); it != energySources.End(); ++it)
    remainingJ += (*it)->GetRemainingEnergy();
  double energyConsumedJ = totalEnergyJ - remainingJ;

  // Detection accuracy (pers. 47).
  double detAcc = (g_tp + g_tn + g_fp + g_fn) > 0
      ? (100.0 * (g_tp + g_tn) / (g_tp + g_tn + g_fp + g_fn)) : 0.0;
  double precision = (g_tp + g_fp) > 0 ? (100.0 * g_tp / (g_tp + g_fp)) : 0.0;
  double recall    = (g_tp + g_fn) > 0 ? (100.0 * g_tp / (g_tp + g_fn)) : 0.0;
  double f1 = (precision + recall) > 0
      ? (2.0 * precision * recall / (precision + recall)) : 0.0;

  // Network lifetime (proyeksi): umur sampai energi awal habis pada daya
  // rata-rata per node yang terukur selama simulasi (pendekatan Code 5 paper).
  double avgPowerW = (cfg::numNodes > 0 && cfg::simTime > 0)
      ? energyConsumedJ / (cfg::numNodes * cfg::simTime) : 0.0;
  double lifetimeS = avgPowerW > 0 ? cfg::initEnergy / avgPowerW : 0.0;

  // Computation overhead (pers. 46, RT/CT x 100): RT = waktu komputasi kontrol
  // per node (3D Cube + Bi-GAN + TLDQN, terukur); CT = RT + latency rata-rata.
  double rtMs = g_authMsPerEntity + g_aiBiganMs + g_aiTldqnMs;
  double ctMs = rtMs + avgLatencyMs;
  double compOverheadPct = ctMs > 0 ? 100.0 * rtMs / ctMs : 0.0;

  // Control overhead: pesan kontrol logis per detik (mps).
  double controlOverheadMps = cfg::simTime > 0
      ? double(g_ctrlMsgs) / cfg::simTime : 0.0;

  // --- Cetak ringkasan ---
  NS_LOG_UNCOND("\n######################################################");
  NS_LOG_UNCOND("#  TAHAP 6: HASIL EVALUASI (node=" << cfg::numNodes << ")");
  NS_LOG_UNCOND("######################################################");
  NS_LOG_UNCOND("Energy consumption : " << energyConsumedJ << " J");
  NS_LOG_UNCOND("Avg latency        : " << avgLatencyMs << " ms");
  NS_LOG_UNCOND("Throughput         : " << throughputKbps << " kbps");
  NS_LOG_UNCOND("PDR                : " << pdr << " %");
  NS_LOG_UNCOND("Packet drop ratio  : " << dropRatio << " %");
  NS_LOG_UNCOND("Detection accuracy : " << detAcc << " %");
  NS_LOG_UNCOND("Precision          : " << precision << " %");
  NS_LOG_UNCOND("Recall             : " << recall << " %");
  NS_LOG_UNCOND("F1-score           : " << f1 << " %");
  NS_LOG_UNCOND("Confusion matrix   : TP=" << g_tp << " TN=" << g_tn
                << " FP=" << g_fp << " FN=" << g_fn);
  NS_LOG_UNCOND("Network lifetime   : " << lifetimeS << " s (proyeksi)");
  NS_LOG_UNCOND("Computation ovhd   : " << compOverheadPct << " % (RT="
                << rtMs << " ms)");
  NS_LOG_UNCOND("Control overhead   : " << controlOverheadMps << " mps");

  // --- Tulis CSV (dipakai ml/metrics_plots.py) ---
  {
    std::ofstream csv(outCsv);
    csv << "metric,value\n";
    csv << "numNodes," << cfg::numNodes << "\n";
    csv << "energy_consumption_J," << energyConsumedJ << "\n";
    csv << "latency_ms," << avgLatencyMs << "\n";
    csv << "throughput_kbps," << throughputKbps << "\n";
    csv << "pdr_percent," << pdr << "\n";
    csv << "packet_drop_ratio_percent," << dropRatio << "\n";
    csv << "detection_accuracy_percent," << detAcc << "\n";
    csv << "precision_percent," << precision << "\n";
    csv << "recall_percent," << recall << "\n";
    csv << "f1_percent," << f1 << "\n";
    csv << "TP," << g_tp << "\n";
    csv << "TN," << g_tn << "\n";
    csv << "FP," << g_fp << "\n";
    csv << "FN," << g_fn << "\n";
    csv << "network_lifetime_s," << lifetimeS << "\n";
    csv << "windows," << g_win.size() << "\n";
    csv << "tcp_throughput_kbps," << (cfg::simTime > 0
        ? tcpRxBytes * 8.0 / 1000.0 / cfg::simTime : 0.0) << "\n";
    csv << "tcp_pdr_percent," << (tcpTx ? 100.0 * tcpRx / tcpTx : 0.0) << "\n";
    csv << "computation_overhead_percent," << compOverheadPct << "\n";
    csv << "control_overhead_mps," << controlOverheadMps << "\n";
    csv << "dto_TP," << g_dtoTp << "\n";
    csv << "dto_TN," << g_dtoTn << "\n";
    csv << "dto_FP," << g_dtoFp << "\n";
    csv << "dto_FN," << g_dtoFn << "\n";
    csv << "nodes_blocked," << g_blockedCount << "\n";
    csv << "attackers_blocked," << g_blockedAttackers << "\n";
    csv << "rt_compute_ms," << rtMs << "\n";
    // Routing multi-hop (integrasi TLDQN langkah 1)
    double plannedHops = 0.0, plannedBenign = 0.0;
    int nBenign = 0;
    for (int i = 0; i < (int)cfg::numNodes; ++i)
    {
      plannedHops += g_tldqn.PlannedHops(i);
      if (!g_bigan.Get(i).actualMal) { plannedBenign += g_tldqn.PlannedHops(i); ++nBenign; }
    }
    csv << "max_range_m," << cfg::maxRange << "\n";
    csv << "route_entries," << g_tldqn.Size() << "\n";
    csv << "planned_hops_mean," << plannedHops / std::max<int>(cfg::numNodes, 1) << "\n";
    csv << "planned_hops_benign_mean," << plannedBenign / std::max(nBenign, 1) << "\n";
    // Estimasi hop aliran sensor NORMAL = 1 + timesForwarded/rxPackets. timesForwarded
    // (FlowMonitor) = jumlah penerusan oleh relay (tidak termasuk kirim sumber) dan juga
    // mencakup paket yang kemudian hilang -> sedikit lebih besar dari hop paket diterima.
    csv << "benign_hops_per_rx_est," << (benignRx ? 1.0 + double(benignFwd) / benignRx : 0.0)
        << "\n";
    csv << "grayhole_drops," << g_grayholeDrops << "\n";
    csv << "grayhole_not_dropped," << g_grayholePassed << "\n";
    csv << "grayhole_drop_prob," << cfg::grayholeDrop << "\n";
    // Validasi telemetry (langkah 2): definisi counter sama dengan FlowMonitor
    // (UnicastForward UDP non-fragmen; SendOutgoing UDP; LocalDeliver UDP). TCP
    // dilaporkan tanpa klaim kesamaan (lihat telem::g_totFwdUdp).
    csv << "telemetry_enabled," << (cfg::telemetry ? 1 : 0) << "\n";
    if (cfg::telemetry)
    {
      // total = semua penerusan UDP; delivered = penerusan paket yang sampai tujuan
      // (= definisi FlowMonitor timesForwarded, lihat telem::g_fwdPerPkt)
      csv << "telemetry_ip_forward_udp_total," << telem::g_totFwdUdp << "\n";
      csv << "telemetry_ip_forward_udp_delivered," << telem::g_totFwdUdpDelivered << "\n";
      csv << "flowmon_udp_times_forwarded," << fmFwdUdp << "\n";
      csv << "telemetry_ip_forward_tcp_total," << telem::g_totFwdTcp << "\n";
      csv << "flowmon_tcp_times_forwarded," << fmFwdTcp << "\n";
      csv << "telemetry_ip_orig_udp_total," << telem::g_totOrigUdp << "\n";
      csv << "flowmon_udp_tx_packets," << txPackets << "\n";
      csv << "telemetry_ip_local_udp_total," << telem::g_totLocalUdp << "\n";
      csv << "flowmon_udp_rx_packets," << rxPackets << "\n";
      csv << "wd_watches," << telem::g_classified << "\n";
      csv << "wd_forwarded," << telem::g_catTot[0] << "\n";
      csv << "wd_forwarded_late," << telem::g_catTot[1] << "\n";
      csv << "wd_unobserved_rxfail," << telem::g_catTot[2] << "\n";
      csv << "wd_unobserved_clean," << telem::g_catTot[3] << "\n";
      // presisi penuh: dibandingkan dgn jumlah energi per window (validator)
      csv << "sensor_energy_consumed_J," << std::setprecision(12) << energyConsumedJ
          << std::setprecision(6) << "\n";
    }
    NS_LOG_UNCOND("\nMetrik ditulis ke: " << outCsv);
    if (cfg::telemetry && enableFlowMonitor && telem::g_totFwdUdpDelivered != fmFwdUdp)
      NS_LOG_UNCOND("  [PERINGATAN] telemetry_ip_forward_udp_delivered ("
                    << telem::g_totFwdUdpDelivered << ") != FlowMonitor UDP timesForwarded ("
                    << fmFwdUdp << ")");
  }

  // Time-series deteksi per window (utk Fig. 18 time-based).
  {
    string tsPath = cfg::tsCsv.empty()
        ? "results/timeseries_" + std::to_string(cfg::numNodes) + "_s" +
              std::to_string(cfg::seed) + ".csv"
        : cfg::tsCsv;
    std::ofstream ts(tsPath);
    ts << "window,t,TP,TN,FP,FN\n";
    for (size_t k = 0; k < g_win.size(); ++k)
      ts << k << "," << (8.3 + k * cfg::windowSec) << "," << g_win[k][0] << ","
         << g_win[k][1] << "," << g_win[k][2] << "," << g_win[k][3] << "\n";
  }

  if (cfg::telemetry) telem::Write();
  if (enableFlowMonitor) monitor->SerializeToXmlFile("results/flowmon.xml", true, true);
  Simulator::Destroy();
  return 0;
}
