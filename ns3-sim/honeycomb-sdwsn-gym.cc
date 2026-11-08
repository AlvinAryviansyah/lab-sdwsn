/*
 * honeycomb-sdwsn-gym.cc
 *
 * Lingkaran online TLDQN <-> ns-3 lewat ns3-gym (OpenGym).
 * Satu proses ns-3 = satu episode = satu paket hop-by-hop menuju sink terdekat.
 *
 * Kontrak keadaan sama dengan ml/routing_env.py (STATE_DIM = 44):
 *   6 slot x 7 fitur: valid, trust, link_quality, stability, energy,
 *                     progress, is_dest
 *   lalu 2 fitur node: remaining, own_energy
 * Aksi: indeks slot 0..5. Slot diurutkan (jarak ke tujuan, id node),
 * aturan kandidat sama dengan honeycomb_topology.py.
 *
 * Reward (sama dengan EnvConfig di routing_env.py):
 *   sampai di sink          +1
 *   hilang / TTL / aksi buruk -1
 *   hop                     -0.05 - 0.10 * (1 - energi kandidat)
 *
 * Bangun (modul opengym dari ns3-gym harus ada di contrib/):
 *   cp ns3-sim/honeycomb-sdwsn-gym.cc $NS3_HOME/scratch/
 *   cd $NS3_HOME && ./ns3 build honeycomb-sdwsn-gym
 * Lalu dari ml/:  python tldqn_routing_gym.py --ns3-dir $NS3_HOME
 *
 * Tanpa modul OpenGym, file ini tidak ikut dipakai honeycomb-sdwsn.cc
 * dan tidak mengubah jalur CSV yang sudah ada.
 */

#include "ns3/core-module.h"
#include "ns3/mobility-module.h"
#include "ns3/opengym-module.h"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <string>
#include <vector>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("HoneycombSdwsnGym");

static const uint32_t K_SLOTS = 6;
static const uint32_t SLOT_FEATURES = 7;
static const uint32_t NODE_FEATURES = 2;
static const uint32_t STATE_DIM = K_SLOTS * SLOT_FEATURES + NODE_FEATURES;

static const double TX_RANGE = 150.0;
static const double PRR_D50 = 120.0;
static const double PRR_WIDTH = 10.0;
static const double STABILITY_MIN = 0.80;
static const double P_FWD_NORMAL = 0.98;
static const double P_FWD_MALICIOUS = 0.30;
static const int TRUST_OBS = 10;
static const double ENERGY_MIN = 0.30;
static const double R_SUCCESS = 1.0;
static const double R_FAIL = 1.0;
static const double C_HOP = 0.05;
static const double C_ENERGY = 0.10;
static const int MAX_HOPS = 12;

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

class HoneycombGymEnv : public OpenGymEnv
{
public:
  HoneycombGymEnv ();
  ~HoneycombGymEnv () override;

  static TypeId GetTypeId ();
  void DoDispose () override;

  Ptr<OpenGymSpace> GetActionSpace () override;
  Ptr<OpenGymSpace> GetObservationSpace () override;
  bool GetGameOver () override;
  Ptr<OpenGymDataContainer> GetObservation () override;
  float GetReward () override;
  std::string GetExtraInfo () override;
  bool ExecuteActions (Ptr<OpenGymDataContainer> action) override;

  void Configure (uint32_t nSensors, uint32_t seed);
  void Start ();

private:
  struct Node
  {
    double x {0};
    double y {0};
    bool sink {false};
    bool malicious {false};
    double energy {1};
    double trust {1};
  };

  double Dist (uint32_t a, uint32_t b) const;
  double Prr (uint32_t a, uint32_t b) const;
  uint32_t NearestSink (uint32_t sensor) const;
  std::vector<int> Candidates (uint32_t cur, uint32_t dest) const;
  void FillObservation (uint32_t cur, uint32_t dest);
  void BeginEpisode ();

  uint32_t m_nSensors {20};
  uint32_t m_nSinks {4};
  std::vector<Node> m_nodes;
  std::vector<std::vector<double>> m_stability;

  uint32_t m_cur {0};
  uint32_t m_dest {0};
  int m_hops {0};
  bool m_gameOver {false};
  float m_reward {0};
  std::string m_outcome {"start"};
  std::vector<float> m_obs;
  Ptr<UniformRandomVariable> m_rng;
};

NS_OBJECT_ENSURE_REGISTERED (HoneycombGymEnv);

HoneycombGymEnv::HoneycombGymEnv ()
{
  m_obs.assign (STATE_DIM, 0.0f);
  m_rng = CreateObject<UniformRandomVariable> ();
}

HoneycombGymEnv::~HoneycombGymEnv ()
{
}

TypeId
HoneycombGymEnv::GetTypeId ()
{
  static TypeId tid = TypeId ("HoneycombGymEnv")
    .SetParent<OpenGymEnv> ()
    .SetGroupName ("OpenGym")
    .AddConstructor<HoneycombGymEnv> ();
  return tid;
}

void
HoneycombGymEnv::DoDispose ()
{
  OpenGymEnv::DoDispose ();
}

void
HoneycombGymEnv::Configure (uint32_t nSensors, uint32_t seed)
{
  m_nSensors = std::min<uint32_t> (std::max<uint32_t> (nSensors, 1), 100);
  m_nSinks = 4;
  m_rng->SetStream (seed + 3);

  const uint32_t n = m_nSensors + m_nSinks;
  m_nodes.assign (n, Node ());
  for (uint32_t i = 0; i < m_nSensors; ++i)
    {
      m_nodes[i].x = g_sensorPos[i][0];
      m_nodes[i].y = g_sensorPos[i][1];
    }
  for (uint32_t k = 0; k < m_nSinks; ++k)
    {
      Node &s = m_nodes[m_nSensors + k];
      s.x = g_sinkPos[k][0];
      s.y = g_sinkPos[k][1];
      s.sink = true;
      s.trust = 1.0;
      s.energy = 1.0;
    }

  uint32_t nMal = std::max<uint32_t> (1, m_nSensors / 10);
  std::vector<uint32_t> order (m_nSensors);
  for (uint32_t i = 0; i < m_nSensors; ++i)
    order[i] = i;
  for (uint32_t i = m_nSensors - 1; i > 0; --i)
    {
      uint32_t j = m_rng->GetInteger (0, i);
      std::swap (order[i], order[j]);
    }
  for (uint32_t i = 0; i < nMal; ++i)
    m_nodes[order[i]].malicious = true;

  m_stability.assign (n, std::vector<double> (n, 1.0));
  for (uint32_t a = 0; a < n; ++a)
    for (uint32_t b = a + 1; b < n; ++b)
      {
        double s = STABILITY_MIN + (1.0 - STABILITY_MIN) * m_rng->GetValue ();
        m_stability[a][b] = m_stability[b][a] = s;
      }

  for (uint32_t i = 0; i < m_nSensors; ++i)
    {
      m_nodes[i].energy = ENERGY_MIN + (1.0 - ENERGY_MIN) * m_rng->GetValue ();
      double p = m_nodes[i].malicious ? P_FWD_MALICIOUS : P_FWD_NORMAL;
      int succ = 0;
      for (int t = 0; t < TRUST_OBS; ++t)
        if (m_rng->GetValue () < p)
          ++succ;
      m_nodes[i].trust = (succ + 1.0) / (TRUST_OBS + 2.0);
    }
}

double
HoneycombGymEnv::Dist (uint32_t a, uint32_t b) const
{
  double dx = m_nodes[a].x - m_nodes[b].x;
  double dy = m_nodes[a].y - m_nodes[b].y;
  return std::hypot (dx, dy);
}

double
HoneycombGymEnv::Prr (uint32_t a, uint32_t b) const
{
  return 1.0 / (1.0 + std::exp ((Dist (a, b) - PRR_D50) / PRR_WIDTH));
}

uint32_t
HoneycombGymEnv::NearestSink (uint32_t sensor) const
{
  uint32_t best = m_nSensors;
  double bestD = Dist (sensor, m_nSensors);
  for (uint32_t k = 1; k < m_nSinks; ++k)
    {
      double d = Dist (sensor, m_nSensors + k);
      if (d < bestD)
        {
          bestD = d;
          best = m_nSensors + k;
        }
    }
  return best;
}

std::vector<int>
HoneycombGymEnv::Candidates (uint32_t cur, uint32_t dest) const
{
  double dct = Dist (cur, dest);
  std::vector<int> out;
  for (uint32_t j = 0; j < m_nSensors; ++j)
    {
      if (j == cur)
        continue;
      if (Dist (cur, j) <= TX_RANGE && Dist (j, dest) < dct)
        out.push_back ((int) j);
    }
  if (Dist (cur, dest) <= TX_RANGE)
    out.push_back ((int) dest);
  std::sort (out.begin (), out.end (), [&] (int a, int b) {
    double da = Dist ((uint32_t) a, dest);
    double db = Dist ((uint32_t) b, dest);
    if (da != db)
      return da < db;
    return a < b;
  });
  if (out.size () > K_SLOTS)
    out.resize (K_SLOTS);
  return out;
}

void
HoneycombGymEnv::FillObservation (uint32_t cur, uint32_t dest)
{
  std::fill (m_obs.begin (), m_obs.end (), 0.0f);
  auto cand = Candidates (cur, dest);
  for (uint32_t k = 0; k < cand.size (); ++k)
    {
      uint32_t j = (uint32_t) cand[k];
      uint32_t base = k * SLOT_FEATURES;
      m_obs[base + 0] = 1.0f;
      m_obs[base + 1] = (float) m_nodes[j].trust;
      m_obs[base + 2] = (float) Prr (cur, j);
      m_obs[base + 3] = (float) m_stability[cur][j];
      m_obs[base + 4] = (float) m_nodes[j].energy;
      m_obs[base + 5] = (float) ((Dist (cur, dest) - Dist (j, dest)) / TX_RANGE);
      m_obs[base + 6] = j == dest ? 1.0f : 0.0f;
    }
  m_obs[K_SLOTS * SLOT_FEATURES] = (float) (Dist (cur, dest) / (4.0 * TX_RANGE));
  m_obs[K_SLOTS * SLOT_FEATURES + 1] = (float) m_nodes[cur].energy;
}

void
HoneycombGymEnv::BeginEpisode ()
{
  m_cur = m_rng->GetInteger (0, m_nSensors - 1);
  m_dest = NearestSink (m_cur);
  m_hops = 0;
  m_reward = 0.0f;
  m_outcome = "start";
  FillObservation (m_cur, m_dest);
  if (Candidates (m_cur, m_dest).empty ())
    {
      m_reward = (float) -R_FAIL;
      m_outcome = "dead_end";
      m_gameOver = true;
    }
}

void
HoneycombGymEnv::Start ()
{
  BeginEpisode ();
  // Tiap Notify mengirim state lalu menunggu aksi, kecuali pesan game-over:
  // Python membalas stop dan proses selesai di dalam Notify itu.
  Notify ();
  while (!m_gameOver)
    Notify ();
  if (m_outcome != "start")
    Notify ();
  Simulator::Stop ();
}

Ptr<OpenGymSpace>
HoneycombGymEnv::GetActionSpace ()
{
  return CreateObject<OpenGymDiscreteSpace> (K_SLOTS);
}

Ptr<OpenGymSpace>
HoneycombGymEnv::GetObservationSpace ()
{
  float low = -2.0f;
  float high = 2.0f;
  std::vector<uint32_t> shape = {STATE_DIM};
  std::string dtype = TypeNameGet<float> ();
  return CreateObject<OpenGymBoxSpace> (low, high, shape, dtype);
}

bool
HoneycombGymEnv::GetGameOver ()
{
  return m_gameOver;
}

Ptr<OpenGymDataContainer>
HoneycombGymEnv::GetObservation ()
{
  std::vector<uint32_t> shape = {STATE_DIM};
  Ptr<OpenGymBoxContainer<float>> box = CreateObject<OpenGymBoxContainer<float>> (shape);
  for (float v : m_obs)
    box->AddValue (v);
  return box;
}

float
HoneycombGymEnv::GetReward ()
{
  return m_reward;
}

std::string
HoneycombGymEnv::GetExtraInfo ()
{
  std::ostringstream os;
  os << "outcome=" << m_outcome
     << ";node=" << m_cur
     << ";dest=" << m_dest
     << ";hops=" << m_hops;
  return os.str();
}

bool
HoneycombGymEnv::ExecuteActions (Ptr<OpenGymDataContainer> action)
{
  if (m_gameOver)
    return true;

  Ptr<OpenGymDiscreteContainer> disc = DynamicCast<OpenGymDiscreteContainer> (action);
  if (!disc)
    {
      m_reward = (float) -R_FAIL;
      m_outcome = "bad_action_type";
      m_gameOver = true;
      return true;
    }
  uint32_t slot = disc->GetValue ();
  auto cand = Candidates (m_cur, m_dest);
  if (slot >= K_SLOTS || slot >= cand.size ())
    {
      m_reward = (float) -R_FAIL;
      m_outcome = "invalid_slot";
      m_gameOver = true;
      return true;
    }

  uint32_t next = (uint32_t) cand[slot];
  ++m_hops;
  double pass = Prr (m_cur, next) * m_stability[m_cur][next];
  if (m_rng->GetValue () >= pass)
    {
      m_reward = (float) -R_FAIL;
      m_outcome = "link_loss";
      m_gameOver = true;
      FillObservation (m_cur, m_dest);
      return true;
    }
  if (next == m_dest)
    {
      m_cur = next;
      m_reward = (float) R_SUCCESS;
      m_outcome = "delivered";
      m_gameOver = true;
      FillObservation (m_cur, m_dest);
      return true;
    }
  double pFwd = m_nodes[next].malicious ? P_FWD_MALICIOUS : P_FWD_NORMAL;
  if (m_rng->GetValue () >= pFwd)
    {
      m_reward = (float) -R_FAIL;
      m_outcome = m_nodes[next].malicious ? "malicious_drop" : "relay_drop";
      m_gameOver = true;
      FillObservation (m_cur, m_dest);
      return true;
    }
  m_cur = next;
  m_reward = (float) (-C_HOP - C_ENERGY * (1.0 - m_nodes[next].energy));
  if (m_hops >= MAX_HOPS)
    {
      m_outcome = "ttl";
      m_reward = (float) -R_FAIL;
      m_gameOver = true;
    }
  else
    m_outcome = "forwarded";
  FillObservation (m_cur, m_dest);
  return true;
}

int
main (int argc, char *argv[])
{
  uint32_t numNodes = 20;
  uint32_t seed = 1;
  uint32_t openGymPort = 5555;

  CommandLine cmd;
  cmd.AddValue ("numNodes", "jumlah sensor (1..100)", numNodes);
  cmd.AddValue ("seed", "RNG seed", seed);
  cmd.AddValue ("openGymPort", "port ZMQ ns3-gym", openGymPort);
  cmd.Parse (argc, argv);

  RngSeedManager::SetSeed (seed);

  Ptr<HoneycombGymEnv> env = CreateObject<HoneycombGymEnv> ();
  env->Configure (numNodes, seed);

  NodeContainer sensors;
  sensors.Create (numNodes);
  NodeContainer sinks;
  sinks.Create (4);
  MobilityHelper mob;
  mob.SetMobilityModel ("ns3::ConstantPositionMobilityModel");
  mob.Install (sensors);
  mob.Install (sinks);
  for (uint32_t i = 0; i < numNodes; ++i)
    {
      Ptr<MobilityModel> m = sensors.Get (i)->GetObject<MobilityModel> ();
      m->SetPosition (Vector (g_sensorPos[i][0], g_sensorPos[i][1], 0));
    }
  for (uint32_t k = 0; k < 4; ++k)
    {
      Ptr<MobilityModel> m = sinks.Get (k)->GetObject<MobilityModel> ();
      m->SetPosition (Vector (g_sinkPos[k][0], g_sinkPos[k][1], 0));
    }

  Ptr<OpenGymInterface> iface = CreateObject<OpenGymInterface> (openGymPort);
  env->SetOpenGymInterface (iface);

  NS_LOG_UNCOND ("honeycomb-sdwsn-gym: " << numNodes
                 << " sensor, port " << openGymPort
                 << ", satu episode paket");
  Simulator::ScheduleNow (&HoneycombGymEnv::Start, env);
  Simulator::Run ();
  Simulator::Destroy ();
  return 0;
}
