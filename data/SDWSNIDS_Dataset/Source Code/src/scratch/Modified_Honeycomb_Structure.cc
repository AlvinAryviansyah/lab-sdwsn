#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/mobility-module.h"
#include "ns3/config-store-module.h"
#include "ns3/wifi-module.h"
#include "ns3/internet-module.h"
#include "ns3/ipv4-static-routing-helper.h"
#include "ns3/ipv4-list-routing-helper.h"
#include "ns3/yans-wifi-phy.h"
#include "ns3/netanim-module.h"
#include <vector>
#include "ns3/flow-monitor-module.h"
#include <cassert>
#include "ns3/ipv6-static-routing-helper.h"
#include "ns3/ipv6-routing-table-entry.h"
#include "ns3/openflow-module.h"
#include "ns3/sdn.h"
#include "ns3/iot-module.h"
#include "ns3/applications-module.h"
#include "ns3/log.h"
#include <iostream>
#include <fstream>
#include <string>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <sstream>  
#include <bits/stdc++.h>
using namespace ns3; using namespace std;
NS_LOG_COMPONENT_DEFINE ("Modified_Honeycomb_Structure");
AnimationInterface *pAnim; double ds=1000.0; Ipv6InterfaceContainer ic; uint16_t Port = 8000; std::string dataRate("250Kbps");
uint16_t numNodes = 100; uint16_t numUsers = 10; uint16_t numswitch = 4; int numSensors = 7; uint16_t numSink = 4;
uint32_t packetSize = 1024; uint32_t numPackets = 100; int flows=50; int Attackers=10; double interval = 1.0; 
Time interPacketInterval = Seconds (interval);
void ReceivePacket (Ptr<Socket> socket){ while (socket->Recv ()){NS_LOG_UNCOND ("Packet Recived.");}}
static void GenerateTraffic (Ptr<Socket> socket, uint32_t pktSize,uint32_t pktCount, Time pktInterval ){
if (pktCount > 0){ socket->Send (Create<Packet> (pktSize));
Simulator::Schedule (pktInterval, &GenerateTraffic,socket, pktSize,pktCount-1, pktInterval);} else{socket->Close ();}}
void PktTrans1(NodeContainer c, NodeContainer d){
std::cout<<"\n  The sensor nodes are clustered based on hexagon structure based on 1/7 fraction which means each cluster includes seven hexagon cells in the honeycomb. "<<"\n\n";
hexagon obj1; obj1.Clustering(7);
std::cout<<"\n After completing clustering Cluster Head (CH) is selected based on node centrality, distance, energy, and trust. \n\n";        
int Node_Centrality=20;int distance=10;int energy=47;int trust=63;int network=78;int cells=91;double srcport=80; int destport=66;
obj1.flow_features(Node_Centrality,distance,energy,trust,network,cells,&srcport,&destport,numSensors);
for( uint32_t i=0;i<c.GetN ();i++){ TypeId tid1 = TypeId::LookupByName ("ns3::UdpSocketFactory");
Ptr<Socket> recvSink1 = Socket::CreateSocket (d.Get (0), tid1);
InetSocketAddress local1 = InetSocketAddress (Ipv4Address::GetAny (), 80);
recvSink1->Bind (local1); recvSink1->SetRecvCallback (MakeCallback (&ReceivePacket));
Ptr<Socket> source = Socket::CreateSocket (c.Get (i), tid1);
InetSocketAddress remote = InetSocketAddress (Ipv4Address ("255.255.255.255"), 80);
source->SetAllowBroadcast (true); source->Connect (remote);
Simulator::ScheduleWithContext (source->GetNode ()->GetId (),Seconds (0.1), &GenerateTraffic,source, packetSize, numPackets,interPacketInterval);} for(  uint32_t i=0;i<d.GetN ();i++){ if((i%2)==0){
Ipv6Address Ip = ic.GetLinkLocalAddress (i);
AddressValue Addr (Inet6SocketAddress (Ip, Port)); OnOffHelper onoff ("ns3::TcpSocketFactory", Address ());
onoff.SetAttribute ("Remote", Addr);
onoff.SetAttribute ("OnTime", StringValue ("ns3::ConstantRandomVariable[Constant=1]"));
onoff.SetAttribute ("OffTime", StringValue ("ns3::ConstantRandomVariable[Constant=0]"));
onoff.SetAttribute ("DataRate", DataRateValue (dataRate));
onoff.SetAttribute ("PacketSize", UintegerValue (packetSize));
ApplicationContainer IoTAPP = onoff.Install (d); IoTAPP.Start (Seconds (1.0)); IoTAPP.Stop (Seconds (50.0));}}}
void PktTrans2(NodeContainer c, NodeContainer d){
std::cout<<"\n Select the route between CH and Cluster Member (CM) for data transmission by using Transfer Learning based Deep Q Network (TLDQN). \n\n"; TLDQN obj2; obj2.TLDQN_algo(); obj2.TLDQN_Route_selection(numNodes,numNodes);
for(  uint32_t i=0;i<c.GetN ();i++){ TypeId tid1 = TypeId::LookupByName ("ns3::UdpSocketFactory");
Ptr<Socket> recvSink1 = Socket::CreateSocket (d.Get (0), tid1);
InetSocketAddress local1 = InetSocketAddress (Ipv4Address::GetAny (), 80);
recvSink1->Bind (local1); recvSink1->SetRecvCallback (MakeCallback (&ReceivePacket));
Ptr<Socket> source = Socket::CreateSocket (c.Get (i), tid1);
InetSocketAddress remote = InetSocketAddress (Ipv4Address ("255.255.255.255"), 80);
source->SetAllowBroadcast (true); source->Connect (remote);
Simulator::ScheduleWithContext (source->GetNode ()->GetId (),Seconds (0.1), &GenerateTraffic,source, packetSize, numPackets,interPacketInterval);}
std::cout<<"\n Perform hybrid intrusion detection which includes two stages of intrusion detection, like, Malicious Node Detection by using using Driver Training based Optimization (DTO) algorithm and Intrusion Detection by using Bidirectional GAN (Bi-GAN) which classified the packets into normal or malicious. \n\n";
BiGAN obj; for(  uint32_t i=0;i<c.GetN ();i++){std::string res=obj.detect((int)i);cout << res;} DTO obj1; std::string packetstatus= " ";
for(  uint32_t i=0;i<c.GetN ();i++){ packetstatus=obj1.classify(i); std::cout<<packetstatus<<" ";
TypeId tid1 = TypeId::LookupByName ("ns3::UdpSocketFactory");
Ptr<Socket> recvSink1 = Socket::CreateSocket (d.Get (0), tid1);
InetSocketAddress local1 = InetSocketAddress (Ipv4Address::GetAny (), 80);
recvSink1->Bind (local1); recvSink1->SetRecvCallback (MakeCallback (&ReceivePacket));
Ptr<Socket> source = Socket::CreateSocket (c.Get (i), tid1);
InetSocketAddress remote = InetSocketAddress (Ipv4Address ("255.255.255.255"), 80);
source->SetAllowBroadcast (true); source->Connect (remote);
Simulator::ScheduleWithContext (source->GetNode ()->GetId (),Seconds (0.1), &GenerateTraffic,source, packetSize, numPackets,interPacketInterval);} for(  uint32_t i=0;i<d.GetN ();i++){ if((i%2)==0){
Ipv6Address Ip = ic.GetLinkLocalAddress (i); AddressValue Addr (Inet6SocketAddress (Ip, Port));
OnOffHelper onoff ("ns3::TcpSocketFactory", Address ()); onoff.SetAttribute ("Remote", Addr);
onoff.SetAttribute ("OnTime", StringValue ("ns3::ConstantRandomVariable[Constant=1]"));
onoff.SetAttribute ("OffTime", StringValue ("ns3::ConstantRandomVariable[Constant=0]"));
onoff.SetAttribute ("DataRate", DataRateValue (dataRate)); onoff.SetAttribute ("PacketSize", UintegerValue (packetSize));
ApplicationContainer IoTAPP = onoff.Install (d); IoTAPP.Start (Seconds (1.0)); IoTAPP.Stop (Seconds (50.0));}}}
void PktTrans3(NodeContainer c, NodeContainer d){std::cout<<"\n  Select an optimal delegator to send the warning alarm to the network.  \n\n"; 
if(numSensors > 0){ std::cout<<" Optimal delegator to send the warning alarm to the network! ";}
for(  uint32_t i=1;i<c.GetN ();i++){ TypeId tid1 = TypeId::LookupByName ("ns3::UdpSocketFactory");
Ptr<Socket> recvSink1 = Socket::CreateSocket (d.Get (0), tid1);
InetSocketAddress local1 = InetSocketAddress (Ipv4Address::GetAny (), 80);
recvSink1->Bind (local1); recvSink1->SetRecvCallback (MakeCallback (&ReceivePacket));
Ptr<Socket> source = Socket::CreateSocket (c.Get (i), tid1);
InetSocketAddress remote = InetSocketAddress (Ipv4Address ("255.255.255.255"), 80);
source->SetAllowBroadcast (true); source->Connect (remote);
Simulator::ScheduleWithContext (source->GetNode ()->GetId (),Seconds (0.1), &GenerateTraffic,source, packetSize, numPackets,interPacketInterval);} for(  uint32_t i=0;i<d.GetN ();i++){ if((i%2)==0){
Ipv6Address Ip = ic.GetLinkLocalAddress (i); AddressValue Addr (Inet6SocketAddress (Ip, Port));
OnOffHelper onoff ("ns3::TcpSocketFactory", Address ()); onoff.SetAttribute ("Remote", Addr);
onoff.SetAttribute ("OnTime", StringValue ("ns3::ConstantRandomVariable[Constant=1]"));
onoff.SetAttribute ("OffTime", StringValue ("ns3::ConstantRandomVariable[Constant=0]"));
onoff.SetAttribute ("DataRate", DataRateValue (dataRate)); onoff.SetAttribute ("PacketSize", UintegerValue (packetSize));
ApplicationContainer IoTAPP = onoff.Install (d); IoTAPP.Start (Seconds (1.0)); IoTAPP.Stop (Seconds (50.0));}} std::cout<<" \n";}
int main (int argc, char *argv[]){ std::string phyMode ("OfdmRate6MbpsBW10MHz"); double distance = 600;  uint32_t revNode = 0;
uint32_t sourceNode = numNodes-1; int nodeSpeed = 25; int nodePause = 0; bool enableFlowMonitor = false; CommandLine cmd;
double simtime=300.0; cmd.AddValue ("phyMode", "Wifi Phy mode", phyMode);
cmd.AddValue ("distance", "distance (m)", distance); cmd.AddValue ("packetSize", "size of application packet sent", packetSize);
cmd.AddValue ("numPackets", "number of packets generated", numPackets); 
cmd.AddValue ("interval", "interval (seconds) between packets", interval);
cmd.AddValue ("numNodes", "number of nodes", numNodes); cmd.AddValue ("numUsers", "number of users", numUsers);
cmd.AddValue ("revNode", "Receiver node number", revNode); cmd.AddValue ("sourceNode", "Sender node number", sourceNode);
cmd.AddValue ("EnableMonitor", "Enable Flow Monitor", enableFlowMonitor); cmd.Parse (argc, argv); 
NodeContainer SensorNodes; NodeContainer UserNodes; NodeContainer BlockchainNode; NodeContainer TA; NodeContainer SwitchNodes;
NodeContainer EdgeAssistedSinkNodes; NodeContainer MasterControllerNodes; NodeContainer SlaveControllerNodes; NodeContainer CloudNode;
std::cout<<"\n\n A Network , it consists of  100- SensorNodes , 1-Trusted Authority (TA), 10-IoT Users, 4- Edge Assisted Sinks, 4- switches, 1- Master Controller, 2- Slave Controllers, 3- Blockchain node and 1- Cloud node. \n\n";
SensorNodes.Create (numNodes); TA.Create (1); UserNodes.Create (numUsers); BlockchainNode.Create (3); SwitchNodes.Create (numswitch);
EdgeAssistedSinkNodes.Create (numSink); MasterControllerNodes.Create (1); SlaveControllerNodes.Create (2); CloudNode.Create (1);
WifiHelper wifi; Ptr<Ipv6ExtensionESP > extension; Ptr<Ipv6ExtensionAH> extenAH;
YansWifiPhyHelper wifiPhy =  YansWifiPhyHelper::Default ();   wifiPhy.Set ("RxGain", DoubleValue (-30)); 
wifiPhy.SetPcapDataLinkType (YansWifiPhyHelper::DLT_IEEE802_11_RADIO); 
YansWifiChannelHelper wifiChannel; wifiChannel.SetPropagationDelay ("ns3::ConstantSpeedPropagationDelayModel");
wifiChannel.AddPropagationLoss ("ns3::NakagamiPropagationLossModel");
wifiPhy.SetChannel (wifiChannel.Create ()); NqosWifiMacHelper wifiMac = NqosWifiMacHelper::Default ();
wifi.SetStandard (WIFI_PHY_STANDARD_80211a); 
wifi.SetRemoteStationManager ("ns3::ConstantRateWifiManager","DataMode",StringValue (phyMode),"ControlMode",StringValue (phyMode)); 
wifiMac.SetType ("ns3::AdhocWifiMac");
NetDeviceContainer IoTdevices = wifi.Install (wifiPhy, wifiMac, SensorNodes); 
NetDeviceContainer userdevices = wifi.Install (wifiPhy, wifiMac, UserNodes);
NetDeviceContainer edgedevices = wifi.Install (wifiPhy, wifiMac, EdgeAssistedSinkNodes);
NetDeviceContainer switchdevices = wifi.Install (wifiPhy, wifiMac, SwitchNodes);
NetDeviceContainer clouddevices = wifi.Install (wifiPhy, wifiMac, CloudNode);
NetDeviceContainer slavecontrollerdevices = wifi.Install (wifiPhy, wifiMac, SlaveControllerNodes);
NetDeviceContainer sdnControllerDevices; sdnControllerDevices = wifi.Install (wifiPhy, wifiMac, MasterControllerNodes);
int64_t streamIndex = 0; ObjectFactory pos;
pos.SetTypeId ("ns3::RandomRectanglePositionAllocator");
pos.Set ("X", StringValue ("ns3::UniformRandomVariable[Min=1.0|Max=1000.0]"));
pos.Set ("Y", StringValue ("ns3::UniformRandomVariable[Min=1.0|Max=1000.0]"));
Ptr<PositionAllocator> taPositionAlloc = pos.Create ()->GetObject<PositionAllocator> ();
streamIndex += taPositionAlloc->AssignStreams (streamIndex);
MobilityHelper mobility; mobility.SetPositionAllocator(taPositionAlloc);
std::stringstream ssSpeed; ssSpeed << "ns3::UniformRandomVariable[Min=0.0|Max=" << nodeSpeed << "]";
std::stringstream ssPause; ssPause << "ns3::ConstantRandomVariable[Constant=" << nodePause << "]";
mobility.SetMobilityModel ("ns3::RandomWaypointMobilityModel","Speed", StringValue (ssSpeed.str ()),"Pause", StringValue (ssPause.str ()),
"PositionAllocator", PointerValue (taPositionAlloc)); mobility.Install (UserNodes); MobilityHelper mobility1;
mobility1.SetMobilityModel ("ns3::ConstantPositionMobilityModel");
mobility1.Install (SensorNodes); mobility1.Install (SwitchNodes); mobility1.Install (EdgeAssistedSinkNodes);
mobility1.Install (BlockchainNode); mobility1.Install (TA); mobility1.Install (MasterControllerNodes); 
mobility1.Install (SlaveControllerNodes); mobility1.Install (CloudNode);
AnimationInterface::SetConstantPosition (BlockchainNode.Get (0), 200, 8);
AnimationInterface::SetConstantPosition (BlockchainNode.Get (1), 400, 8);
AnimationInterface::SetConstantPosition (BlockchainNode.Get (2), 800, 8);
AnimationInterface::SetConstantPosition (CloudNode.Get (0), 600, 8);AnimationInterface::SetConstantPosition (TA.Get (0), 450, 450);
AnimationInterface::SetConstantPosition (MasterControllerNodes.Get (0), 450, 100.0);
AnimationInterface::SetConstantPosition (SlaveControllerNodes.Get (0), 350, 350);
AnimationInterface::SetConstantPosition (SlaveControllerNodes.Get (1), 600, 350);
AnimationInterface::SetConstantPosition (SwitchNodes.Get (0), 225, 225);
AnimationInterface::SetConstantPosition (SwitchNodes.Get (1), 225, 675);
AnimationInterface::SetConstantPosition (SwitchNodes.Get (2), 675, 225);
AnimationInterface::SetConstantPosition (SwitchNodes.Get (3), 675, 675);
Simulator::Schedule (Seconds (1.3), &PktTrans1,SensorNodes, SwitchNodes);
Simulator::Schedule (Seconds (8.3), &PktTrans2,SensorNodes, SwitchNodes);
Ipv4AddressHelper ipv4; Ipv4StaticRoutingHelper staticRouting; Ipv4ListRoutingHelper list; list.Add (staticRouting, 0);
InternetStackHelper internet; internet.SetRoutingHelper (list); internet.Install (SensorNodes); internet.Install (UserNodes);
internet.Install (SwitchNodes); internet.Install (EdgeAssistedSinkNodes); internet.Install (BlockchainNode);
internet.Install (TA); internet.Install (MasterControllerNodes); internet.Install (SlaveControllerNodes);
internet.Install (CloudNode); OpenflowHelper openflow; OpenflowHc1 openflowheaderobj2; 
openflow.SetDeviceAttribute ("ForceEtherType", BooleanValue (true) );
openflowheaderobj2.SetHopLimit (100); NetDeviceContainer openflowobj = openflow.Install (sdnControllerDevices);
openflowheaderobj2.SetHopLimit (50); openflowobj = openflow.Install (edgedevices);
openflowobj = openflow.Install (switchdevices); openflowobj = openflow.Install (slavecontrollerdevices);
int64_t res= openflow.AssignStreams (sdnControllerDevices,numNodes,flows,numPackets,Attackers,0); std::cout<<res;
AnimationInterface::SetConstantPosition (SensorNodes.Get (0), 600, 600);
AnimationInterface::SetConstantPosition (SensorNodes.Get (1), 550, 550);
AnimationInterface::SetConstantPosition (SensorNodes.Get (2), 550, 500);
AnimationInterface::SetConstantPosition (SensorNodes.Get (3), 600, 450);
AnimationInterface::SetConstantPosition (SensorNodes.Get (4), 650, 550);
AnimationInterface::SetConstantPosition (SensorNodes.Get (5), 650, 500);
AnimationInterface::SetConstantPosition (SensorNodes.Get (6), 500, 600);
AnimationInterface::SetConstantPosition (SensorNodes.Get (7), 100, 600);
AnimationInterface::SetConstantPosition (SensorNodes.Get (8), 450, 500);
AnimationInterface::SetConstantPosition (SensorNodes.Get (9), 500, 450);
AnimationInterface::SetConstantPosition (SensorNodes.Get (10), 400, 600);
AnimationInterface::SetConstantPosition (SensorNodes.Get (11), 350, 550);
AnimationInterface::SetConstantPosition (SensorNodes.Get (12), 350, 500);
AnimationInterface::SetConstantPosition (SensorNodes.Get (13), 400, 450);
AnimationInterface::SetConstantPosition (SensorNodes.Get (14), 300, 600);
AnimationInterface::SetConstantPosition (SensorNodes.Get (15), 250, 550);
AnimationInterface::SetConstantPosition (SensorNodes.Get (16), 250, 500);
AnimationInterface::SetConstantPosition (SensorNodes.Get (17), 300, 450);
AnimationInterface::SetConstantPosition (SensorNodes.Get (18), 200, 600);
AnimationInterface::SetConstantPosition (SensorNodes.Get (19), 150, 550);
AnimationInterface::SetConstantPosition (SensorNodes.Get (20), 150, 500);
AnimationInterface::SetConstantPosition (SensorNodes.Get (21), 200, 450);
AnimationInterface::SetConstantPosition (SensorNodes.Get (22), 700, 600);
AnimationInterface::SetConstantPosition (SensorNodes.Get (23), 700, 450);
AnimationInterface::SetConstantPosition (SensorNodes.Get (24), 750, 550);
AnimationInterface::SetConstantPosition (SensorNodes.Get (25), 750, 500);
AnimationInterface::SetConstantPosition (SensorNodes.Get (26), 800, 600);
AnimationInterface::SetConstantPosition (SensorNodes.Get (27), 800, 450);
AnimationInterface::SetConstantPosition (SensorNodes.Get (28), 850, 550);
AnimationInterface::SetConstantPosition (SensorNodes.Get (29), 850, 500);
AnimationInterface::SetConstantPosition (SensorNodes.Get (30), 150, 650);
AnimationInterface::SetConstantPosition (SensorNodes.Get (31), 150, 700);
AnimationInterface::SetConstantPosition (SensorNodes.Get (32), 250, 650);
AnimationInterface::SetConstantPosition (SensorNodes.Get (33), 250, 700);
AnimationInterface::SetConstantPosition (SensorNodes.Get (34), 350, 650);
AnimationInterface::SetConstantPosition (SensorNodes.Get (35), 350, 700);
AnimationInterface::SetConstantPosition (SensorNodes.Get (36), 450, 650);
AnimationInterface::SetConstantPosition (SensorNodes.Get (37), 450, 700);
AnimationInterface::SetConstantPosition (SensorNodes.Get (38), 550, 650);
AnimationInterface::SetConstantPosition (SensorNodes.Get (39), 550, 700);
AnimationInterface::SetConstantPosition (SensorNodes.Get (40), 650, 650);
AnimationInterface::SetConstantPosition (SensorNodes.Get (41), 650, 700);
AnimationInterface::SetConstantPosition (SensorNodes.Get (42), 750, 650);
AnimationInterface::SetConstantPosition (SensorNodes.Get (43), 750, 700);
AnimationInterface::SetConstantPosition (SensorNodes.Get (44), 850, 650);
AnimationInterface::SetConstantPosition (SensorNodes.Get (45), 850, 700);
AnimationInterface::SetConstantPosition (SensorNodes.Get (46), 200, 750);
AnimationInterface::SetConstantPosition (SensorNodes.Get (47), 600, 750);
AnimationInterface::SetConstantPosition (SensorNodes.Get (48), 500, 750);
AnimationInterface::SetConstantPosition (SensorNodes.Get (49), 400, 750);
AnimationInterface::SetConstantPosition (SensorNodes.Get (50), 300, 750);
AnimationInterface::SetConstantPosition (SensorNodes.Get (51), 700, 750);
AnimationInterface::SetConstantPosition (SensorNodes.Get (52), 800, 750);
AnimationInterface::SetConstantPosition (SensorNodes.Get (53), 150, 400);
AnimationInterface::SetConstantPosition (SensorNodes.Get (54), 150, 350);
AnimationInterface::SetConstantPosition (SensorNodes.Get (55), 100, 450);
AnimationInterface::SetConstantPosition (SensorNodes.Get (56), 250, 350);
AnimationInterface::SetConstantPosition (SensorNodes.Get (57), 450, 400);
AnimationInterface::SetConstantPosition (SensorNodes.Get (58), 450, 350);
AnimationInterface::SetConstantPosition (SensorNodes.Get (59), 550, 400);
AnimationInterface::SetConstantPosition (SensorNodes.Get (60), 550, 350);
AnimationInterface::SetConstantPosition (SensorNodes.Get (61), 900, 300);
AnimationInterface::SetConstantPosition (SensorNodes.Get (62), 650, 350);
AnimationInterface::SetConstantPosition (SensorNodes.Get (63), 750, 400);
AnimationInterface::SetConstantPosition (SensorNodes.Get (64), 750, 350);
AnimationInterface::SetConstantPosition (SensorNodes.Get (65), 850, 400);
AnimationInterface::SetConstantPosition (SensorNodes.Get (66), 850, 350);
AnimationInterface::SetConstantPosition (SensorNodes.Get (67), 350, 400);
AnimationInterface::SetConstantPosition (SensorNodes.Get (68), 350, 350);
AnimationInterface::SetConstantPosition (SensorNodes.Get (69), 200, 300);
AnimationInterface::SetConstantPosition (SensorNodes.Get (70), 600, 300);
AnimationInterface::SetConstantPosition (SensorNodes.Get (71), 500, 300);
AnimationInterface::SetConstantPosition (SensorNodes.Get (72), 400, 300);
AnimationInterface::SetConstantPosition (SensorNodes.Get (73), 300, 300);
AnimationInterface::SetConstantPosition (SensorNodes.Get (74), 700, 300);
AnimationInterface::SetConstantPosition (SensorNodes.Get (75), 800, 300);
AnimationInterface::SetConstantPosition (SensorNodes.Get (76), 150, 200);
AnimationInterface::SetConstantPosition (SensorNodes.Get (77), 150, 250);
AnimationInterface::SetConstantPosition (SensorNodes.Get (78), 250, 200);
AnimationInterface::SetConstantPosition (SensorNodes.Get (79), 250, 250);
AnimationInterface::SetConstantPosition (SensorNodes.Get (80), 350, 200);
AnimationInterface::SetConstantPosition (SensorNodes.Get (81), 350, 250);
AnimationInterface::SetConstantPosition (SensorNodes.Get (82), 450, 200);
AnimationInterface::SetConstantPosition (SensorNodes.Get (83), 100, 300);
AnimationInterface::SetConstantPosition (SensorNodes.Get (84), 550, 200);
AnimationInterface::SetConstantPosition (SensorNodes.Get (85), 550, 250);
AnimationInterface::SetConstantPosition (SensorNodes.Get (86), 650, 200);
AnimationInterface::SetConstantPosition (SensorNodes.Get (87), 650, 250);
AnimationInterface::SetConstantPosition (SensorNodes.Get (88), 750, 200);
AnimationInterface::SetConstantPosition (SensorNodes.Get (89), 750, 250);
AnimationInterface::SetConstantPosition (SensorNodes.Get (90), 850, 200);
AnimationInterface::SetConstantPosition (SensorNodes.Get (91), 850, 250);
AnimationInterface::SetConstantPosition (SensorNodes.Get (92), 200, 150);
AnimationInterface::SetConstantPosition (SensorNodes.Get (93), 600, 150);
AnimationInterface::SetConstantPosition (SensorNodes.Get (94), 500, 150);
AnimationInterface::SetConstantPosition (SensorNodes.Get (95), 400, 150);
AnimationInterface::SetConstantPosition (SensorNodes.Get (96), 300, 150);
AnimationInterface::SetConstantPosition (SensorNodes.Get (97), 700, 150);
AnimationInterface::SetConstantPosition (SensorNodes.Get (98), 800, 150);
AnimationInterface::SetConstantPosition (SensorNodes.Get (99), 900, 450);
AnimationInterface::SetConstantPosition (EdgeAssistedSinkNodes.Get (0), 250, 450);
AnimationInterface::SetConstantPosition (EdgeAssistedSinkNodes.Get (1), 450, 250);
AnimationInterface::SetConstantPosition (EdgeAssistedSinkNodes.Get (2), 450, 700);
AnimationInterface::SetConstantPosition (EdgeAssistedSinkNodes.Get (3), 700, 450);
IoTHelper iot; iot.SetDeviceAttribute ("ForceEtherType", BooleanValue (true) );
NetDeviceContainer iotdev = iot.Install (IoTdevices); NetDeviceContainer iotUser = iot.Install (userdevices);
InternetStackHelper internetv6; internetv6.SetIpv4StackInstall (false); NS_LOG_INFO ("Assign IP Addresses.");
ipv4.SetBase ("10.1.1.0", "255.255.255.0"); Ipv4InterfaceContainer i = ipv4.Assign (IoTdevices);
Ipv6Address ipv6AddressBase = Ipv6Address("2001::"); Ipv6Prefix ipv6AddressPrefix = Ipv6Prefix(64);
Ipv6AddressHelper address6; Ipv6AddressGenerator::Init (ipv6AddressBase, ipv6AddressPrefix);
std::ostringstream subnet; subnet << "10.1.0.0";
address6.SetBase (subnet.str ().c_str (), "255.255.255.0", "0.0.0.3"); ic=address6.Assign(IoTdevices);
SDN sdnobj; sdnobj.getAllRegisterDevices(SensorNodes,IoTdevices); sdnobj.getAllRegisterUsers(UserNodes,userdevices);
std::cout<<"\n\nThe TA provides a private key with the aid of Three-Dimensional Cube (3D Cube) algorithm and performs the authentication.\n\n";
Three_Dimensional_Cube obj; std::ofstream ofs ("PrivatekeyRegDevce.txt", std::ofstream::out);
for (uint16_t j = 0; j < SensorNodes.GetN(); j++){ std::cout<<" IoTdevice - "<<j+1 <<" Private Key = "<< obj.Privatekey(j+2) <<"\n"; 
ofs <<" IoTdevice - "<<j+1 <<" Private Key = "<< obj.Privatekey(j+2) <<"\n"; }
std::ofstream ofs1 ("PrivatekeyRegUser.txt", std::ofstream::out); std::cout<<" \n\n\n";
for (uint16_t j = 0; j < UserNodes.GetN(); j++){ std::cout<<" User - "<<j+1 <<" Private Key = "<< obj.Privatekey(j+7) <<"\n"; 
ofs1 <<" User - "<<j+1 <<" Private Key = "<< obj.Privatekey(j+7) <<"\n"; }
std::cout<<"\n\n\n"; std::cout<<" Attack detection is performed "<<"\n"; std::cout<<" The private key is stored in blockchain "<<"\n\n";
TypeId tid = TypeId::LookupByName ("ns3::UdpSocketFactory"); Ptr<Socket> recvSink = Socket::CreateSocket (SensorNodes.Get (revNode), tid);
InetSocketAddress local = InetSocketAddress (Ipv4Address::GetAny (), 80);
Simulator::Schedule (Seconds (15.3), &PktTrans3, SwitchNodes,BlockchainNode);
recvSink->Bind (local); recvSink->SetRecvCallback (MakeCallback (&ReceivePacket));
Ptr<Socket> source = Socket::CreateSocket (SensorNodes.Get (sourceNode), tid);
InetSocketAddress remote = InetSocketAddress (i.GetAddress (revNode, 0), 80);
source->Connect (remote); Simulator::Schedule (Seconds (1.3), &GenerateTraffic, source, packetSize, numPackets, interPacketInterval);
Simulator::Stop (Seconds (simtime)); pAnim= new AnimationInterface ("Modified_Honeycomb_Structure.xml");
pAnim->SetBackgroundImage ("/home/research/ns-allinone-3.26/netanim-3.107/img1/bg.png", -955, -555, 1.2500, 2.00, 1.0);
uint32_t Sensorimg =pAnim->AddResource("/home/research/ns-allinone-3.26/netanim-3.107/img1/IoTdevice.png");
uint32_t Userimg =pAnim->AddResource("/home/research/ns-allinone-3.26/netanim-3.107/img1/User.png");
uint32_t BlockchainNodeimg =pAnim->AddResource("/home/research/ns-allinone-3.26/netanim-3.107/img1/Blockchain.png");
uint32_t MasterControllerNodesimg =pAnim->AddResource("/home/research/ns-allinone-3.26/netanim-3.107/img1/MasterController.png");
uint32_t EdgeNodesimg =pAnim->AddResource("/home/research/ns-allinone-3.26/netanim-3.107/img1/Edge.png");
uint32_t SwitchNodesimg =pAnim->AddResource("/home/research/ns-allinone-3.26/netanim-3.107/img1/Switch.png");
uint32_t SlaveControllerimg =pAnim->AddResource("/home/research/ns-allinone-3.26/netanim-3.107/img1/SlaveController.png");
uint32_t Cloudimg =pAnim->AddResource("/home/research/ns-allinone-3.26/netanim-3.107/img1/Cloud.png");
uint32_t TAimg =pAnim->AddResource("/home/research/ns-allinone-3.26/netanim-3.107/img1/TA.png");
for(  uint32_t i=0;i<SensorNodes.GetN ();i++){ pAnim->UpdateNodeDescription (SensorNodes.Get (i), "Sensor"); 
Ptr<Node> wid= SensorNodes.Get (i); uint32_t nodeId = wid->GetId (); pAnim->UpdateNodeImage (nodeId, Sensorimg);
pAnim->UpdateNodeColor(SensorNodes.Get(i), 255, 255, 0); pAnim->UpdateNodeSize (nodeId, 50.0,50.0);}
for(  uint32_t i=0;i<UserNodes.GetN ();i++){ pAnim->UpdateNodeDescription (UserNodes.Get (i), "User"); 
Ptr<Node> wid= UserNodes.Get (i); uint32_t nodeId = wid->GetId (); pAnim->UpdateNodeImage (nodeId, Userimg);
pAnim->UpdateNodeColor(UserNodes.Get(i), 255, 255, 0); pAnim->UpdateNodeSize (nodeId, 70.0,70.0);}
for(  uint32_t i=0;i<BlockchainNode.GetN ();i++){ pAnim->UpdateNodeDescription (BlockchainNode.Get (i), " "); 
Ptr<Node> wid= BlockchainNode.Get (i); uint32_t nodeId = wid->GetId (); pAnim->UpdateNodeImage (nodeId, BlockchainNodeimg);
pAnim->UpdateNodeColor(BlockchainNode.Get(i), 0, 255, 0); pAnim->UpdateNodeSize (nodeId, 100.0,100.0);}         
for(  uint32_t i=0;i<EdgeAssistedSinkNodes.GetN ();i++){ pAnim->UpdateNodeDescription (EdgeAssistedSinkNodes.Get (i), "Edge_Assisted_Sink"); 
Ptr<Node> wid= EdgeAssistedSinkNodes.Get (i); uint32_t nodeId = wid->GetId ();
pAnim->UpdateNodeImage (nodeId, EdgeNodesimg); pAnim->UpdateNodeColor(EdgeAssistedSinkNodes.Get(i), 0, 255, 0); 
pAnim->UpdateNodeSize (nodeId, 60.0,60.0);}
for(  uint32_t i=0;i<MasterControllerNodes.GetN ();i++){ pAnim->UpdateNodeDescription (MasterControllerNodes.Get (i), "Master_Controller"); 
Ptr<Node> wid= MasterControllerNodes.Get (i); uint32_t nodeId = wid->GetId ();
pAnim->UpdateNodeImage (nodeId, MasterControllerNodesimg);pAnim->UpdateNodeColor(MasterControllerNodes.Get(i), 0, 255, 0); 
pAnim->UpdateNodeSize (nodeId, 125.0,125.0);}for(  uint32_t i=0;i<SwitchNodes.GetN ();i++){
pAnim->UpdateNodeDescription (SwitchNodes.Get (i), "Switch"); 
Ptr<Node> wid= SwitchNodes.Get (i);uint32_t nodeId = wid->GetId ();
pAnim->UpdateNodeImage (nodeId, SwitchNodesimg); pAnim->UpdateNodeColor(SwitchNodes.Get(i), 0, 255, 0); 
pAnim->UpdateNodeSize (nodeId, 75.0,75.0);} for(  uint32_t i=0;i<SlaveControllerNodes.GetN ();i++){
pAnim->UpdateNodeDescription (SlaveControllerNodes.Get (i), "Slave_Controller"); 
Ptr<Node> wid= SlaveControllerNodes.Get (i); uint32_t nodeId = wid->GetId ();
pAnim->UpdateNodeImage (nodeId, SlaveControllerimg); pAnim->UpdateNodeColor(SlaveControllerNodes.Get(i), 0, 255, 0); 
pAnim->UpdateNodeSize (nodeId, 100.0,100.0);} for(  uint32_t i=0;i<TA.GetN ();i++){
pAnim->UpdateNodeDescription (TA.Get (i), "TA");  Ptr<Node> wid= TA.Get (i); uint32_t nodeId = wid->GetId ();
pAnim->UpdateNodeImage (nodeId, TAimg); pAnim->UpdateNodeColor(TA.Get(i), 0, 255, 0); 
pAnim->UpdateNodeSize (nodeId, 100.0,100.0);} for(  uint32_t i=0;i<CloudNode.GetN ();i++){
pAnim->UpdateNodeDescription (CloudNode.Get (i), "Cloud"); 
Ptr<Node> wid= CloudNode.Get (i); uint32_t nodeId = wid->GetId ();
pAnim->UpdateNodeImage (nodeId, Cloudimg); pAnim->UpdateNodeColor(CloudNode.Get(i), 0, 255, 0); 
pAnim->UpdateNodeSize (nodeId, 125.0,125.0);} FlowMonitorHelper flowmon; Ptr<FlowMonitor> monitor = flowmon.InstallAll();
Simulator::Run (); monitor->CheckForLostPackets (); uint32_t LostPacketsum = 0; uint32_t rxPacketsum = 0;
uint32_t DropPacketsum = 0; double Delaysum = 0.035; 
Ptr<Ipv4FlowClassifier> classifier = DynamicCast<Ipv4FlowClassifier> (flowmon.GetClassifier ());
std::map<FlowId, FlowMonitor::FlowStats> stats = monitor->GetFlowStats ();
for (std::map<FlowId, FlowMonitor::FlowStats>::const_iterator i = stats.begin (); i != stats.end (); ++i){
rxPacketsum += (i->second.txBytes/(numNodes*10)); LostPacketsum += i->second.lostPackets;
DropPacketsum += i->second.packetsDropped.size(); Delaysum += i->second.delaySum.GetSeconds();} Simulator::Destroy (); return 0;}
