/**
 * @file main.cc
 * @brief Simple pkt generator application using core library.
 */
#include <arp.h>
#include <dpdk.h>
#include <ether.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <ipv4.h>
#include <machnet_config.h>
#include <math.h>
#include <packet.h>
#include <pmd.h>
#include <udp.h>
#include <utils.h>
#include <worker.h>

#include <deque>
#include <csignal>
#include <iostream>
#include <optional>
#include <vector>

#include "ttime.h"

// The packets we send carry an ethernet, IPv4 and UDP header.
// The payload includes at least two 64-bit unsigned integers. The first one is
// used as a sequence number (0-indexed), and the second is the timestamp at
// packet creation (TSC).
constexpr uint16_t kMinPacketLength =
    std::max(sizeof(juggler::net::Ethernet) + sizeof(juggler::net::Ipv4) +
                 sizeof(juggler::net::Udp) + 2 * sizeof(uint64_t),
             64ul);

DEFINE_uint32(pkt_size, kMinPacketLength, "Packet size.");
DEFINE_uint64(tx_batch_size, 32, "DPDK TX packet batch size");
DEFINE_string(
    config_json, "../src/apps/machnet/config.json",
    "Machnet JSON configuration file (shared with the pktgen application).");
DEFINE_string(remote_ip1, "", "IPv4 address of the remote sender.");
DEFINE_string(remote_ip2, "", "IPv4 address of the remote receiver.");
DEFINE_bool(ping, false, "Ping-pong remote host for RTT measurements.");
DEFINE_bool(active_generator, false,
            "When 'true' this host is generating the traffic, otherwise it is "
            "bouncing.");
DEFINE_bool(zerocopy, true, "Use memcpy to fill packet payload.");
DEFINE_string(rtt_log, "", "Log file for RTT measurements.");

DEFINE_uint64(rate, 1500, "Rate in Kbit/s.");
DEFINE_uint64(burst, 30000, "Bucket size in bytes.");
DEFINE_uint64(qlen, 32000, "Queue length in bytes.");
DEFINE_uint64(num_aggregates, 2050, "Number of pqp instantiations.");
DEFINE_uint64(num_queues, 64, "Number of queues per pqp instantiations.");


// This is the source/destination UDP port used by the application.
const uint16_t kAppUDPPort = 6666;

static volatile int g_keep_running = 1;

void int_handler([[maybe_unused]] int signal) { g_keep_running = 0; }

// Structure to keep application statistics related to TX and RX packets/errors.
struct stats {
  stats()
      : tx_success(0),
        tx_bytes(0),
        rx_count(0),
        rx_bytes(0),
        err_no_mbufs(0),
        err_tx_drops(0),
        enq_cycles(0),
        deq_cycles(0) {}
  uint64_t tx_success;
  uint64_t tx_bytes;
  uint64_t rx_count;
  uint64_t rx_bytes;
  uint64_t err_no_mbufs;
  uint64_t err_tx_drops;
  uint64_t enq_cycles;
  uint64_t deq_cycles;
};


struct pqp_queue_index {
  pqp_queue_index(uint64_t s_i, uint64_t q_i)
      : pqp(s_i),
        queue(q_i) {}
  uint64_t pqp;
  uint64_t queue;
};

struct pqp_struct {
  /**
   * @param rate rate in Mbps
   * @param burst rate in Mbps
   * @param num_queues rate in Mbps
   * @param q_sizes queue size counters in bytes
   * `std::vector<uint64_t>` format.
   * @param qs Local IP address to be used by the applicaton in
   * `std::vector<juggler::dpdk::Packet *>` format.
   */
  pqp_struct(float rate, uint64_t burst, uint16_t num_queues,
                std::vector<uint64_t> q_sizes)
      : tokens(burst),
        rate(rate),
        bytes_per_ms(rate*1024.0*1024.0/8000.0),
        burst(burst),
        num_queues(num_queues),
        i(0),    
        scheduled(0),
        last_dequeue_time(0),
        last_dq_rate_calc_time(0),
        upper_threshold(0),
        queue_sizes(q_sizes),
        bytes_in_last_period({}),
        dequeues_in_last_period({}),
        last_enqueue_calc_time({}),
        magic_packets({}),
        active_queues({}) {
          bytes_in_last_period.resize(num_queues, 0);
          dequeues_in_last_period.resize(num_queues, 0);
          last_enqueue_calc_time.resize(num_queues, 0);
          magic_packets.resize(num_queues, 0);
          upper_threshold = 100.0 * bytes_per_ms;
        }
  uint64_t tokens;
  float rate;
  float bytes_per_ms;
  uint64_t burst;
  uint16_t num_queues;
  uint16_t i;
  bool scheduled;
  uint64_t last_dequeue_time;
  uint64_t last_dq_rate_calc_time;
  float upper_threshold;
  std::vector<uint64_t> queue_sizes;
  std::vector<uint64_t> bytes_in_last_period;
  std::vector<uint64_t> dequeues_in_last_period;
  std::vector<uint64_t> last_enqueue_calc_time;
  std::vector<uint64_t> magic_packets;
  std::vector<uint64_t> active_queues;
};


/**
 * @brief This structure contains all the metadata required for all the routines
 * related to the pqp.
 */
struct pqp_context {
  /**
   * @param rate Rate for each pqp in Mbps
   * `float` format.
   * @param qlen Size of each queue in pqp
   * `uint_32t` format.
   * @param burst Size of bucket in each pqp
   * `uint_32t` format.
   * @param num_pqps Number of pqps
   * `uint_16t` format.
   * @param num_queues Number of queues per pqp
   * `uint_8t` format.
   * @param pqps_list List of pqps
   * `std::vector<pqp_struct*>` format.
   */
  pqp_context(float rate,
               uint32_t qlen,
               uint32_t burst,
               uint16_t num_pqps,
               uint8_t num_queues,
               std::vector<pqp_struct*> pqps_list)
      : rate(rate),
        qlen(qlen),
        burst(burst),
        num_pqps(num_pqps),
        num_queues(num_queues),
        pqps(pqps_list){}
  float rate;
  uint32_t qlen;
  uint32_t burst;
  uint16_t num_pqps;
  uint8_t num_queues;
  std::vector<pqp_struct*> pqps;
};

/**
 * @brief This structure contains all the metadata required for all the routines
 * implemented in this application.
 */
struct task_context {
  /**
   * @param local_mac Local MAC address of the DPDK port in
   * `juggler::net::Ethernet::Address` format.
   * @param local_ip Local IP address to be used by the applicaton in
   * `juggler::net::Ipv4::Address` format.
   * @param remote_mac1 (std::optional) Remote host's MAC address in
   * `juggler::net::Ethernet::Address` format. `std::nullopt` in passive mode.
   * @param remote_ip1 (std::optional) Remote host's IP address to be used by the
   * applicaton in `juggler::net::Ipv4::Address` format. `std::nullopt` in
   * passive mode.
   * @param remote_mac2 (std::optional) Remote host's MAC address in
   * `juggler::net::Ethernet::Address` format. `std::nullopt` in passive mode.
   * @param remote_ip2 (std::optional) Remote host's IP address to be used by the
   * applicaton in `juggler::net::Ipv4::Address` format. `std::nullopt` in
   * passive mode.
   * @param packet_size Size of the packets to be generated.
   * @param rxring Pointer to the RX ring previously initialized.
   * @param txring Pointer to the TX ring previously initialized.
   */
  task_context(juggler::net::Ethernet::Address local_mac,
               juggler::net::Ipv4::Address local_ip,
               std::optional<juggler::net::Ethernet::Address> remote_mac1,
               std::optional<juggler::net::Ipv4::Address> remote_ip1,
               std::optional<juggler::net::Ethernet::Address> remote_mac2,
               std::optional<juggler::net::Ipv4::Address> remote_ip2,
               uint16_t packet_size, juggler::dpdk::RxRing *rxring,
               juggler::dpdk::TxRing *txring,
               std::vector<std::vector<uint8_t>> payloads,
               pqp_context* pqp_ctx)
      : local_mac_addr(local_mac),
        local_ipv4_addr(local_ip),
        remote_mac_addr(remote_mac1),
        remote_ipv4_addr(remote_ip1),
        remote_mac_addr2(remote_mac2),
        remote_ipv4_addr2(remote_ip2),
        packet_size(packet_size),
        rx_ring(CHECK_NOTNULL(rxring)),
        tx_ring(CHECK_NOTNULL(txring)),
        packet_payloads(payloads),
        pqp_ctx(pqp_ctx),
        arp_handler(local_mac, {local_ip}),
        statistics(),
        rtt_log() {}
  const juggler::net::Ethernet::Address local_mac_addr;
  const juggler::net::Ipv4::Address local_ipv4_addr;
  const std::optional<juggler::net::Ethernet::Address> remote_mac_addr;
  const std::optional<juggler::net::Ipv4::Address> remote_ipv4_addr;
  const std::optional<juggler::net::Ethernet::Address> remote_mac_addr2;
  const std::optional<juggler::net::Ipv4::Address> remote_ipv4_addr2;
  const uint16_t packet_size;

  juggler::dpdk::PacketPool *packet_pool;
  juggler::dpdk::RxRing *rx_ring;
  juggler::dpdk::TxRing *tx_ring;

  std::vector<std::vector<uint8_t>> packet_payloads;
  pqp_context* pqp_ctx;
  juggler::ArpHandler arp_handler;

  stats statistics;
  juggler::utils::TimeLog rtt_log;
};



/**
 * @brief Resolves the MAC address of a remote IP using ARP, with busy-waiting.
 *
 * This function sends an ARP request to resolve the MAC address of a given
 * remote IP address and waits for a response for the provided timeout duration.
 * If the MAC address is successfully resolved within the timeout, it is
 * returned; otherwise, a nullopt is returned.
 *
 * @param local_mac Local Ethernet MAC address.
 * @param local_ip Local IPv4 address.
 * @param tx_ring Pointer to the DPDK transmit ring.
 * @param rx_ring Pointer to the DPDK receive ring.
 * @param remote_ip Remote IPv4 address whose MAC address needs to be resolved.
 * @param timeout_in_sec Maximum time in seconds to wait for the ARP response.
 *
 * @return MAC address of the remote IP if successfully resolved, otherwise
 * returns nullopt.
 */
std::optional<juggler::net::Ethernet::Address> ArpResolveBusyWait(
    const juggler::net::Ethernet::Address &local_mac,
    const juggler::net::Ipv4::Address &local_ip, juggler::dpdk::TxRing *tx_ring,
    juggler::dpdk::RxRing *rx_ring,
    const juggler::net::Ipv4::Address &remote_ip, int timeout_in_sec) {
  juggler::ArpHandler arp_handler(local_mac, {local_ip});

  arp_handler.GetL2Addr(tx_ring, local_ip, remote_ip);
  auto start = std::chrono::steady_clock::now();
  while (true) {
    auto now = std::chrono::steady_clock::now();
    if (now - start > std::chrono::seconds(timeout_in_sec)) {
      LOG(ERROR) << "ARP resolution timed out.";
      return std::nullopt;
    }

    juggler::dpdk::PacketBatch batch;
    auto nr_rx = rx_ring->RecvPackets(&batch);
    for (uint16_t i = 0; i < nr_rx; i++) {
      const auto *packet = batch[i];
      if (packet->length() <
          sizeof(juggler::net::Ethernet) + sizeof(juggler::net::Arp)) {
        continue;
      }
      auto *eh = packet->head_data<juggler::net::Ethernet *>();
      if (eh->eth_type.value() != juggler::net::Ethernet::kArp) {
        continue;
      }

      auto *arph = packet->head_data<juggler::net::Arp *>(
          sizeof(juggler::net::Ethernet));
      arp_handler.ProcessArpPacket(tx_ring, arph);
      auto remote_mac = arp_handler.GetL2Addr(tx_ring, local_ip, remote_ip);
      if (remote_mac.has_value()) {
        batch.Release();
        return remote_mac;
      }
    }
    batch.Release();
  }
}


/**
 * @brief Helper function to report TX/RX statistics at second granularity. It
 * keeps a checkpoint of the statistics since the previous report to calculate
 * per-second statistics.
 *
 * @param now Current TSC.
 * @param cur Pointer to the main statistics object.
 */
void report_stats(uint64_t now, void *context) {
  static const size_t kGiga = 1E9;
  thread_local uint64_t last_report_timestamp;
  thread_local stats stats_checkpoint;
  const auto *ctx = static_cast<task_context *>(context);
  const stats *cur = &ctx->statistics;

  auto cycles_elapsed = now - last_report_timestamp;
  if (cycles_elapsed > juggler::time::s_to_cycles(1)) {
    auto sec_elapsed = juggler::time::cycles_to_s<double>(cycles_elapsed);
    auto packets_sent = cur->tx_success - stats_checkpoint.tx_success;
    auto tx_pps = static_cast<double>(packets_sent) / sec_elapsed;
    auto tx_gbps =
        static_cast<double>(cur->tx_bytes - stats_checkpoint.tx_bytes) /
        sec_elapsed * 8.0 / kGiga;
    auto packets_received = cur->rx_count - stats_checkpoint.rx_count;
    auto rx_pps = static_cast<double>(packets_received / sec_elapsed);
    auto rx_gbps =
        static_cast<double>(cur->rx_bytes - stats_checkpoint.rx_bytes) /
        sec_elapsed * 8.0 / kGiga;
    auto packets_dropped = cur->err_tx_drops - stats_checkpoint.err_tx_drops;
    auto tx_drop_pps = static_cast<double>(packets_dropped / sec_elapsed);
    auto enq_cycles = cur->enq_cycles - stats_checkpoint.enq_cycles;
    auto deq_cycles = cur->deq_cycles - stats_checkpoint.deq_cycles;
    auto cycles_per_pkt_enq = static_cast<double>(enq_cycles) / static_cast<double>(packets_received);
    auto cycles_per_pkt_deq = static_cast<double>(deq_cycles) / static_cast<double>(packets_sent);
    
    LOG(INFO) << juggler::utils::Format(
        "[TX PPS: %lf (%lf Gbps), RX PPS: %lf (%lf Gbps), TX_DROP PPS: %lf, Cycles(Enq: %lf, Deq:%lf)]",
        tx_pps, tx_gbps, rx_pps, rx_gbps, tx_drop_pps, cycles_per_pkt_enq, cycles_per_pkt_deq);

    // Update local variables for next report.
    stats_checkpoint = *cur;
    last_report_timestamp = now;
  }
}

void report_final_stats(void *context) {
  if (!FLAGS_ping) return;

  auto *ctx = static_cast<task_context *>(context);
  const stats *st = &ctx->statistics;
  auto *rtt_log = &ctx->rtt_log;

  using stats_tuple =
      std::tuple<double, double, size_t, size_t, size_t, size_t, size_t>;
  decltype(auto) rtt_stats_calc =
      [](std::vector<uint64_t> &samples) -> std::optional<stats_tuple> {
    uint64_t sum = 0;
    uint64_t count = 0;
    for (const auto &sample : samples) {
      if (sample == 0) break;
      sum += sample;
      count++;
    }
    if (count == 0) {
      return std::nullopt;
    }
    double mean = static_cast<double>(sum) / count;
    double variance = 0;
    for (const auto &sample : samples) {
      if (sample == 0) break;
      variance += (sample - mean) * (sample - mean);
    }
    variance /= count;
    double stddev = std::sqrt(variance);

    sort(samples.begin(), samples.end());

    auto min = samples[0];
    auto p50 = samples[static_cast<size_t>(count * 0.5)];
    auto p99 = samples[static_cast<size_t>(count * 0.99)];
    auto p999 = samples[static_cast<size_t>(count * 0.999)];
    auto max = samples[count - 1];

    return std::make_tuple(mean, stddev, p50, p99, p999, min, max);
  };

  if (FLAGS_rtt_log != "") {
    rtt_log->DumpToFile(FLAGS_rtt_log);
  }

  auto rtt_stats = rtt_log->Apply<std::optional<stats_tuple>>(rtt_stats_calc);
  if (!rtt_stats.has_value()) {
    LOG(INFO) << "RTT (ns): no samples";
    return;
  }
  auto [mean, stddev, p50, p99, p999, min, max] = rtt_stats.value();
  LOG(INFO) << "RTT (ns) mean= " << mean << ", stddev=" << stddev
            << ", p50=" << p50 << ", p99=" << p99 << ", p999=" << p999
            << ", min=" << min << ", max=" << max;
  LOG(INFO) << juggler::utils::Format(
      "Application Statistics (TOTAL) - [TX] Sent: %lu, Drops: %lu, "
      "DropsNoMbuf: %lu "
      "[RX] Received: %lu",
      st->tx_success, st->err_tx_drops, st->err_no_mbufs, st->rx_count);
}

// classifies packet into pqp and queue. This can be redefined later.
// I am classifying based on port numbers. Flows with contiguous ports are classified into one pqp
// and a separate queue within a pqp for each port
pqp_queue_index* classify(juggler::net::Ipv4 * ipv4h, task_context* ctx){
    uint64_t pqp = 0;
    uint64_t queue = 0;
    auto *udph = reinterpret_cast<juggler::net::Udp *>(ipv4h + 1);
    uint16_t dst_port =  uint16_t(udph->src_port.port.value());

    if (ipv4h->dst_addr.address == juggler::be32_t(ctx->remote_ipv4_addr.value().address)){
      pqp += 1;
      pqp += (dst_port / ctx->pqp_ctx->num_queues);
      queue += (dst_port % ctx->pqp_ctx->num_queues);
    } else if (ipv4h->dst_addr.address == juggler::be32_t(ctx->remote_ipv4_addr2.value().address)){
      pqp += ctx->pqp_ctx->num_pqps/2;
      pqp += (dst_port / ctx->pqp_ctx->num_queues);
      queue += (dst_port % ctx->pqp_ctx->num_queues);
    }

    if (pqp){
      pqp_queue_index* sqi = new pqp_queue_index(pqp, queue);
      return sqi;
    }
    return NULL;
}

// this function is called for any pqp, when one of its queues is full and we need to accept a packet
void phantom_dequeue(uint64_t now, pqp_struct* pqp){
  uint64_t time_elapsed = now - pqp->last_dequeue_time;
  float time_elapsed_ms = juggler::time::cycles_to_ms(time_elapsed);
  // based on last dequeue time, decide how many 'tokens' to dequeue from all queues
  // no upper limit, because we call this function non-periodically when a queue becomes full
  pqp->tokens = pqp->bytes_per_ms * time_elapsed_ms;
  uint16_t active_queues = pqp->active_queues.size();
  // no need to execute more often than this, because packets may be dropped anyways
  if (pqp->tokens < (1500 * active_queues)){
    return;
  }
  pqp->last_dequeue_time = now;
  // divides tokens in a work-conserving fair manner through a water filling algorithm
  while (active_queues && pqp->tokens / active_queues){
    uint64_t per_queue_tokens = pqp->tokens / active_queues;
    for (uint16_t i=0; i<active_queues; i++){
      uint64_t q_i = pqp->active_queues[i];
      // std::cout<<"before "<< pqp->queue_sizes[q_i]<<std::endl;
      if (pqp->queue_sizes[q_i] > per_queue_tokens){
        pqp->queue_sizes[q_i] -= per_queue_tokens;
        pqp->tokens -= per_queue_tokens;
        pqp->dequeues_in_last_period[q_i] += per_queue_tokens;
      } else {
        pqp->tokens -= pqp->queue_sizes[q_i];
        pqp->dequeues_in_last_period[q_i] += pqp->queue_sizes[q_i];
        pqp->queue_sizes[q_i] = 0;
        pqp->active_queues.erase(std::remove(pqp->active_queues.begin(), pqp->active_queues.end(), q_i), pqp->active_queues.end());
        active_queues = pqp->active_queues.size();
      }
      if (juggler::time::cycles_to_ms(now - pqp->last_enqueue_calc_time[q_i]) > 200){
        if (active_queues){ 
          pqp->upper_threshold = (pqp->bytes_per_ms * 200.0) / active_queues;
        } else {
          pqp->upper_threshold = (pqp->bytes_per_ms * 200.0);
        }
        if (pqp->bytes_in_last_period[q_i] < (pqp->upper_threshold * 0.5)){
          // need to check this here too to catch flows that are not sending any packets
          pqp->queue_sizes[q_i] -= std::min(pqp->queue_sizes[q_i], pqp->magic_packets[q_i]);
          if (!pqp->queue_sizes[q_i]){
            pqp->active_queues.erase(std::remove(pqp->active_queues.begin(), pqp->active_queues.end(), q_i), pqp->active_queues.end());
            active_queues = pqp->active_queues.size();
          }
          pqp->magic_packets[q_i] = 0;
        }
        pqp->last_enqueue_calc_time[q_i] = now;
        pqp->bytes_in_last_period[q_i] = 0;
      }
    }
  }
}



// Main routine. Receives incoming packets, classifies them and determines whether to transmit
// or drop packets
void enqueue(uint64_t now, void *context) {
  auto ctx = static_cast<task_context *>(context);
  auto pqp_ctx = ctx->pqp_ctx;
  auto rx = ctx->rx_ring;
  auto tx = ctx->tx_ring;
  auto *st = &ctx->statistics;

  juggler::dpdk::PacketBatch rx_batch;
  auto packets_received = rx->RecvPackets(&rx_batch);
  std::array<size_t, juggler::dpdk::PacketBatch::kMaxBurst> tx_bytes;
  juggler::dpdk::PacketBatch tx_batch;
  uint64_t batch_cycles = 0;
  for (uint16_t i = 0; i < rx_batch.GetSize(); i++) {
    auto *packet = rx_batch[i];
    CHECK_LE(sizeof(juggler::net::Ethernet), packet->length());

    auto *eh = packet->head_data<juggler::net::Ethernet *>();

    // Check if the packet is an ARP request and process it.
    if (eh->eth_type.value() == juggler::net::Ethernet::kArp) {
      const auto *arph = packet->head_data<juggler::net::Arp *>(
          sizeof(juggler::net::Ethernet));
      ctx->arp_handler.ProcessArpPacket(tx, arph);
      // We are going to drop this packet, so we need to explicitly reclaim the
      // mbuf now.
      juggler::dpdk::Packet::Free(packet);
      continue;
    }

    auto *ipv4h = packet->head_data<juggler::net::Ipv4 *>(sizeof(*eh));
    // configure headers to relay traffic, we have a proxy like architecture, where sender attempts to send data
    // to a receiver but iptables on sender (and receiver) reroute packets to the pqp proxy. So we need to replace
    // mac and ip headers here.
    if (ipv4h->src_addr.address == juggler::be32_t(ctx->remote_ipv4_addr.value().address)){
      
      eh->src_addr = ctx->local_mac_addr;
      eh->dst_addr = ctx->remote_mac_addr2.value();

      ipv4h->dst_addr.address = juggler::be32_t(ctx->remote_ipv4_addr2.value().address);
      ipv4h->src_addr.address = juggler::be32_t(ctx->local_ipv4_addr.address);
      // since header changed, need to recompute checksum.
      packet->set_l2_len(sizeof(*eh));
      packet->set_l3_len(sizeof(*ipv4h));
      packet->offload_udpv4_csum();

    } else if (ipv4h->src_addr.address == juggler::be32_t(ctx->remote_ipv4_addr2.value().address)){
      
      eh->src_addr = ctx->local_mac_addr;
      eh->dst_addr = ctx->remote_mac_addr.value();

      ipv4h->dst_addr.address = juggler::be32_t(ctx->remote_ipv4_addr.value().address);
      ipv4h->src_addr.address = juggler::be32_t(ctx->local_ipv4_addr.address);

      // since header changed, need to recompute checksum.
      packet->set_l2_len(sizeof(*eh));
      packet->set_l3_len(sizeof(*ipv4h));
      packet->offload_udpv4_csum();

    } else {

      juggler::net::Ethernet::Address tmp;
      juggler::utils::Copy(&tmp, &eh->dst_addr, sizeof(tmp));
      juggler::utils::Copy(&eh->dst_addr, &eh->src_addr, sizeof(eh->dst_addr));
      juggler::utils::Copy(&eh->src_addr, &tmp, sizeof(eh->src_addr));

      ipv4h->dst_addr.address = ipv4h->src_addr.address;
      ipv4h->src_addr.address = juggler::be32_t(ctx->local_ipv4_addr.address);

    }

    // classify packet into a pqp id and queue id
    auto pqp_queue_id = classify(ipv4h, ctx);
    uint64_t cycles = _rdtsc();
    if (pqp_queue_id){

      auto pqp_id = pqp_queue_id->pqp;
      auto queue_id = pqp_queue_id->queue;
      // if queue is full, only then phantom_dequeue is called
      if(pqp_ctx->pqps[pqp_id]->queue_sizes[queue_id] + packet->length() > pqp_ctx->qlen){
        phantom_dequeue(now, pqp_ctx->pqps[pqp_id]);
      }
      if(pqp_ctx->pqps[pqp_id]->queue_sizes[queue_id] + packet->length() <= pqp_ctx->qlen){
        // can accept and transmit this packet
        if (!pqp_ctx->pqps[pqp_id]->queue_sizes[queue_id]){
          // this queue has become active
          pqp_ctx->pqps[pqp_id]->active_queues.push_back(queue_id);
        }
        pqp_ctx->pqps[pqp_id]->queue_sizes[queue_id] += packet->length();
        pqp_ctx->pqps[pqp_id]->bytes_in_last_period[queue_id] += packet->length();

        if (!pqp_ctx->pqps[pqp_id]->last_enqueue_calc_time[queue_id]){
          pqp_ctx->pqps[pqp_id]->last_enqueue_calc_time[queue_id] = now; 
        }

        // calculate theta to decide whether to add magic packets or not (we are doing it for fairness here)
        uint16_t active_queues = pqp_ctx->pqps[pqp_id]->active_queues.size();
        if (active_queues){ 
          pqp_ctx->pqps[pqp_id]->upper_threshold = (pqp_ctx->pqps[pqp_id]->bytes_per_ms * 200.0) / active_queues;
        } else {
          pqp_ctx->pqps[pqp_id]->upper_threshold = (pqp_ctx->pqps[pqp_id]->bytes_per_ms * 200.0);
        }

        // if number of packets accepted in this time period has exceeded upper threshold, add magic packets
        if (pqp_ctx->pqps[pqp_id]->bytes_in_last_period[queue_id] > (pqp_ctx->pqps[pqp_id]->upper_threshold * 1.5)){
          if (pqp_id > 1024)
          // need to phantom dequeue before to ensure next packet sees a full queue
          phantom_dequeue(now, pqp_ctx->pqps[pqp_id]);
          pqp_ctx->pqps[pqp_id]->magic_packets[queue_id] += (pqp_ctx->qlen - pqp_ctx->pqps[pqp_id]->queue_sizes[queue_id]);
          if (!pqp_ctx->pqps[pqp_id]->queue_sizes[queue_id]){
            pqp_ctx->pqps[pqp_id]->active_queues.push_back(queue_id);
          }
          pqp_ctx->pqps[pqp_id]->queue_sizes[queue_id] = pqp_ctx->qlen;
          pqp_ctx->pqps[pqp_id]->last_enqueue_calc_time[queue_id] = now;
          pqp_ctx->pqps[pqp_id]->bytes_in_last_period[queue_id] = 0;
        } else if (juggler::time::cycles_to_ms(now - pqp_ctx->pqps[pqp_id]->last_enqueue_calc_time[queue_id]) > 200){
          // check every T time to decide whether to remove magic packets
          if (pqp_ctx->pqps[pqp_id]->bytes_in_last_period[queue_id] < (pqp_ctx->pqps[pqp_id]->upper_threshold * 0.5)){
            // remove magic packets
            pqp_ctx->pqps[pqp_id]->queue_sizes[queue_id] -= std::min(pqp_ctx->pqps[pqp_id]->queue_sizes[queue_id], pqp_ctx->pqps[pqp_id]->magic_packets[queue_id]);
            if (!pqp_ctx->pqps[pqp_id]->queue_sizes[queue_id])
              pqp_ctx->pqps[pqp_id]->active_queues.erase(std::remove(pqp_ctx->pqps[pqp_id]->active_queues.begin(), pqp_ctx->pqps[pqp_id]->active_queues.end(), queue_id), pqp_ctx->pqps[pqp_id]->active_queues.end());
            pqp_ctx->pqps[pqp_id]->magic_packets[queue_id] = 0;
          }
          pqp_ctx->pqps[pqp_id]->last_enqueue_calc_time[queue_id] = now;
          pqp_ctx->pqps[pqp_id]->bytes_in_last_period[queue_id] = 0;
        }

        tx_batch.Append(packet);

        const auto tx_bytes_index = tx_batch.GetSize() - 1;
        st->rx_bytes += packet->length();
        tx_bytes[tx_bytes_index] = packet->length();
        if (tx_bytes_index != 0)
          tx_bytes[tx_bytes_index] += tx_bytes[tx_bytes_index - 1];
      } else {
        // drop the packet, free the mbuf
        juggler::dpdk::Packet::Free(packet);
      }

    } else {
      // Unclassified packets are dequeued immediately
      tx_batch.Append(packet);

      const auto tx_bytes_index = tx_batch.GetSize() - 1;
      st->rx_bytes += packet->length();
      tx_bytes[tx_bytes_index] = packet->length();
      if (tx_bytes_index != 0)
        tx_bytes[tx_bytes_index] += tx_bytes[tx_bytes_index - 1];
    }
    cycles = _rdtsc() - cycles;
    batch_cycles += cycles;
  }
  rx_batch.Clear();

  auto packets_sent = tx->TrySendPackets(&tx_batch);
  st->err_tx_drops += packets_received - packets_sent;
  st->tx_success += packets_sent;
  if (packets_sent) st->tx_bytes += tx_bytes[packets_sent - 1];
  st->rx_count += packets_received;
  st->enq_cycles += batch_cycles;
}

int main(int argc, char *argv[]) {
  ::google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  gflags::SetUsageMessage("Simple packet pqp.");

  signal(SIGINT, int_handler);

  // Parse the remote IP address.
  std::optional<juggler::net::Ipv4::Address> remote_ip1;
  std::optional<juggler::net::Ipv4::Address> remote_ip2;
  if (FLAGS_remote_ip1.empty()) {
    LOG(ERROR) << "Remote IP address is required in pqp mode.";
    exit(1);
  }
  if (FLAGS_remote_ip2.empty()) {
    LOG(ERROR) << "Remote IP address is required in pqp mode.";
    exit(1);
  }
  remote_ip1 = juggler::net::Ipv4::Address::MakeAddress(FLAGS_remote_ip1);
  remote_ip2 = juggler::net::Ipv4::Address::MakeAddress(FLAGS_remote_ip2);
  CHECK(remote_ip1.has_value())
      << "Invalid remote IP address: " << FLAGS_remote_ip1;
  CHECK(remote_ip2.has_value())
      << "Invalid remote IP address: " << FLAGS_remote_ip2;

  // Load the configuration file.
  juggler::MachnetConfigProcessor machnet_config(FLAGS_config_json);
  if (machnet_config.interfaces_config().size() != 1) {
    // For machnet config, we expect only one interface to be configured.
    LOG(ERROR) << "Only one interface should be configured.";
    exit(1);
  }

  const auto &interface = *machnet_config.interfaces_config().begin();

  juggler::dpdk::Dpdk dpdk_obj;
  dpdk_obj.InitDpdk(machnet_config.GetEalOpts());

  const auto pmd_port_id = dpdk_obj.GetPmdPortIdByMac(interface.l2_addr());
  if (!pmd_port_id.has_value()) {
    LOG(ERROR) << "Failed to find DPDK port ID for MAC address: "
               << interface.l2_addr().ToString();
    exit(1);
  }

  juggler::dpdk::PmdPort pmd_obj(pmd_port_id.value());
  pmd_obj.InitDriver();

  auto *rxring = pmd_obj.GetRing<juggler::dpdk::RxRing>(0);
  CHECK_NOTNULL(rxring);
  auto *txring = pmd_obj.GetRing<juggler::dpdk::TxRing>(0);
  CHECK_NOTNULL(txring);

  // Resolve the remote host's MAC address if we are in active mode.
  std::optional<juggler::net::Ethernet::Address> remote_l2_addr1 = std::nullopt;
  std::optional<juggler::net::Ethernet::Address> remote_l2_addr2 = std::nullopt;
  remote_l2_addr1 =
      ArpResolveBusyWait(interface.l2_addr(), interface.ip_addr(), txring,
                          rxring, remote_ip1.value(), 5);
  if (!remote_l2_addr1.has_value()) {
    LOG(ERROR) << "Failed to resolve remote host's MAC address.";
    exit(1);
  }
  remote_l2_addr2 =
      ArpResolveBusyWait(interface.l2_addr(), interface.ip_addr(), txring,
                          rxring, remote_ip2.value(), 5);
  if (!remote_l2_addr2.has_value()) {
    LOG(ERROR) << "Failed to resolve remote host's MAC address.";
    exit(1);
  }

  const uint16_t packet_len =
      std::max(std::min(static_cast<uint16_t>(FLAGS_pkt_size),
                        juggler::dpdk::PmdRing::kDefaultFrameSize),
               kMinPacketLength);

  std::vector<std::vector<uint8_t>> packet_payloads;
  if (!FLAGS_zerocopy) {
    const size_t kMaxPayloadMemory = 1 << 30;
    auto packet_payloads_nr = kMaxPayloadMemory / packet_len;
    // Round up to the closest power of 2.
    packet_payloads_nr = 1 << (64 - __builtin_clzll(packet_payloads_nr - 1));

    LOG(INFO) << "Allocating " << packet_payloads_nr
              << " payload buffers. (total: " << packet_payloads_nr * packet_len
              << " bytes)";
    packet_payloads.resize(packet_payloads_nr);
    for (auto &payload : packet_payloads) {
      payload.resize(packet_len);
    }
  }

  // auto tx_packet_pool = std::make_unique<juggler::dpdk::PacketPool>(4096);
  // We share the packet pool attached to the RX ring. Since we plan to handle a
  // queue pair from a single core this is safe.
  std::vector<pqp_struct*> pqps_list;
  float rate = float(static_cast<uint64_t>(FLAGS_rate)) / 1000.0;
  for (uint16_t i=0; i<FLAGS_num_aggregates; i++){
    std::vector<uint64_t> q_sizes;
    for (uint8_t j=0; j<FLAGS_num_queues; j++){
      q_sizes.push_back(0);
    }
    pqp_struct* pqp = new pqp_struct(rate, FLAGS_burst, static_cast<uint16_t>(FLAGS_num_queues), q_sizes);
    pqps_list.push_back(pqp);
  }

  pqp_context* pqp_ctx = new pqp_context(rate, FLAGS_qlen, FLAGS_burst, FLAGS_num_aggregates, FLAGS_num_queues, pqps_list);
  task_context task_ctx(interface.l2_addr(), interface.ip_addr(),
                        remote_l2_addr1, remote_ip1, remote_l2_addr2, remote_ip2, packet_len, rxring, txring,
                        packet_payloads, pqp_ctx);

  auto packet_pqp = [](uint64_t now, void *context) {
    enqueue(now, context);
    report_stats(now, context);
  };

  auto routine =
      packet_pqp;
  // Create a task object to pass to worker thread.
  auto task =
      std::make_shared<juggler::Task>(routine, static_cast<void *>(&task_ctx));

  std::cout << "Starting in phantom modes." << std::endl;

  juggler::WorkerPool<juggler::Task> WPool({task}, {interface.cpu_mask()});
  WPool.Init();

  // Set worker to running.
  WPool.Launch();

  while (g_keep_running) {
    sleep(5);
  }

  WPool.Pause();
  WPool.Terminate();
  report_final_stats(&task_ctx);
  pmd_obj.DumpStats();

  return (0);
}
