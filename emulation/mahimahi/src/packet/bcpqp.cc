#include <math.h>
#include "bcpqp.hh"
#include "timestamp.hh"
#include "infinite_packet_queue.hh"

using namespace std;


BCPQP::BCPQP( const string & args )
  : DroppingPacketQueue(args),
    burst ( get_arg( args, "burst") ),
    rate ( get_arg( args, "rate") ),
    pacing_rate ( get_arg( args, "pacing") ),
    num_flows ( get_arg( args, "flows") ),
    quantum ( get_arg( args, "quantum") ),
    limit(get_arg( args, "limit") ),
    rtt(get_arg( args, "resetlimit") ),
    last_limits({}),
    queue_size_in_bytes_ ({}),
    dequeues_in_last_rtt ({}),
    magic_packets ({}),
    queue_drain_rates ({}),
    last_dequeue_time (timestamp()),
    last_grad_compute_time (timestamp()),
    rate_in_bytes(1024.0 * rate / 8.0),
    gradient_threshold(0.5),
    log_(),
    timerwheel( {} )
{
  if ( burst == 0 || limit == 0 || rate ==0 || num_flows ==0 || quantum ==0) {
    throw runtime_error( "BCPQP must have burst, rate (Kbit/s), flow, quantum and limit arguments." );
  }

  // logfile for tbf
  for (uint32_t i=0; i<num_flows; i++){
    queue_size_in_bytes_.push_back(0);
    last_limits.push_back(0);
    dequeues_in_last_rtt.push_back(0);
    magic_packets.push_back(0);
    queue_drain_rates.push_back(0.0);
  }

  const string logfile = "bcpqp.log";
  if ( not logfile.empty() ) {
      log_.reset( new ofstream( logfile ) );
      if ( not log_->good() ) {
          throw runtime_error( logfile + ": error opening for writing" );
      }
      *log_ << "# init timestamp: " << initial_timestamp() << endl;
      *log_ << "# rate in bytes per second: " << rate_in_bytes << endl;
  }
  // internal queue to emulate a secondary bottleneck
  pacing_rate = 1024.0 * pacing_rate / 8.0;
  transmission_opportunity = (1000.0 * PACKET_SIZE) / pacing_rate;
  timerwheel_size = 2 * (num_flows * limit * 1024) / pacing_rate;
  timerwheel_size = timerwheel_size / transmission_opportunity;

  for (uint32_t i=0; i<timerwheel_size; i++){
    timerwheel.push_back(QueuedPacket("empty", 1));
  }
  reference_timestamp = timestamp();
}

QueuedPacket BCPQP::dequeue( void )
{ 
  // update number of tokens
  uint64_t now = timestamp();
  float time_since_token_update = float(now - last_dequeue_time) / 1000;
  uint32_t tokens_to_add = time_since_token_update * rate_in_bytes;
  tokens = std::min(burst, tokens + tokens_to_add);
  last_dequeue_time = now;
  // quantum-based round robin to distribute tokens between queues in a work-conserving manner.
  while (true){
    if (tokens <= 0){
      break;
    } 
    bool empty_queue = true;
    for (uint32_t i=0; i<num_flows; i++){
      empty_queue &= (queue_size_in_bytes_[i] == 0);
    }
    if (empty_queue){
      break;
    }
    while(true){ // to find a non-empty queue
      if (queue_size_in_bytes_[q] > 0){ // found a non-empty queue
        uint32_t temp_tks = std::min(tokens, partial_quantum);
        if (queue_size_in_bytes_[q] > temp_tks){
          // adjust queue size and remaining tokens
          queue_size_in_bytes_[q] -= temp_tks;
          dequeues_in_last_rtt[q] += temp_tks;
          tokens -= temp_tks;
          if (partial_quantum > temp_tks){
            // if fewer tokens were available than this queue's fair share, its remaining partial share must be adjusted when tokens become available next
            partial_quantum -= temp_tks;
          } else {
            // otherwise move to next queue
            q = (q + 1) % num_flows;  
            partial_quantum = PACKET_SIZE;

            if (q == 5) {
              partial_quantum = 5*PACKET_SIZE;

            }
          }
        } else {
          // queue size smaller than fair share, *work conserving*
          queue_size_in_bytes_[q] = 0;
          tokens -= temp_tks;
          dequeues_in_last_rtt[q] += temp_tks;
          q = (q + 1) % num_flows;  
          partial_quantum = PACKET_SIZE;
          if (q == 5) {
            partial_quantum = 5*PACKET_SIZE;

          }
        }
        break;
      } else { // move to next queue to be *work-conserving*
        q = (q + 1) % num_flows;  
        // if moving onto next queue, it must get opportunity to dequeue-adjust a full packet
        partial_quantum = PACKET_SIZE;
        if (q == 5) {
          partial_quantum = 5*PACKET_SIZE;

        }
      }
    }
  }

  if (now - last_grad_compute_time > rtt){
    for (uint32_t i=0; i<num_flows; i++){
        float queue_drain_rate = (float) ( (int) queue_size_in_bytes_[i] - (int) last_limits[i]) / ((float)(now - last_grad_compute_time)/1000.0);
        float dequeue_rate = (float) dequeues_in_last_rtt[i] / ((float)(now - last_grad_compute_time)/1000.0);
        if (dequeue_rate > 0){
          queue_drain_rates[i] = queue_drain_rate / dequeue_rate;
        } else {
          queue_drain_rates[i] = 0.0;
        }
        if (queue_drain_rate > gradient_threshold * dequeue_rate){
            *log_ << now << " AQM: " << queue_size_in_bytes_[i] << " " << last_limits[i] << " " << queue_drain_rate << endl;
            magic_packets[i] += (limit - queue_size_in_bytes_[i]);
            queue_size_in_bytes_[i] = limit;
        } else if (queue_drain_rate < gradient_threshold * dequeue_rate * -1) {
            *log_ << now << " AQM(-ve): " << queue_size_in_bytes_[i] << " " << last_limits[i] << " " << queue_drain_rate << endl;
            if (magic_packets[i] < queue_size_in_bytes_[i]){
              queue_size_in_bytes_[i] -= magic_packets[i];
              magic_packets[i] = 0;
            } else {
              magic_packets[i] -= queue_size_in_bytes_[i];
              queue_size_in_bytes_[i] = 0;
            }
        }
        last_limits[i] = queue_size_in_bytes_[i];
        dequeues_in_last_rtt[i] = 0;
    }
    last_grad_compute_time = now;
  }
  
  // dequeue a packet if actually available
  QueuedPacket ret = QueuedPacket("arbit", 0);
  if (not empty()) {
    *log_ << now << " Dequeued: ";
    for (uint32_t i=0; i<num_flows; i++){
      *log_ << queue_size_in_bytes_[i] << " ";
    }
    *log_ << endl;
    // *log_ << "Packet Dequeued." << endl;
    ret = std::move( DroppingPacketQueue::dequeue () );
  }

  if ((now - reference_timestamp) / transmission_opportunity > r_index){
    // could queue a packet
    r_index++;
    w_index = std::max(r_index +1, w_index);
    uint32_t now_index = r_index % timerwheel_size;
    if (timerwheel[now_index].contents != "empty"){
      accept(std::move(timerwheel[now_index])); 
    }
    timerwheel[now_index] = QueuedPacket("empty", 1);
  }
  return ret;
}


void BCPQP::enqueue(QueuedPacket&& p)
{
  uint16_t src, dst;
  std::string src_ip, dst_ip;
  _parse_ports_4((const unsigned char *) p.contents.substr(24,4).c_str(), &dst, &src);
  _parse_ip_4((const unsigned char *) p.contents.substr(16,8).c_str(), &src_ip, &dst_ip);
  // *log_<< "Paket header "<< src_ip << " "<< dst_ip<<endl;
  uint16_t q_i = src % num_flows;
  uint32_t now = timestamp();
  // if (queue_size_in_bytes_[src % num_flows] + p.contents.size() < limit && queue_drain_rates[src % num_flows] < 0.33 && good_with( size_bytes() + p.contents.size(),
  if (queue_size_in_bytes_[q_i % num_flows] + p.contents.size() < limit && good_with( size_bytes() + p.contents.size(),
                        size_packets() + 1 )){
    queue_size_in_bytes_[q_i % num_flows] += p.contents.size();
    *log_ << now << " Enqueued " << queue_size_in_bytes_[q_i % num_flows] << " " <<p.contents.size() << " for " << q_i%num_flows << endl;
    // accept( std::move( p ) );
    timerwheel[(w_index % timerwheel_size)] = std::move(p);
    w_index++;
  } else {
    queue_drain_rates[q_i % num_flows] = 0.0;
    *log_ << now << " Dropped " << queue_size_in_bytes_[q_i % num_flows] << " " <<p.contents.size() << " for " << q_i%num_flows << endl;
  }
  assert( good() );
}

