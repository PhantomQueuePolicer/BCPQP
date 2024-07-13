#include <math.h>
#include "pqp.hh"
#include "timestamp.hh"
#include "infinite_packet_queue.hh"

using namespace std;


PQP::PQP( const string & args )
  : DroppingPacketQueue(args),
    burst ( get_arg( args, "burst") ),
    limit ( get_arg( args, "limit") ),
    rate ( get_arg( args, "rate") ),
    num_flows ( get_arg( args, "flows") ),
    quantum ( get_arg( args, "quantum") ),
    queue_size_in_bytes_ ({}),
    last_dequeue_time (timestamp()),
    rate_in_bytes(1024.0 * rate / 8.0),
    log_()
{
  if ( burst == 0 || limit == 0 || rate ==0 || num_flows ==0 || quantum ==0) {
    throw runtime_error( "PQP must have burst, rate (Kbit/s), flow, quantum and limit arguments." );
  }

  // logfile for tbf
  for (uint32_t i=0; i<num_flows; i++){
    queue_size_in_bytes_.push_back(0);
  }

  const string logfile = "pqp.log";
  if ( not logfile.empty() ) {
      log_.reset( new ofstream( logfile ) );
      if ( not log_->good() ) {
          throw runtime_error( logfile + ": error opening for writing" );
      }
      *log_ << "# init timestamp: " << initial_timestamp() << endl;
      *log_ << "# rate in bytes per second: " << rate_in_bytes << endl;
  }
  // internal queue
}

QueuedPacket PQP::dequeue( void )
{ 
  // update number of tokens
  uint64_t now = timestamp();
  float time_since_token_update = float(now - last_dequeue_time) / 1000;
  uint32_t tokens_to_add = time_since_token_update * rate_in_bytes;
  tokens = std::min(burst, tokens + tokens_to_add);
  *log_ << "times (now) (time_since_token_update) (diff): " << now << " " << last_dequeue_time << " " << time_since_token_update << " " << tokens_to_add << " " << tokens << endl;
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
          tokens -= temp_tks;
          if (partial_quantum > temp_tks){
            // if fewer tokens were available than this queue's fair share, its remaining partial share must be adjusted when tokens become available next
            partial_quantum -= temp_tks;
          } else {
            // otherwise move to next queue
            q = (q + 1) % num_flows;  
            partial_quantum = PACKET_SIZE;
          }
        } else {
          // queue size smaller than fair share, *work conserving*
          queue_size_in_bytes_[q] = 0;
          tokens -= temp_tks;
          q = (q + 1) % num_flows;  
          partial_quantum = PACKET_SIZE;
        }
        break;
      } else { // move to next queue to be *work-conserving*
        q = (q + 1) % num_flows;  
        // if moving onto next queue, it must get opportunity to dequeue-adjust a full packet
        partial_quantum = PACKET_SIZE;
      }
    }
  }
  // dequeue a packet if actually available
  QueuedPacket ret = QueuedPacket("arbit", 0);
  if (not empty()) {
    *log_ << "Packet Dequeued." << endl;
    ret = std::move( DroppingPacketQueue::dequeue () );
  }
  return ret;
}


void PQP::enqueue(QueuedPacket&& p)
{
  uint16_t src, dst;
  _parse_ports_((const unsigned char *) p.contents.substr(24,4).c_str(), &src, &dst);
  if (queue_size_in_bytes_[src % num_flows] + p.contents.size() < limit && good_with( size_bytes() + p.contents.size(),
                        size_packets() + 1 )){
    queue_size_in_bytes_[src % num_flows] += p.contents.size();
    *log_ << "Enqueued packet for" << src << " in queue: " << src%num_flows << " size: " << queue_size_in_bytes_[src % num_flows] << " " <<p.contents.size() << endl;
    accept( std::move( p ) );
  } else {
    *log_ << "Dropped packet for" << src << " in queue: " << src%num_flows << " size: " << queue_size_in_bytes_[src % num_flows] << " " <<p.contents.size() << endl;
  }
  assert( good() );
}

