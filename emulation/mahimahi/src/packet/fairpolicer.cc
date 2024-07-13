#include <math.h>
#include "fairpolicer.hh"
#include "timestamp.hh"
#include "infinite_packet_queue.hh"

using namespace std;


FairPolicer::FairPolicer( const string & args )
  : DroppingPacketQueue(args),
    burst ( get_arg( args, "burst") ),
    rate ( get_arg( args, "rate") ),
    num_flows ( get_arg( args, "flows") ),
    last_dequeue_time (timestamp()),
    rate_in_bytes(1024.0 * rate / 8.0),
    buckets ({}),
    log_()
{
  if ( burst == 0 || rate == 0 ) {
    throw runtime_error( "Fair Policer must have burst and rate (Kbit/s) arguments." );
  }

  for (uint32_t i=0; i<num_flows; i++){
    buckets.push_back(0);
  }

  // logfile for tbf
  const string logfile = "fairpolicer.log";
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

QueuedPacket FairPolicer::dequeue( void )
{ 
  // update number of tokens
  uint64_t now = timestamp();
  float time_since_token_update = float(now - last_dequeue_time) / 1000;
  int32_t tokens_to_add = time_since_token_update * rate_in_bytes;
  tokens = std::min(burst, tokens + tokens_to_add);
//   *log_ << "times (now) (time_since_token_update) (diff): " << now << " " << last_dequeue_time << " " << time_since_token_update << " " << tokens_to_add << " " << tokens << endl;
  last_dequeue_time = now;
  // dequeue packet if there is one
  // quantum-based round robin to distribute tokens between queues in a work-conserving manner.
  while (true){
    if (tokens <= 0){
      break;
    } 
    bool empty_queue = true;
    for (uint32_t i=0; i<num_flows; i++){
      empty_queue &= (buckets[i] == 0);
    }
    if (empty_queue){
      break;
    }
    while(true){ // to find a non-empty queue
      if (buckets[q] > 0){ // found a non-empty queue
        int32_t temp_tks = std::min(tokens, partial_quantum);
        if (buckets[q] > temp_tks){
          // adjust queue size and remaining tokens
          buckets[q] -= temp_tks;
          tokens -= temp_tks;
          combined_queue_size -= temp_tks;
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
          buckets[q] = 0;
          tokens -= temp_tks;
          combined_queue_size -= temp_tks;
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
  QueuedPacket ret = QueuedPacket("arbit", 0);
  if (not empty()) {
    *log_ << "Dequeued: " << size_bytes() << endl;
    ret = std::move( DroppingPacketQueue::dequeue () );
  }
  return ret;
}


void FairPolicer::enqueue(QueuedPacket&& p)
{
  uint16_t src, dst;
  int pkt_len = (int) p.contents.size();
  int32_t threshold = burst - combined_queue_size;
  _parse_ports_fpl((const unsigned char *) p.contents.substr(24,4).c_str(), &dst, &src);
  if (buckets[src % num_flows] + pkt_len < threshold){
    buckets[src % num_flows] += pkt_len;
    combined_queue_size += pkt_len;
    *log_ << "Enqueued " << pkt_len <<" "<< src << " " << buckets[src % num_flows] << " " << threshold << endl;
    accept( std::move( p ) );
  } else {
    *log_ << "Dropped " << pkt_len <<" "<< src << " " << buckets[src % num_flows] << " " << threshold << endl;
  }
  assert( good() );
}

