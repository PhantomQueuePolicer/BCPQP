#include <math.h>
#include "shaper.hh"
#include "timestamp.hh"
#include "infinite_packet_queue.hh"
#include "drop_tail_packet_queue.hh"
#include "drop_head_packet_queue.hh"
#include "codel_packet_queue.hh"
#include "pie_packet_queue.hh"

using namespace std;

AbstractPacketQueue* make_packet_queue( const uint32_t & type, uint32_t limit)
{
    if (limit == 0) {
      cerr << "Warning: limit(size of tbf queue in bytes) should be given (except for Infinite Queue)" << endl;
    }
    AbstractPacketQueue* pkt_q;
    const string bytes = std::to_string(limit);
    if ( type == 1 ) {
        pkt_q = new InfinitePacketQueue( "" );
        return pkt_q;
    } else if ( type == 3 ) {
        pkt_q = new DropTailPacketQueue( "bytes="+bytes );
        return pkt_q;
    } else if ( type == 2 ) {
        pkt_q = new DropHeadPacketQueue( "bytes="+bytes );
        return pkt_q;
    } else if ( type == 4 ) {
        pkt_q = new CODELPacketQueue( "bytes="+bytes+",target=5,interval=100" ) ;
        return pkt_q;
    } else if ( type == 5 ) {
        pkt_q = new PIEPacketQueue( "bytes="+bytes+",qdelay_ref=15,max_burst=150" );
        return pkt_q;
    } else {
        cerr << "Unknown queue type: " << type << endl;
        throw runtime_error("Available Queues: 1(Infinite), 2(Droptail), 3(DropHead), 4(CoDel), 5(Pie)");
    }

    return nullptr;
}

Shaper::Shaper( const string & args )
  : DroppingPacketQueue(args),
    burst ( get_arg( args, "burst") ),
    limit ( get_arg( args, "limit") ),
    rate ( get_arg( args, "rate") ),
    num_flows ( get_arg( args, "flows") ),
    queuetype ( get_arg(args, "queue")),
    last_dequeue_time (timestamp()),
    rate_in_bytes(1024.0 * rate / 8.0),
    log_(),
    deficits({}),
    packet_queues({})
{
  if ( burst == 0 || limit == 0 || rate ==0 ) {
    throw runtime_error( "Shaper must have burst, rate (Kbit/s) and limit arguments." );
  }

  // logfile for tbf
  const string logfile = "shaper.log";
  if ( not logfile.empty() ) {
      log_.reset( new ofstream( logfile ) );
      if ( not log_->good() ) {
          throw runtime_error( logfile + ": error opening for writing" );
      }
      *log_ << "# init timestamp: " << initial_timestamp() << endl;
      *log_ << "# rate in bytes per second: " << rate_in_bytes << endl;
  }
  // internal queue
  for (uint32_t i=0; i<num_flows; i++){
    AbstractPacketQueue* packet_queue_ = std::move(make_packet_queue(queuetype, limit));
    packet_queues.push_back(packet_queue_);
    deficits.push_back(0);
  }
}

QueuedPacket Shaper::dequeue( void )
{ 
  // update number of tokens
  uint64_t now = timestamp();
  float time_since_token_update = float(now - last_dequeue_time) / 1000;
  uint32_t tokens_to_add = time_since_token_update * rate_in_bytes;
  tokens = std::min(burst, tokens + tokens_to_add);
  last_dequeue_time = now;
  // first put packets based on number of tokens in the link queue and then dequeue
  QueuedPacket ret = QueuedPacket("arbit", 0);
  if (not empty()) {
    ret = std::move( DroppingPacketQueue::dequeue () );
    *log_ << now << " Dequeued " <<ret.contents.size() << endl;
  }
  while (true){
    if (tokens < 500){
      break;
    } 
    bool empty_queue = true;
    for (uint32_t i=0; i<num_flows; i++){
      empty_queue &= (packet_queues[i]->empty());
    }
    if (empty_queue){
      break;
    }
    *log_ << now << " Packets to be Dequeued " <<ret.contents.size() << endl;
    bool notenoughtokens = false;
    while(true){ // to find a non-empty queue
      if (tokens < 500){
        break;
      }
      if (!packet_queues[q]->empty()){ // found a non-empty queue
        deficits[q] += 500;
        tokens -= 500;
        if (packet_queues[q]->front().contents.size() < deficits[q]){
        //   *log_ << now << " not enough tokens " <<packet_queues[q]->front().contents.size()<<" "<<tokens << endl;
        //   notenoughtokens = true;
        //   break;
        // } else {
          *log_ << now << " Shaper dequeue " <<packet_queues[q]->front().contents.size() << endl;
          QueuedPacket p = std::move( packet_queues[q]->front() );
          packet_queues[q]->dequeue();
          deficits[q] -= p.contents.size();
          doenqueue( std::move(p) );
          q = (q + 1) % num_flows;  
        }
      } else { // move to next queue to be *work-conserving*
        *log_ << now << " Next dequeue " << endl;
        q = (q + 1) % num_flows;  
        // if moving onto next queue, it must get opportunity to dequeue-adjust a full packet
      }
      bool empty_queue = true;
      for (uint32_t i=0; i<num_flows; i++){
        empty_queue &= (packet_queues[i]->empty());
      }
      if (empty_queue){
        break;
      }
    }
    if (notenoughtokens){
      break;
    }
  }
  return ret;
}


void Shaper::doenqueue( QueuedPacket && p )
{
  if ( good_with( size_bytes() + p.contents.size(),
		  size_packets() + 1 ) ) {
    accept( std::move( p ) );
    *log_ << "packet enqueued: " << size_bytes() << " " << size_packets() << endl;
  }
  assert( good() );
}


void Shaper::enqueue(QueuedPacket&& p)
{
  uint16_t src, dst;
  std::string src_ip, dst_ip;
  _parse_ports_8((const unsigned char *) p.contents.substr(24,4).c_str(), &src, &dst);
  _parse_ip_5((const unsigned char *) p.contents.substr(16,8).c_str(), &src_ip, &dst_ip);
  *log_<< "Paket header "<< src_ip << " "<< dst_ip<<endl;
  uint16_t q_i = src%num_flows;
  
  packet_queues[q_i]->enqueue(std::move(p));
}

