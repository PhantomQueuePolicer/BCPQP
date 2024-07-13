#include <math.h>
#include "policer.hh"
#include "timestamp.hh"
#include "infinite_packet_queue.hh"

using namespace std;


Policer::Policer( const string & args )
  : DroppingPacketQueue(args),
    burst ( get_arg( args, "burst") ),
    rate ( get_arg( args, "rate") ),
    last_dequeue_time (timestamp()),
    rate_in_bytes(1024.0 * rate / 8.0),
    log_()
{
  if ( burst == 0 || rate ==0 ) {
    throw runtime_error( "Policer must have burst and rate (Kbit/s) arguments." );
  }

  // logfile for tbf
  const string logfile = "policer.log";
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

QueuedPacket Policer::dequeue( void )
{ 
  // update number of tokens
  uint64_t now = timestamp();
  float time_since_token_update = float(now - last_dequeue_time) / 1000;
  uint32_t tokens_to_add = time_since_token_update * rate_in_bytes;
  tokens = std::min(burst, tokens + tokens_to_add);
  *log_ << "times (now) (time_since_token_update) (diff): " << now << " " << last_dequeue_time << " " << time_since_token_update << " " << tokens_to_add << " " << tokens << endl;
  last_dequeue_time = now;
  // dequeue packet if there is one
  QueuedPacket ret = QueuedPacket("arbit", 0);
  if (not empty()) {
    *log_ << "Dequeued: " << size_bytes() << endl;
    ret = std::move( DroppingPacketQueue::dequeue () );
  }
  return ret;
}


void Policer::enqueue(QueuedPacket&& p)
{
  if (p.contents.size() < tokens){
    tokens -= p.contents.size();
    *log_ << "Enqueued " <<p.contents.size() << endl;
    accept( std::move( p ) );
  } else {
    *log_ << "Dropped " << p.contents.size() << endl;
  }
  assert( good() );
}

