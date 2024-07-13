/* -*-mode:c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*- */

#ifndef FAIRPOLICER_HH
#define FAIRPOLICER_HH

#include <random>
#include <queue>
#include <string>
#include <fstream>
#include <memory>
#include "dropping_packet_queue.hh"
#include "abstract_packet_queue.hh"


inline void _parse_ports_fpl( const unsigned char *s, uint16_t *src, uint16_t *dst ) {
        *src = (s[0] << 8) | s[1];
        *dst = (s[2] << 8) | s[3];
        }

class FairPolicer : public DroppingPacketQueue
{
private:
    const static unsigned int PACKET_SIZE = 1504;
    //Configuration parameters
    int32_t burst;
    float rate;
    uint16_t num_flows;

    //State variables
    int32_t tokens = 0;
    uint64_t last_dequeue_time;
    float rate_in_bytes;
    uint16_t active_flows = 0;
    int32_t combined_queue_size = 0;
    int32_t partial_quantum = PACKET_SIZE;
    uint32_t q = 0;
    std::vector<int32_t> buckets;

    std::unique_ptr<std::ofstream> log_;


    virtual const std::string & type( void ) const override
    {
        static const std::string type_ { "fpl" };
        return type_;
    }


public:
    FairPolicer( const std::string & args );

    void enqueue( QueuedPacket && p ) override;

    QueuedPacket dequeue( void ) override;
};

#endif /* FAIRPOLICER_HH */ 
