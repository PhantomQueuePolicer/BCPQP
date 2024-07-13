/* -*-mode:c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*- */

#ifndef PQP_HH
#define PQP_HH

#include <random>
#include <queue>
#include <string>
#include <fstream>
#include <memory>
#include "dropping_packet_queue.hh"
#include "abstract_packet_queue.hh"

inline void _parse_ports_( const unsigned char *s, uint16_t *src, uint16_t *dst ) {
        *src = (s[0] << 8) | s[1];
        *dst = (s[2] << 8) | s[3];
        }

class PQP : public DroppingPacketQueue
{
private:
    const static unsigned int PACKET_SIZE = 1504;
    //Configuration parameters
    uint32_t burst;
    uint32_t limit;
    float rate;
    uint32_t num_flows;
    uint32_t quantum;

    //State variables
    uint32_t tokens = 0;
    std::vector<uint32_t> queue_size_in_bytes_;
    uint64_t last_dequeue_time;
    uint32_t partial_quantum = PACKET_SIZE;
    uint32_t q = 0;
    float rate_in_bytes;

    std::unique_ptr<std::ofstream> log_;


    virtual const std::string & type( void ) const override
    {
        static const std::string type_ { "pqp" };
        return type_;
    }


public:
    PQP( const std::string & args );

    void enqueue( QueuedPacket && p ) override;

    QueuedPacket dequeue( void ) override;
};

#endif /* PQP_HH */ 
