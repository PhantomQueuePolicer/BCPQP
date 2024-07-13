/* -*-mode:c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*- */

#ifndef SHAPER_HH
#define SHAPER_HH

#include <random>
#include <queue>
#include <string>
#include <fstream>
#include <memory>
#include "dropping_packet_queue.hh"
#include "abstract_packet_queue.hh"

inline void _parse_ports_8( const unsigned char *s, uint16_t *src, uint16_t *dst ) {
    *src = (s[0] << 8) | s[1];
    *dst = (s[2] << 8) | s[3];
}

inline void _parse_ip_5( const unsigned char *s, std::string *src, std::string *dst) {
    *src = std::to_string(s[0]) + "." + std::to_string(s[1]) + "." + std::to_string(s[2]) + "."+std::to_string(s[3]);
    *dst = std::to_string(s[4]) + "." + std::to_string(s[5]) + "." + std::to_string(s[6]) + "."+std::to_string(s[7]);
}

class Shaper : public DroppingPacketQueue
{
private:
    const static unsigned int PACKET_SIZE = 1504;
    //Configuration parameters
    uint32_t burst;
    uint32_t limit;
    float rate;
    uint32_t num_flows;
    uint32_t queuetype;
    uint16_t q =0;

    //State variables
    uint32_t tokens = 0;
    int queue_size_in_bytes_ = 0;
    uint64_t last_dequeue_time;
    float rate_in_bytes;

    std::queue<QueuedPacket> internal_queue_tbf {};
    std::unique_ptr<std::ofstream> log_;
    std::vector<uint32_t> deficits;
    std::vector<AbstractPacketQueue*> packet_queues;

    void doenqueue( QueuedPacket && p );


    virtual const std::string & type( void ) const override
    {
        static const std::string type_ { "shp" };
        return type_;
    }


public:
    Shaper( const std::string & args );

    void enqueue( QueuedPacket && p ) override;

    QueuedPacket dequeue( void ) override;
};

#endif /* SHAPER_HH */ 
