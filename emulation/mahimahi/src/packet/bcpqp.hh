/* -*-mode:c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*- */

#ifndef BCPQP_HH
#define BCPQP_HH

#include <random>
#include <queue>
#include <string>
#include <fstream>
#include <memory>
#include "dropping_packet_queue.hh"
#include "abstract_packet_queue.hh"

inline void _parse_ports_4( const unsigned char *s, uint16_t *src, uint16_t *dst ) {
    *src = (s[0] << 8) | s[1];
    *dst = (s[2] << 8) | s[3];
}

inline void _parse_ip_4( const unsigned char *s, std::string *src, std::string *dst) {
    *src = std::to_string(s[0]) + "." + std::to_string(s[1]) + "." + std::to_string(s[2]) + "."+std::to_string(s[3]);
    *dst = std::to_string(s[4]) + "." + std::to_string(s[5]) + "." + std::to_string(s[6]) + "."+std::to_string(s[7]);
}

class BCPQP : public DroppingPacketQueue
{
private:
    const static unsigned int PACKET_SIZE = 1504;
    //Configuration parameters
    uint32_t burst;
    float rate;
    float pacing_rate;
    uint32_t num_flows;
    uint32_t quantum;
    uint32_t limit;
    uint64_t rtt;

    //State variables
    std::vector<uint32_t> last_limits;
    uint32_t tokens = 0;
    std::vector<uint32_t> queue_size_in_bytes_;
    std::vector<uint32_t> dequeues_in_last_rtt;
    std::vector<uint32_t> magic_packets;
    std::vector<float> queue_drain_rates;
    uint64_t last_dequeue_time;
    uint64_t last_grad_compute_time;
    uint32_t partial_quantum = PACKET_SIZE;
    uint32_t q = 0;
    float rate_in_bytes;
    float gradient_threshold;

    std::unique_ptr<std::ofstream> log_;

    // pacer
    std::vector<QueuedPacket> timerwheel;
    float transmission_opportunity = 0;
    uint64_t reference_timestamp = 0;
    uint32_t w_index = 0;
    uint32_t r_index = 0;
    uint32_t timerwheel_size = 0;




    virtual const std::string & type( void ) const override
    {
        static const std::string type_ { "bcp" };
        return type_;
    }


public:
    BCPQP( const std::string & args );

    void enqueue( QueuedPacket && p ) override;

    QueuedPacket dequeue( void ) override;
};

#endif /* BCPQP_HH */ 
