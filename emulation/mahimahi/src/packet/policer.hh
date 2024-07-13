/* -*-mode:c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*- */

#ifndef POLICER_HH
#define POLICER_HH

#include <random>
#include <queue>
#include <string>
#include <fstream>
#include <memory>
#include "dropping_packet_queue.hh"
#include "abstract_packet_queue.hh"


class Policer : public DroppingPacketQueue
{
private:
    const static unsigned int PACKET_SIZE = 1504;
    //Configuration parameters
    uint32_t burst;
    float rate;

    //State variables
    uint32_t tokens = 0;
    uint64_t last_dequeue_time;
    float rate_in_bytes;

    std::unique_ptr<std::ofstream> log_;


    virtual const std::string & type( void ) const override
    {
        static const std::string type_ { "plc" };
        return type_;
    }


public:
    Policer( const std::string & args );

    void enqueue( QueuedPacket && p ) override;

    QueuedPacket dequeue( void ) override;
};

#endif /* POLICER_HH */ 
