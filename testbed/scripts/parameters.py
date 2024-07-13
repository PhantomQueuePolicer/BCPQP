import sys
import argparse



def main(args):
    args.rate = int(args.rate)
    args.maxrtt = int(args.maxrtt)
    args.ns = int(args.ns)
    args.nq = int(args.nq)
    rate_bytes = args.rate * 1000 / 8
    rtt_sec = args.maxrtt / 1000.0
    bdp = rate_bytes * rtt_sec

    reno_policer_size = 1500 * ((bdp / 1500)**2) / 18
    cubic_policer_size = (0.072 / rtt_sec) * (bdp) ** (4/3)
    bdp_2 = max(reno_policer_size, cubic_policer_size)

    bucket_size = 0.5 * (1500 / rate_bytes)
    num_buckets = (rtt_sec*2) / bucket_size
    bucket_size_us = bucket_size*10e6

    print("Shaper: docker run -v /var/run/machnet:/var/run/machnet ghcr.io/microsoft/machnet/machnet:latest release_build/src/apps/shaper/shaper --remote_ip1 %s --remote_ip2 %s --rate %d --qlen %d --num-shapers %d --num-queues %d --num-buckets %d --bucket-len %d\n" %(args.sender, args.receiver, args.rate, bdp*2, args.ns, args.nq, num_buckets, bucket_size_us))
    print("Policer (BDP): docker run -v /var/run/machnet:/var/run/machnet ghcr.io/microsoft/machnet/machnet:latest release_build/src/apps/policer/policer --remote_ip1 %s --remote_ip2 %s --rate %d --qlen %d --num-shapers %d\n" %(args.sender, args.receiver, args.rate, bdp*2, args.ns))
    print("Policer (BDP^2): docker run -v /var/run/machnet:/var/run/machnet ghcr.io/microsoft/machnet/machnet:latest release_build/src/apps/policer/policer --remote_ip1 %s --remote_ip2 %s --rate %d --qlen %d --num-shapers %d\n" %(args.sender, args.receiver, args.rate, bdp_2, args.ns))
    print("FairPolicer: docker run -v /var/run/machnet:/var/run/machnet ghcr.io/microsoft/machnet/machnet:latest release_build/src/apps/fairpolicer/fairpolicer --remote_ip1 %s --remote_ip2 %s --rate %d --qlen %d --num-shapers %d --num-queues %d\n" %(args.sender, args.receiver, args.rate, bdp_2, args.ns, args.nq))
    print("BCPQP: docker run -v /var/run/machnet:/var/run/machnet ghcr.io/microsoft/machnet/machnet:latest release_build/src/apps/bcpqp/bcpqp --remote_ip1 %s --remote_ip2 %s --rate %d --qlen %d --num-shapers %d --num-queues %d\n" %(args.sender, args.receiver, args.rate, 10*bdp_2, args.ns, args.nq))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rate', metavar='rate',
                        required=True, help='Rate in Kbps')
    parser.add_argument('--maxrtt', metavar='maxrtt',
                        required=True, help='Max RTT (ms)')
    parser.add_argument('--nq', metavar='nq',
                        required=True, help='Number of queues')
    parser.add_argument('--ns', metavar='ns',
                        required=True, help='Number of shapers/policers')
    parser.add_argument('--sender', metavar='sender',
                        required=True, help='Sender IP')
    parser.add_argument('--receiver', metavar='receiver',
                        required=True, help='Receiver IP')
    args = parser.parse_args()

    main(args)