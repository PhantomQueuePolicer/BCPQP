
# exp logistics config

# end point config
cc_flows = ["reno 1, cubic 1, vegas 1, bbr 1", "reno 4"] # CC + number of flows
data_to_send = [10 * 1000 * 1000, 10 * 1000 * 1000, 10 * 1000 * 1000, 10 * 1000 * 1000] # bytes
repeats = [1, 1, 1, 1] # number of times each flow repeats
periods = [15, 15, 15, 15] # time between each repitition in seconds
start_times = [0, 0, 0, 0] # start time for first time in seconds

# rate enforcement mechanism
queue_types = ["shp"]
# options:
    # Shaper: shp
    # Phantom Queue Policer: pqp
    # BC-PQP: bcp
    # Policer: plc
    # FairPolicer: fpl

# rate enforcement configurations
queue_sizes = [300000] # size in bytes
num_queues = 5 # number of per-flow queues in shaper
limit_refresh = 120 # set to max or p99 RTT
rate = 7500 # kbit/s
pacing_rate = 20000 # kbit/s (secondary bottleneck rate, keep larger than rate)
burst = 3000 # bytes
lower_queue_sizes = [87000] # TODO: remove this if not used anymore

# logs directory
exp_dir = "fairness"
