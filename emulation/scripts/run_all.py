import sys
import os
import subprocess
import configs

exp_dir = configs.exp_dir
# topology
# server --> traffic shapper: 
mm_before_shaper_string = "mm-delay 10 mm-link ../mahimahi/traces/1.up ../mahimahi/traces/1.up --uplink-log=mm.log --uplink-queue="
mm_after_shaper_string = "mm-delay 45"
mm_loss = ""
exp_string = "python3 exp.py"

queue_types = configs.queue_types
queue_sizes = [480000, 480000, 480000, 480000, 480000, 480000, 480000]
vqueue_sizes = configs.queue_sizes
lqueue_sizes = configs.lower_queue_sizes
limit_refresh = configs.limit_refresh
rate = configs.rate
pacing_rate = configs.pacing_rate
burst = configs.burst
num_flows = configs.num_queues

tbf_map = {"droptail": 3, "codel": 4, "pie": 5}

for queue in queue_types:
    internal_queue = 0 # won't put any internal queue
    queue_ = queue
    if "_" in queue:
        internal_queue = tbf_map[queue.split("_")[1]]
        queue_ = queue.split("_")[0]
    for i in range(len(vqueue_sizes)):
        s = queue_sizes[i]
        vs = vqueue_sizes[i]
        exp_name = exp_dir+"/%s_%d%d" %(queue, vs, rate)
        while True:
            mm_string = "%s%s --uplink-queue-args=\"bytes=%d,burst=%d,limit=%d,rate=%d,pacing=%d,queue=%d,flows=%d,quantum=1504,eeta=100,lowerlimit=%d,upperlimit=%d,resetlimit=%d\" %s %s %s %s" %(mm_before_shaper_string, queue_, s, burst, vs, rate, pacing_rate, internal_queue, num_flows, lqueue_sizes[i], vqueue_sizes[i], limit_refresh, mm_after_shaper_string, mm_loss, exp_string, exp_name)
            proc = subprocess.Popen(mm_string, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)

            (out, err) = proc.communicate()

            if "done" in str(out):
                print(queue, s)
                print(str(out))
                break
            else:
                print(queue, s)
                for l in str(err).split("\\n"):
                    print (l)