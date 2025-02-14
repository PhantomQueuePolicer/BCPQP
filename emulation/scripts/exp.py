import os
import time
import configs
import sys
import json 
from traffic_generator import *

cc_exps = configs.cc_flows

data_to_send = configs.data_to_send
data_to_send_v = [600000, 12000000, 10000, 600000, 600000, 600000, 20000, 10000, 600000, 600000]
start_times = configs.start_times
periods = configs.periods
repeats = configs.repeats
burst_sizes = [5, 3]
config_rate = configs.rate
config_queuetype = configs.queue_types
config_rtt = configs.RTT_ms
config_queuesize = configs.queue_sizes
config_burtsize = configs.burst

queue_exps = [sys.argv[1]] # bytes
results_dir = sys.argv[1]

os.makedirs(results_dir, exist_ok=True)

def log_configs(file_name):
    configs = {
        "cc_flows": cc_exps, 
        "data_to_send": data_to_send, 
        "repeats": repeats, 
        "periods": periods, 
        "rate": config_rate, 
        "queue_type": config_queuetype, 
        "rtt": config_rtt,
        "queue_size": config_queuesize, 
        "burst/bucket": config_burtsize,
    }
    with open(file_name, "w") as file:
        json.dump(configs, file, indent=4)
        

base_ports = {"cubic":16000, "bbr": 20000, "abc": 24000, "reno": 26000, "vegas": 30000}
pt = 0
for cc in cc_exps:
    for q in queue_exps:
        exp_name = cc.replace(" ","_").replace(",","_").replace("-","").replace(":","_")
        config_filename = f"{results_dir}/{exp_name}/configs.json"
        if not os.path.exists("%s/%s" %(results_dir, exp_name)):
            os.mkdir("%s/%s" %(results_dir, exp_name))
        cc_types = cc.split(", ")
        congestions = []
        ports = []
        schedule = []
        i = 0
        for cc_n in cc_types:
            cc_, n = cc_n.split(" ")
            if ":" in cc_:
                cc_ = cc_.split(":")[1]
            for j in range(int(n)):
                if "video" in cc_n:
                    schedule.append((i, start_times[i], data_to_send_v[i], periods[i], repeats[i]))
                else:
                    schedule.append((i, start_times[i], data_to_send[i], periods[i], repeats[i]))
                congestions.append(cc_)
                ports.append(base_ports[cc_]+pt)
                pt+=1
                i+=1
        if "b-" in cc:
            cfg = Config(congestions, ports, schedule, bursts=burst_sizes)
        else:
            cfg = Config(congestions, ports, schedule, bursts=[1,1])
        if "abc" in cc:
            os.system("sudo sysctl -w net.ipv4.tcp_ecn=1")
        tg = TrafficGenerator(cfg, tcpdump_fname="%s/%s/reciever_capture.pcap" %(results_dir, exp_name), tcpdump_client_fname="%s/%s/sender_capture.pcap" %(results_dir, exp_name), tcpdump=True)
        tg.run()
        os.system("cp mm.log %s/%s/mm.log" %(results_dir, exp_name))
        os.system("cp phantom_log.log %s/%s/phantom.log" %(results_dir, exp_name))
        time.sleep(5)
        tg.report_metrics("%s/%s/server.log" %(results_dir, exp_name))
        tg.close()
        os.system("cat /dev/null > mm.log")
        log_configs(config_filename)        

print("done")
