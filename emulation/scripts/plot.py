from plot_functions import *
import sys
import itertools
import configs

cc_exps = configs.cc_flows

qss = configs.queue_types
qsizes = configs.queue_sizes
rt = configs.rate
queue_exps = ["%s_%d%d" %(q, s, rt) for s in qsizes for q in qss]
print(queue_exps)

base_ports = {"cubic":16000, "bbr": 20000, "abc":24000}

ip = "100.64.0.2."
dir_res = configs.exp_dir
delays = []
all_ports = []
names = []
fairness = []
fairnesses = []
throughputs = []
drops = []
app_drops = []
dup_retransmits = []
retransmission_delays = []
rebuffering = []
buffer_avg = []

plots_dir = "%s/plots" %(dir_res)
if not os.path.exists(plots_dir):
   os.mkdir(plots_dir)

for cc in cc_exps:
    for q in queue_exps:
         print(cc, q)
         exp_dir = dir_res+"/"+q+"/"+cc.replace(" ","_").replace(",","_").replace(":","_").replace("-","")
         name = cc+" "+q.replace("_"," ")
         names.append(name)
         flow_bws, flow_delays, flow_ports, fdr, drops_all, pkts_all = parse_mahimahilog(exp_dir+"/mm.log")
         drops.append(((fdr[0] - fdr[1]) / fdr[1])*100)
         cum_flows, tm, mbps, _ = cummulate(flow_bws)
         throughputs.append(calculate_throughputs(cum_flows, time_c=3000))
         packet_drops, retransmissions, retransmit_delays = parse_pcap(exp_dir, "capture.pcap", flow_ports, time_c=3000)
         app_drops.append(sum([len(x) for x in packet_drops])/fdr[1]*100)
         dup_retransmits.append(sum([len(retransmissions[i]) - len(retransmit_delays[i]) for i in range(len(retransmissions))]))
         retransmission_delays.append(retransmit_delays)
         plot3(cum_flows, tm, mbps, [], [], [], [], fname="%s/throughput.png" %(exp_dir), labels=["Reno", "Cubic", "Vegas", "BBR"], lines=[":", "--", "-.", "-"], yheight=(0, 15), trange=(2, 22))
         plot_drops(packet_drops, fname="%s/tcpdump_drops.png" %(exp_dir), slow_start=0, trange=-1)
         fairness_ = plot_fairness(cum_flows, fname="%s/fairness.png" %(exp_dir))
         fairness.append(np.mean(fairness_))
         fairnesses.append(fairness_)
         rebuffering.append("None")
         buffer_avg.append("None")
         delays.append(flow_delays)
         all_ports.append(flow_ports)

i = 0

q_delays = []
rate = []
drops_q = []
dup_ret = []
labels_ = []
for cc in cc_exps:
    q_dels = []
    rt = []
    dr_q = []
    dret = []
    labels_.append(cc.split(" ")[0])
    for q in queue_exps:
        name = cc+" "+q.replace("_"," ")
        dds = []
        for j in range(len(delays[i])):
            dds += [d[1] for d in delays[i][j]]        
        q_dels.append(np.mean(dds))
        rt.append(throughputs[i][0])
        dr_q.append(drops[i])
        dret.append(dup_retransmits[i])
        i+=1
    droptail_r = rt[0]
    rt = [r/droptail_r for r in rt]
    print(q_dels)
    q_dels = [(187.5)*(q/1000) for q in q_dels]
    print(q_dels)
    q_delays.append(q_dels)
    rate.append(rt)
    drops_q.append(dr_q)
    dup_ret.append(dret)

names_ = []

for i in range(len(names)):
    dds = []
    for j in range(len(delays[i])):
        dds += [d[1] for d in delays[i][j]]
    print(names[i]," Delay: ", np.mean(dds))
    if rebuffering[i] != "None":
        print(names[i], "Avg Buffer: ", buffer_avg[i], "Rebuffering: ", rebuffering[i])
    print(names[i]," Fairness: ", fairness[i], ", Throughput: ", throughputs[i][0], ", Drop-Rate: ", drops[i])
    print(names[i], "App v Shaper drops: ", app_drops[i], "Duplicate Transmissions: ", dup_retransmits[i], "Retransmission Delay: ", np.mean(list(itertools.chain.from_iterable(retransmission_delays[i]))))
