import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# import matplotlib.pyplot as plt
sns.set()
sns.palplot(sns.color_palette("cubehelix", 5))
from scipy.interpolate import make_interp_spline, BSpline
import numpy as np
import os
# sns.set(rc={'axes.facecolor':'white', 'figure.facecolor':'white'})
sns.set_style("whitegrid")

# plt.rcParams['text.usetex'] = True
SMALL_SIZE = 13
MEDIUM_SIZE = 16
BIGGER_SIZE = 18

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
# plt.rcParams['ps.fonttype'] = 42

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

td = 200

def bw_cap(time_s, file="", ud = (1,10)):
  if file == "":
    if ud == 1:
      return [1500*1000*8/(1024*1024)]*time_s
    else:
      _, down = ud
      bytes_s = (1000 / down) * 2 * 8 * 1500
      return [bytes_s/(1024*1024)]*time_s
  else:
    f = open(file, 'r')
    data = f.read().split("\n")[:-1]
    times = [int(d) for d in data]
    ret_times = [d for d in times]
    last_s = ret_times[-1]
    while (last_s/1000) < time_s:
      new_times = [d+last_s for d in times]
      ret_times += new_times
      last_s = ret_times[-1]
    MTU = 1500
    bw_s = [0]*(int(last_s/1000) + 1)
    for t in ret_times:
      t_s = int(t/1000)
      bw_s[t_s] += MTU
    mbps = [(d*8)/(1024*1024) for d in bw_s]
    return mbps
# print(bw_cap(100, ud = 1))
# print(bw_cap(15, ud = (1,11)))
# bw_cap(330, file="Verizon-LTE-short.down")

def get_schedule(tup, stats):
  starts = [stats[i][0] for i in tup]
  period = [stats[i][2] for i in tup]
  repeat = [stats[i][3] for i in tup]
  schedule = []
  for i in tup:
    schedule += [(i, starts[i] + j*period[i]) for j in range(repeat[i])]
  return sorted(schedule, key=lambda x:x[1])

def parse_video_server_log(file, merge=[]):
  f = open(file, 'r')
  data = f.read()
  data = data.split('(')[1:]
  stats = []
  stats_ = []
  chunks = []
  periods = []
  for d in data:
    flow_stats = [f for f in d.split("\n") if "start" in f][0]
    stats.append(flow_stats)
    flow_stats = flow_stats.split(": ")[1:]
    start, data_size, period, repeat = [int(f.split(", ")[0]) for f in flow_stats]
    stats_.append((start, data_size, period, repeat))
    chunk_downloads = [float(f.split("downloaded: ")[1].split(" ")[0]) for f in d.split("\n") if "Chunk" in f]
    chunks.append(chunk_downloads)
    periods.append(period)
  if merge != []:
    for tup in merge:
      merged_flow = []
      schedule = get_schedule(tup, stats_)
      for f, _ in schedule:
        if len(merged_flow) == 0 or chunks[f][0] > merged_flow[-1]:
          merged_flow.append(chunks[f][0])
        else:
          merged_flow.append(merged_flow[-1])
        del chunks[f][0]
      chunks.append(merged_flow)
      periods.append(periods[0]/len(tup))
    for tup in merge:
      for t in tup[::-1]:
        del chunks[t]
        del periods[t]
  return flow_stats, chunks, periods

def netflix_log(file, ts=36.5):
  f = open(file, 'r')
  data = f.read()
  data = data.split('\n')
  data = [d for d in data if d!=""]

  times = [(ts+float(d.split(" at ")[0]))*(1000/td) for d in data]
  playing_br = [d.split("x")[1].split("),")[0] for d in data]
  playing_br = intify_nf(playing_br)
  buffer_size = [float(d.split("/ ")[2].split(" at")[0]) for d in data]
  qt = (playing_br, times)
  print(qt[:5])
  return qt, [], times[-1], 0

def plot_queue_sizes(files, trange=-1, fname="", labels=[], yrange=-1):
  i = 0
  plt.figure(figsize=(4,3))
  colors = ["red", "green", "royalblue", "red", "purple", "red"]
  lines = ["-.", "--", ":"]
  for file in files:
    f = open(file, 'r')
    data = f.read()
    data = data.split('\n')[5:]
    times = [int(d.split(" ")[0])/1000 for d in data if d!="" and "packet" not in d]
    queues = [int(d.split(" ")[2])/1000 for d in data if d!="" and "packet" not in d]
    plt.plot(times, queues, color=colors[i+1], label=labels[i], linestyle = lines[i])
    i += 1
  plt.ylabel("Queue Buildup (KB)")
  plt.xlabel("Time (s)")
  x1,x2,y1,y2 = plt.axis()
  if trange!=-1:
    x1, x2 = trange[0], trange[1]
  if yrange!=-1:
    y1, y2 = yrange[0], yrange[1]
    # y1 = np.interp(x1, times, drops) - 5
    # y2 = np.interp(x2, times, drops) + 5
  plt.axis([x1,x2,0,y2+0.1*y2])
  plt.legend()
  # plt.title(title)
  if fname != "":
    plt.savefig(fname, bbox_inches="tight")


def parse_downlink_log(file):
  f = open(file, 'r')
  data = f.read()
  data = data.split('\n')
  timestamp = int(data[3].split(' ')[-1])
  # print(timestamp)
  data = [d.split(' ') for d in data][:-1]
  downlink_bytes = [0] * int(int(data[-1][0]) / td)
  for row in data:
    if row[1] == '-':
      sec = int(int(row[0])/td)
      # if sec < len(downlink_bytes):
      downlink_bytes[sec] += int(row[2])
      # else:
      #   downlink_bytes.append(int(row[2]))
  down_mbps = [float(d*8)*(1000/td) / (1024*1024) for d in downlink_bytes]
  # print(len(down_mbps))
  return down_mbps, timestamp

def convert_to_ms(ts):
  hr, min, sec = ts.split(':')
  try:
    time_s = float(hr)*60*60 + float(min)*60 + float(sec)
  except:
    print(ts)
    sec = sec.replace("IP","")
    time_s = float(hr)*60*60 + float(min)*60 + float(sec[:9])
  return time_s*1000


def parse_mahimahilog(file, td=200):
  f = open(file, 'r')
  data = f.read()
  data = data.split('\n')
  flows = {}
  enq_data = [d for d in data if " + " in d]
  deq_data = [d for d in data if " - " in d]
  drop_data = [d for d in data if " d " in d]
  print(len(deq_data), len(drop_data) - len(deq_data), len(enq_data) - len(deq_data))
# 53 - 80 2 44600 16775
  for d in deq_data:
    ts, d = d.split(" - ")
    bytes_, delay, src, dst = d.split(" ")
    try:
      ts, bytes_, delay, src, dst = int(ts), int(bytes_), int(delay), int(src), int(dst)
      # if src >=16000 or src <=16100 or src >= 20000 or src <= 20100:
      if src not in flows:
        flows[src] = []
      flows[src].append((ts, bytes_, delay))
    except:
      continue
  all_drops = []
  all_pkts = []
  for d in drop_data:
    ts, d = d.split(" d ")
    pkts, bytes_ = d.split(" ")
    all_drops.append((int(ts), int(pkts)))

  for d in deq_data:
    ts, d = d.split(" - ")
    # bytes_, _, _, _ = d.split(" ")
    all_pkts.append((int(ts), 1))

  flow_bws, flow_delays, flow_ports = get_mm_throughput_delay(flows, td)
  deq_size = 0
  enq_size = 0
  for p in flow_ports:
    deq_size += len([d for d in deq_data if str(p) in d])
    enq_size += len([d for d in enq_data if str(p) in d])
  return flow_bws, flow_delays, flow_ports, (enq_size , deq_size), all_drops, all_pkts

def get_mm_throughput_delay(flows, td):
  flow_bws = []
  flow_ports = []
  flow_delays = []
  for f in flows.keys():
    print(f, len(flows[f]))
    delays = [(o[0], o[2]) for o in flows[f]]
    abs_time = [o[0] for o in flows[f]]
    time = [t-abs_time[0] for t in abs_time]
    byte = [o[1] for o in flows[f]]
    if sum(byte) < 10000:
      continue
    bytes_recv = [0]* (int(int(time[-1])/td)+1)
    for t,b in zip(time, byte):
      t = int(int(t)/td)
      try:
        bytes_recv[t] += b
      except:
        print(t, time[-1])
    mbps = [float(d*8)*(1000/td)/(1024*1024) for d in bytes_recv]
    flow_bws.append((mbps, abs_time[0], abs_time[-1]))
    flow_delays.append(delays)
    flow_ports.append(f)
  return flow_bws, flow_delays, flow_ports

def parse_pcap(directory, file, ports, time_c=0):
  if not os.path.exists("%s/tcpdump.log" %(directory)):
    os.system("tcpdump -r %s/%s > %s/tcpdump.log" %(directory, file, directory))
  f = open("%s/tcpdump.log" %(directory), 'r')
  data = f.read()
  data = data.split("\n")
  data = [d for d in data if "IP" in d]
  abs_time_stamp = convert_to_ms(data[0].split(" ")[0])
  packet_drop_times = []
  retransmit_sequeunces = []
  retransmission_delays = []
  for p in ports:
    if p == 20011:
      data_p = [d for d in data if ".isdnlog >" in d]
    else:
      data_p = [d for d in data if ".%d >" %(p) in d]
    data_p = [d for d in data_p if " seq " in d]
    data_p = [d for d in data_p if " ack " in d]
    data_p = [d for d in data_p if " length 0" not in d]
    times = [convert_to_ms(l.split(" ")[0])-abs_time_stamp for l in data_p]
    # seq = [int(l.split("seq ")[1].split(":")[1].split(",")[0]) for l in data_p]
    seq = [int(int(l.split("seq ")[1].split(":")[1].split(",")[0])/1448) for l in data_p]
    seq_debug = [int(l.split("seq ")[1].split(":")[1].split(",")[0]) for l in data_p]
    lengths = [int(int(l.split("length ")[1].split(":")[0])/725) for l in data_p]
    # print(seq[:10], lengths[:10])
    print(p, len(seq), seq[:10])
    if len(seq) == 0:
      continue
    seq_n = seq[0]
    drops_ts = []
    dropped_pkt = {}
    dropped_pkt_time = {}
    pkt_seqs = []
    retransmits = []
    retransmit_delays = []
    small_pkt_drop = False
    small_pkt_size = 1448
    for i in range(1,len(seq)):
      if seq[i] > seq_n + lengths[i]:
        # print(p, "Drop:", seq[i], seq_n, lengths[i])
        seq_n += lengths[i]
        while seq_n < seq[i]:
          # dropped_pkt[seq_n + 1448 - lengths[i]] = -1
          dropped_pkt[seq_n + 1 - lengths[i]] = -1
          # dropped_pkt_time[seq_n + 1448 - lengths[i]] = times[i]
          dropped_pkt_time[seq_n + 1 - lengths[i]] = times[i]
          seq_n += 1
          if times[i] > time_c:
            drops_ts.append((times[i], 1))
        pkt_seqs.append(seq_n)
      elif seq[i] < seq_n + lengths[i]:
        # print(p, "Retransmit:", seq[i], seq_debug[i], seq_n, lengths[i])
        if seq[i] in pkt_seqs:
          retransmits.append(seq[i])
          # print("Double Trasmission for", p, seq[i], "at", times[i], seq_debug[i])
        elif seq[i] not in dropped_pkt:
          retransmits.append(seq[i])
        elif dropped_pkt[seq[i]] != -1:
          retransmits.append(seq[i])
          # print("Double Retrasmission for", p, seq[i], "at", times[i], seq_debug[i])
        else:
          retransmits.append(seq[i])
          retransmit_delays.append(times[i] - dropped_pkt_time[seq[i]])
          dropped_pkt[seq[i]] = 1
      else:
        seq_n += lengths[i]
        pkt_seqs.append(seq_n)
        small_pkt_size = 1448
        small_pkt_drop = False
    retransmit_sequeunces.append(retransmits)
    retransmission_delays.append(retransmit_delays)
    packet_drop_times.append(drops_ts)
  return packet_drop_times, retransmit_sequeunces, retransmission_delays

def parse_pcap2(directory, file, ports):
  if not os.path.exists("%s/tcpdump.log" %(directory)):
    os.system("tcpdump -r %s/%s > %s/tcpdump.log" %(directory, file, directory))
  f = open("%s/tcpdump.log" %(directory), 'r')
  data = f.read()
  data = data.split("\n")
  data = [d for d in data if "IP" in d]
  abs_time_stamp = convert_to_ms(data[0].split(" ")[0])
  packet_drop_times = []
  retransmit_sequeunces = []
  retransmission_delays = []
  for p in ports:
    if p == 20011:
      data_p = [d for d in data if ".isdnlog >" in d]
    else:
      data_p = [d for d in data if ".%d >" %(p) in d]
    data_p = [d for d in data_p if " seq " in d]
    data_p = [d for d in data_p if " ack " in d]
    data_p = [d for d in data_p if " length 0" not in d]
    times = [convert_to_ms(l.split(" ")[0])-abs_time_stamp for l in data_p]
    seq = [int(l.split("seq ")[1].split(":")[1].split(",")[0]) for l in data_p]
    lengths = [int(l.split("length ")[1]) for l in data_p]
    print(p, len(seq))
    seq_n = seq[0]
    drops_ts = []
    dropped_pkt = {}
    dropped_pkt_time = {}
    pkt_seqs = []
    retransmits = []
    retransmit_delays = []
    small_pkt_drop = False
    small_pkt_size = 1448
    for i in range(1,len(seq)):
      if seq[i] > seq_n + lengths[i]:
        print(p, "Drop:", seq[i], seq_n, lengths[i])
        seq_n += lengths[i]
        drops_ts.append(times[i])
        while seq_n < seq[i]:
          dropped_pkt[seq_n + 1448 - lengths[i]] = -1
          dropped_pkt_time[seq_n + 1448 - lengths[i]] = times[i]
          seq_n += 1448
        pkt_seqs.append(seq_n)
      elif seq[i] < seq_n + lengths[i]:
        print(p, "Retransmit:", seq[i], seq_n, lengths[i], seq[i-1])
        if seq[i] in pkt_seqs:
          retransmits.append(seq[i])
        elif seq[i] not in dropped_pkt and lengths[i] < 1448:
          print("Smaller packet was lost")
          retransmits.append(seq[i])
          retransmit_delays.append(times[i] - dropped_pkt_time[seq[i] + (1448 - lengths[i])])
          # retransmit_delays.append(times[i] - dropped_pkt_time[seq[i]])
          dropped_pkt[seq[i]] = 1
          small_pkt_drop = True
          small_pkt_size = lengths[i]
        elif seq[i] not in dropped_pkt and small_pkt_drop:
          print("Seq number shifted due to small packet")
          retransmits.append(seq[i])
          retransmit_delays.append(times[i] - dropped_pkt_time[seq[i]+ (1448 - small_pkt_size)])
          dropped_pkt[seq[i]] = 1
        elif dropped_pkt[seq[i]] != -1:
          retransmits.append(seq[i])
        else:
          retransmits.append(seq[i])
          retransmit_delays.append(times[i] - dropped_pkt_time[seq[i]])
          dropped_pkt[seq[i]] = 1
      else:
        seq_n += lengths[i]
        pkt_seqs.append(seq_n)
        small_pkt_size = 1448
        small_pkt_drop = False
    retransmit_sequeunces.append(retransmits)
    retransmission_delays.append(retransmit_delays)
    packet_drop_times.append(drops_ts)
  return packet_drop_times, retransmit_sequeunces, retransmission_delays


def parse_log(file, flows, bg=False):
  f = open(file, 'r')
  data = f.read()
  data = data.split('\n')
  print(len(data))
  data = [d for d in data if "> ubuntu" in d or "> cs-ammart2" in d or "> 192.168.1.174" in d]
  # data = [d for d in data if "> crab" in d]
  print(len(data))
  flow_bws = []
  for fl in flows:
    flow_log = [d for d in data if fl in d]
    flow_log = [d for d in flow_log if 'length' in d]
    print(len(flow_log))
    # print(flow_log[:2])
    abs_times = [convert_to_ms(e.split(' ')[0]) for e in flow_log]
    byte = [e.split('length ')[1] for e in flow_log]
    byte = [int(e.split(' ')[0].replace(':','')) for e in byte]
    time = [t-abs_times[0] for t in abs_times]
    # print(len(time))
    # print(time[-1])
    bytes_recv = [0]* (int(int(time[-1])/td)+1)
    for t,b in zip(time, byte):
      t = int(int(t)/td)
      try:
        bytes_recv[t] += b
      except:
        print(t, time[-1])
    mbps = [float(d*8)*(1000/td)/(1024*1024) for d in bytes_recv]
    flow_bws.append((mbps, abs_times[0], abs_times[-1]))
    # print(mbps[:10])
    # print(byte[:10])
  if bg:
    sent_data2 = [d for d in data if 'length' not in d]
    print(sent_data2)
    sent_data = [d for d in data if 'length' in d]
    for fl in flows:
      sent_data = [d for d in sent_data if fl not in d]
    print(sent_data[:5])
    abs_times = [convert_to_ms(e.split(' ')[0]) for e in sent_data]
    byte = [int(e.split('length ')[1].split(' ')[0].replace(':','')) for e in sent_data]
    time = [t-abs_times[0] for t in abs_times]
    bytes_recv = [0]* (int(int(time[-1])/td)+1)
    for t,b in zip(time, byte):
      t = int(int(t)/td)
      try:
        bytes_recv[t] += b
      except:
        print(t, time[-1])
    mbps = [float(d*8)*(1000/td)/(1000*1000) for d in bytes_recv]
    flow_bws = [(mbps, abs_times[0], abs_times[-1])] + flow_bws
    # print(mbps[:10])
    
  return flow_bws

def calculate_fariness(flows, time, trange=-1):
  if trange == -1:
    trange = (time[int(len(time)/20)], time[int(8.5*len(time)/10)])
  flow_shares = []
  for f in flows:
    flow_shares.append(sum(f[trange[0]:trange[1]]))
  return sum(flow_shares)**2 / (len(flow_shares)*sum([f**2 for f in flow_shares]))

def calculate_avg_fariness(flows, time):
  fairness = []
  for i in range(len(time)):
    f_2 = 0
    f_s = 0
    n = 0
    for f in flows:
      f_2 += (f[i]**2)
      f_s += f[i]
      if f[i] > 0:
        n+=1
    if n>1:
      fairness.append(f_s**2 / (n * f_2))
  return np.mean(fairness)
  

def calculate_throughputs(flows, time_c=0):
  flow_shares = []
  from_i = int(time_c / td)
  for f in flows:
    if len(f[from_i:]):
      flow_shares.append(sum(f[from_i:])/len(f[from_i:]))
  tpts = []
  for i in range(from_i, len(flows[0])):
    tpts.append(sum([f[i] for f in flows]))
  return sum(flow_shares), tpts

def cummulate(flows, tdiff=0):
  times = [[s,e] for _,s,e in flows]
  min_t, max_t = min([min(t) for t in times]), max([max(t) for t in times])
  # print(times, min_t, max_t)
  mbps = [0] * (int((max_t - min_t) / td) + 1)
  cum_flows = []
  for f,s,e in flows:
    start_buf = int((s-min_t) / td)
    # end_buf = int((max_t-e) / td)
    buf_mbps = [0]* start_buf + f + [0]* ((len(mbps)) - (len(f) + start_buf))
    mbps = np.add(mbps, buf_mbps)
    # print(mbps[:10])
    cum_flows.append(buf_mbps)
  return cum_flows, range(len(mbps)), mbps, min_t

def bw_cap2(time_s, pkts):
  bps = (pkts*1000) * 1500 * 8
  mbps = bps / (1024*1024)
  return [mbps] * time_s

def plot_video_play(chunks, periods, fname):
  max_period = max(periods)
  colors = ["red", "green", "navy", "rebeccapurple", "yellow", "blue"]
  deadline = [0]
  rebuffering = 0
  for i in range(len(chunks[0])):
    max_time = max([c[i] for c in chunks])
    if max_time > deadline[i]:
      rebuffering += (max_time - deadline[i])
      deadline.append(max_time+max_period)
    else:
      deadline.append(deadline[i]+max_period)
  plt.figure(figsize=(10,10))
  plt.plot(list(range(len(deadline))), deadline, "k", label="Chunk Deadline")
  for i in range(len(chunks)):
    plt.plot(list(range(len(chunks[i]))), chunks[i], ":", label="Flow %d"%(i+1), color = colors[i])
  plt.xlabel("Chunk#")
  plt.ylabel("Time (s)")
  plt.legend()
  plt.savefig(fname, bbox_inches="tight")
  return deadline[1] - max_period, rebuffering

def plot_video_buffer(chunks, periods, fname):
  colors = ["red", "green", "navy", "rebeccapurple", "yellow", "blue"]
  max_period = max(periods)
  buffer_by_second = []
  buffer_by_flows = [[0 for i in range(len(c)*max_period+20)] for c in chunks]
  for i in range(len(chunks)):
    for c in chunks[i]:
      c_i = int(c)+1
      buffer_by_flows[i][c_i] += periods[i]
  for i in range(len(buffer_by_flows[0])):
    for buf in buffer_by_flows:
      prev_buf = 0
      if i > 0:
        prev_buf = buf[i-1]
      buf[i] += prev_buf
    if len([b for b in buffer_by_flows if b[i]==0]) == 0:
      for buf in buffer_by_flows:
        buf[i] -= 1
        if buf[i] < 0:
          buf[i] = 0
  for i in range(len(buffer_by_flows[0])):
    buffer_by_second.append(min([f[i] for f in buffer_by_flows]))
  plt.figure(figsize=(10,10))
  for i in range(len(buffer_by_flows)):
    plt.plot(list(range(len(buffer_by_flows[i]))), buffer_by_flows[i], ":", label="Flow %d"%(i+1), color = colors[i])
  plt.fill_between(list(range(len(buffer_by_second))), buffer_by_second, 0, label="Overall Buffer",
                 facecolor="darkslategray", # The fill color
                 color='darkslategray',       # The outline color
                 alpha=0.2)
  plt.xlabel("Time (s)")
  plt.ylabel("Playback Buffer (s)")
  plt.legend()
  plt.savefig(fname, bbox_inches="tight")
  return max(buffer_by_second[:60])


def plot_drops(drops, fname="", slow_start=0, trange=-1):
  all_drops = []
  for d in drops:
    all_drops += d
  all_drops = sorted(all_drops, key=lambda x:x[0])
  if len(all_drops) == 0:
    return
  t0 = all_drops[0][0]
  tn = all_drops[-1][0]
  # times = [t-t0 for t,d in drops]
  # drops = [d for t,d in drops]
  drops = [d for t,d in all_drops if t-t0 > slow_start]
  times = [t-t0 for t,_ in all_drops if t-t0 > slow_start]
  drops = [sum(drops[:i]) for i in range(len(drops))]
  # print(fname, drops[-1])
  plt.figure(figsize=(8,3))
  plt.step(times, drops, "b--")
  plt.ylabel("Packet Drops")
  plt.xlabel("Time (ms)")
  x1,x2,y1,y2 = plt.axis()
  if trange!=-1:
    x1, x2 = trange[0], trange[1]
    y1 = np.interp(x1, times, drops) - 5
    y2 = np.interp(x2, times, drops) + 5
  plt.axis([x1,x2,y1,y2])
  # plt.title(title)
  if fname != "":
    plt.savefig(fname, bbox_inches="tight")

  plt.show()

def plot_fairness(flows, fname=""):
  jains = []
  for i in range(len(flows[0])):
    this_f = []
    for j in range(len(flows)):
      if flows[j][i] > 0:
        this_f.append(flows[j][i])
    if len(this_f) > 1:
      jains.append(sum(this_f)**2 / (len(this_f)*sum([f**2 for f in this_f])))

  plt.figure(figsize=(8,3))
  plt.plot(list(range(len(jains))), jains, "b--")
  plt.ylabel("Fairness Index")
  plt.xlabel("Time")
  plt.legend()
  x1,x2,y1,y2 = plt.axis()
  plt.axis((x1,x2,0,1))
  # plt.title(title)
  if fname != "":
    plt.savefig(fname, bbox_inches="tight")

  plt.show()
  return jains


def plot_drops_pkts(drops, pkts, fname="", slow_start=0):
  if len(drops) == 0:
    return
  t0 = min(drops[0][0], pkts[0][0])
  times_d = [t-t0 for t,d in drops if t-t0 > slow_start*1000]
  drops = [d for t,d in drops if t-t0 > slow_start*1000]
  times_p = [t-t0 for t,p in pkts if t-t0 > slow_start*1000]
  pkts = [p for t,p in pkts if t-t0 > slow_start*1000]
  drops = [sum(drops[:i]) for i in range(len(drops))]
  pkts = [sum(pkts[:i]) for i in range(len(pkts))]
  print(fname, drops[-1])
  plt.figure(figsize=(8,3))
  plt.plot(times_d, drops, "b--", label="Drops")
  # plt.plot(times_p, pkts, "r--", label="DeqPackets")
  plt.ylabel("Packets")
  plt.xlabel("Time (ms)")
  plt.legend()
  # plt.title(title)
  if fname != "":
    plt.savefig(fname, bbox_inches="tight")
  # plt.ion()
  plt.show()

def plot(mm_bw, mm_ts, flows, tm, mbps, bw_tup, labels, file=""):
  cap_bw = bw_cap2(len(mm_bw), bw_tup)
  mm_time = range(len(mm_bw))
  cap_time = range(len(cap_bw))
  max_bw = max(max(cap_bw), max(mm_bw))
  plt.figure(figsize=(10,10))
  plt.plot(cap_time, cap_bw, "b--", label="Capped Bandwidth")
  plt.plot(mm_time, mm_bw, "g", label = "MahiMahi observed Bandwidth")
  plt.plot(tm, mbps, "y", label = "Goodput")
  i = 0
  for f in flows:
    l = labels[i]
    i += 1
    plt.plot(tm, f, label=l)
  x1,x2,y1,y2 = plt.axis()
  plt.axis((x1,x2,0,max_bw + (max_bw / 2)))
  plt.ylabel("Mbps")
  plt.xlabel("Time (%d-th of s)" %(1000/td))
  plt.legend()
  # plt.title(title)
  plt.show()

def plot_bytes_v_time(compare_flow, base_flow, tm, cc, fname="", trange=-1):
  plt.figure(figsize=(8,3))
  lines = ['--', ':', '-.', '-', 'o-', '--', '--', '--', ':', '-.', '-', 'o-', '--', '--', '--', ':', '-.', '-', 'o-', '--', '--']
  colors = ["black", "royalblue", "navy", "rebeccapurple", "gray", "red"]
  colors = ["red", "green", "red", "rebeccapurple", "yellow", "blue", "black", "royalblue", "navy", "rebeccapurple", "gray", "red", "black", "royalblue", "navy", "rebeccapurple", "gray", "red"]
  tm = [t/(1000/td) for t in tm]
  base_low = [(10**6 / 8 )*sum(base_flow[:i])*(1/(1000/td)) + 48000 for i in range(len(base_flow))]
  base_high = [(10**6 / 8 )*sum(base_flow[:i])*(1/(1000/td)) for i in range(len(base_flow))]
  base_flow = [(10**6 / 8 )*sum(base_flow[:i])*(1/(1000/td)) for i in range(len(base_flow))]

  compare_flow = [(10**6 / 8 )*sum(compare_flow[:i])*(1/(1000/td)) for i in range(len(compare_flow))]
  plt.plot(tm, base_flow, "k")
  plt.plot(tm, compare_flow, "b--", label=cc)
  plt.fill_between(tm, base_high, base_low, label="$+ Q$",
                 facecolor="darkslategray", # The fill color
                 color='darkslategray',       # The outline color
                 alpha=0.2)
  plt.ylabel("Bytes Sent")
  plt.xlabel("Time (s)")
  if trange!=-1:
    x1, x2 = trange[0], trange[1]
    y1 = np.interp(x1, tm, base_flow) - 5
    y2 = np.interp(x2, tm, base_flow) + 5
  plt.axis([x1,x2,y1,y2])
  plt.legend()
  if fname != "":
    plt.savefig(fname, bbox_inches="tight")



def plot3(flows, tm, mbps, bws, bwt, bw_e, tm_e, fname="", trange=-1, yheight=-1, smooth=1, labels=[], lines=[], texts=[], lutil=1, ports=[], uplo=-1):
  # cap_bw = bw_cap2(len(mm_bw), bw_tup)
  # mm_time = range(len(mm_bw))
  # cap_time = range(len(cap_bw))
  # max_bw = max(max(cap_bw), max(mm_bw))
  plt.figure(figsize=(4,3))
  # plt.figure(figsize=(10,7))
  # font = {'family' : 'Helvetica',
  #       'weight' : 'normal',
  #       'size'   : 20}
  # plt.rc('font', **font)
  # plt.plot(cap_time, cap_bw, "b--", label="Capped Bandwidth")
  # plt.plot(mm_time, mm_bw, "g", label = "MahiMahi observed Bandwidth")
  # plt.plot(tm, smoothen(mbps, 10), "y", label = "Goodput")
  # lines = ['--', ':', '-.', '-', 'o-', '--', '--', '--', ':', '-.', '-', 'o-', '--', '--', '--', ':', '-.', '-', 'o-', '--', '--']
  colors = ["purple", "green", "royalblue", "red", "gray", "red"]
  # colors = ["red", "green", "red", "rebeccapurple", "yellow", "blue", "black", "royalblue", "navy", "rebeccapurple", "gray", "red", "black", "royalblue", "navy", "rebeccapurple", "gray", "red"]
  tm = [t/(1000/td) for t in tm]
  i = 0
  rets = []
  for f in flows:
    # l = labels[i]
    i += 1
    label = "f%d" %(i)
    line = "solid"
    cc = colors[i-1]
    if lines != []:
      line = lines[i-1]
    if labels != []:
      label = labels[i-1]
    elif ports != []:
      if ports[i-1] >= 30000:
        cc = "blue"
        label = "f%d-Vegas" %(i)
      elif ports[i-1] >= 24000:
        cc = "green"
        label = "f%d-Reno" %(i)
      elif ports[i-1] >= 20000:
        cc = "red"
        label = "f%d-BBR" %(i)
      else:
        cc = "black"
        label = "f%d-Cubic" %(i)
    rets += [smoothen(f,smooth)]
    plt.plot(tm, smoothen(f, smooth), label=label, color=colors[i], linestyle=line)
  if lutil != -1:
    link_util = [sum(item) for item in zip(*flows)]
    plt.fill_between(tm, smoothen(link_util, smooth), 0,
                 facecolor="darkslategray", # The fill color
                 color='darkslategray',       # The outline color
                 alpha=0.2)
  if uplo != -1:
    up, lo, r = uplo
    plt.plot(tm, [up*r for t in tm], ":", color="k")
    plt.plot(tm, [lo*r for t in tm], ":", color="k")
    plt.plot(tm, [r for t in tm], "--", color="k")
    plt.fill_between(tm, [up*r for t in tm], [lo*r for t in tm],
                 facecolor="darkslategray", # The fill color
                 color='darkslategray',       # The outline color
                 alpha=0.2)
  # plt.plot(tm, [7.5 for t in tm], "--", color="k")
  # plt.plot(tm, [8.5 for t in tm], "-", color="k")


  if len(bwt):
    plt.plot(bwt, bws, "-", label="Link Bandwidth", color="k")
  if len(tm_e):
    plt.plot(tm_e, bw_e, label="Max_BW")
  x1,x2,y1,y2 = plt.axis()
  if trange!=-1:
    x1, x2 = trange[0], trange[1]
  if yheight != -1:
    y1, y2 = yheight[0], yheight[1]
  plt.axis([x1,x2,y1,y2])
  # plt.grid(b=True, which='major', color='white', linestyle='-')

  
  if texts != []:
    txs = [t[0] for t in texts]
    plt.plot(txs,[0]*len(txs),"k*", label = "Demand Changes")
    for i,j,t in texts:
      plt.text(i,j,t, size=10)
  # plt.axis((x1,x2,0,max_bw + (max_bw / 2)))
  plt.ylabel("Throughput (Mbps)")
  plt.xlabel("Time (s)")
  # plt.grid()
  plt.legend()
  # plt.title(title)
  if fname != "":
    plt.savefig(fname, bbox_inches="tight")
    return rets
  plt.show()

def cummulative_increasing(x):
  ret = []
  for i in range(len(x)):
    ret.append((sum(x[:i])/8))
  return ret

def plot_bars(xs, ys, x_label, y_label, fname, ynames=None, legend_labels = []):
  plt.figure(figsize=(2,3))
  i = 0
  plt.bar(xs, ys, color="navy")
  plt.ylabel(y_label)
  plt.xlabel(x_label)
  if ynames:
    plt.yticks(ticks=[1,2,3,4,5,6,7,8], labels=ynames)
  if fname != "":
    plt.savefig(fname, bbox_inches="tight")
    return

def plot_bars_multi(xs, ys, x_label, y_label, legend_labels, fname, ynames=None):
    plt.figure(figsize=(10, 5))
    
    num_bars = len(ys)
    bar_width = 0.8 / num_bars  # Adjust the width of each bar
    
    for i, y in enumerate(ys):
        x_pos = np.arange(len(xs)) + (i - 0.5 * (num_bars - 1)) * bar_width
        plt.bar(x_pos, y, width=bar_width, label=legend_labels[i])
    
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.xticks(np.arange(len(xs)), xs)
    if ynames:
        plt.yticks(ticks=[1, 2, 3, 4, 5, 6, 7, 8], labels=ynames)
    plt.grid(False)
    if legend_labels:
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    
    if fname != "":
        plt.savefig(fname, bbox_inches="tight")
    
    plt.show()

def plot4(flows, tm, mbps, bw_tup, fname=""):
  plt.figure(figsize=(10,5))
  i = 0
  for f in flows:
    i += 1
    plt.plot(tm, cummulative_increasing(f), label="flow %d" %(i))
  x1,x2,y1,y2 = plt.axis()
  plt.ylabel("MB")
  plt.xlabel("Time (%d-th of s)" %(1000/td))
  plt.grid()
  plt.legend()
  if fname != "":
    plt.savefig(fname, bbox_inches="tight")
    return
  plt.show()


def smoothen(f, avg_over):
  return [np.mean(f[i - min(i,avg_over):i]) for i in range(len(f))]

def flow_download(flows, trange=-1):
  if trange == -1:
    return [sum(f)/8/(1000/td) for f in flows]
  else:
    return [sum(f[trange[0]:trange[1]])/8/(1000/td) for f in flows]


def plot2(flows, tm, mbps, quality, rebuf, rebuf_pt, bws=[], bwt=[], smooth=10, fair_share=-1, num_bulk=-1, fname="", trange=-1, yheight=-1, lutil=[]):
  plt.figure(figsize=(5,3))
  # font = {'family' : 'Helvetica',
  #       'weight' : 'normal',
  #       'size'   : 30}
  # plt.rc('font', **font)
  # plt.rcParams.update({'font.size': 30})
  fig, ax1 = plt.subplots(figsize=(6,4))
  ax2 = ax1.twinx()
  # ax1.plot(tm, mbps, "y", label = "Goodput")
  i = 0
  line = ['--', ':', '--', '--', '--', '--', '--']
  colors = ["gray", "navy", "skyblue", "rebeccapurple", "yellow", "blue"]
  # colors = []
  tm = [t/(1000/td) for t in tm]
  for f in flows:
    if i==0:
      print("Video flow cummulative:", sum(f)/8/(1000/td))
      ax1.plot(tm, smoothen(f, smooth), line[i], label="Video flow", color=colors[i])
    else:
      print("Bulk flow %d cummulative:" %(i), sum(f)/8/(1000/td))
      ax1.plot(tm, smoothen(f, smooth), line[i], label="Bulk Flow", color=colors[i])
    i += 1
  
  link_util = [sum(item) for item in zip(*flows)]
  if lutil != []:
    link_util = lutil
  ax1.fill_between(tm, smoothen(link_util, smooth), 0, #label="Link Utilization",
                 facecolor="darkslategray", # The fill color
                 color='darkslategray',       # The outline color
                 alpha=0.2)
  # ax1.plot(tm, smoothen(link_util, smooth), "*-", label="Link Utilization", color="gray")
  if fair_share!=-1:
    ax1.plot([tm[0], tm[-1]], [fair_share, fair_share], '--', label="fair share")
  for r in rebuf:
    xs, ys = [r[0]], [rebuf_pt]
    if r[1] != 0:
      xs += [r[0]+r[1]]
      ys += [rebuf_pt]
    ax2.plot([x/(1000/td) for x in xs], ys, 'g-*', label="Rebuffering")
  tmq = [t/(1000/td) for t in quality[1]]
  print(tmq, quality[0])
  ax2.plot(tmq, [q-1 for q in quality[0]], "k-", label="Video Quality", linewidth=3)
  if len(bwt):
    ax1.plot(bwt, bws, label="Link BW")
  x1,x2,y1,y2 = ax1.axis()
  if trange!=-1:
    x1, x2 = trange[0], trange[1]
  if yheight != -1:
    y2 = yheight
  else:
    y2 += 0.25*y2
  ax2.axis([x1,x2,-0.2,9])
  ax1.axis([x1,x2,-0.2,y2])    
  ax1.set_ylabel("Throughput (Mbps)")
  ax1.grid()
  ax2.set_ylabel("Quality")
  ax2.set_yticks([1,2,3,4,5,6,7])
  ax2.set_yticklabels(['240p', '360p', '480p', '720p', '1080p', "1440p", "2160p"])
  ax1.set_xlabel("Time (s)")
  ax1.legend(loc=2)
  ax2.legend(loc=1)
  # plt.title(title)
  if fname != "":
    plt.savefig(fname, bbox_inches="tight")
    return
  plt.show()


def intify(xs):
  vlevels = ['tiny','small', 'medium', 'large', 'hd720', 'hd1080', "hd1440", "hd2160"]
  return [(t,vlevels.index(q.replace("'",""))+1) for t,q in xs]

def intify_nf(xs):
  vlevels = ['tiny','small', '432', '540', '720', 'hd1080', "hd1440", "hd2160"]
  return [vlevels.index(q.replace("'",""))+1 for q in xs]

def linearize(xy, end):
  xs = []
  ys = []
  prev = xy[0][1]
  for x,y in xy:
    x = int(x/td)
    xs += [x, x]
    ys += [prev,y]
    prev = y
  xs += [int(end/td)]
  ys += [ys[-1]]
  return ys, xs

def vlog2(file, diff= 0):
  f = open(file, 'r')
  data = f.read()
  data = data.split('\n')
  # for x in data:
  #   print(x)
  # data = [x for x in data if 'video.html' in x or '\u200b' in x]
  data = [x for x in data if 'Failed' not in x]
  data = [x for x in data if 'Intervention' not in x]
  # for x in data:
  #   print(x)
  start_ts = int(data[0].split(' ')[1]) - diff*1000
  # start_qu = [x for x in data if 'Starting quality' in x][0].split(' ')[-1].replace('"', '')
  q_change = [x for x in data if 'Playback quality changed ' in x]
  q_t_series = [(int(x.split(' ')[1])-start_ts, x.split(' ')[-1].replace('"','')) for x in q_change]
  rebuf = [x for x in data if 'Rebuffering' in x]
  vplay = [x for x in data if 'Playing' in x]
  startup_latency = int(vplay[0].split(" ")[0]) - start_ts
  rebuffers = [(0,startup_latency/td)]
  if len(vplay) == len(rebuf)+1:
    vplay = vplay[1:]
  if len(vplay)+1 == len(rebuf):
    rebuf = rebuf[1:]
  if len(rebuf):
    assert(len(rebuf) == len(vplay))
    rebuf_t_series = [(int(rebuf[i].split(' ')[1]) -start_ts, int(vplay[i].split(' ')[1]) -start_ts) for i in range(len(rebuf))]
    rebuffers += [(int(x/td), int((y-x)/td)) for x,y in rebuf_t_series]
  end_ts = int([x for x in data if 'Ended' in x][0].split(' ')[1]) - start_ts
  quality_series = linearize(intify(q_t_series),end_ts)
  return quality_series, rebuffers, end_ts, startup_latency

def vlog(file, diff= 0):
  f = open(file, 'r')
  data = f.read()
  data = data.split('\n')
  print(data[-2])
  # data = [x for x in data if 'video.html' in x]
  start_ts = int(data[0].split(': ')[1].split(', ')[0]) - diff*1000
  start_qu = data[0].split(': ')[1].split(', ')[1]
  q_change = [x for x in data if 'quality-changed' in x]
  q_t_series = [(0, start_qu)]+[(int(x.split(': ')[1].split(', ')[0])-start_ts, x.split(': ')[1].split(', ')[1]) for x in q_change]
  # rebuf = [x for x in data if 'Rebuffering' in x]
  # vplay = [x for x in data if 'Video playing' in x]
  # assert(len(rebuf) == len(vplay))
  # rebuf_t_series = [(int(rebuf[i].split(' ')[1]) -start_ts, int(vplay[i].split(' ')[1]) -start_ts) for i in range(len(rebuf))]
  end_ts = [x for x in data if 'end' in x]
  if end_ts == []:
    end_ts = 296000
  else:
    end_ts = end_ts[0]
    end_ts = int(end_ts.split(': ')[1].split(', ')[0]) - start_ts
  print(q_t_series)
  quality_series = linearize(intify(q_t_series),end_ts)
  print(quality_series)
  rebuffers = []
  return quality_series, rebuffers, end_ts

def parse_crab_log(fname, string="assigned_bw", fnames=[]):
  f = open(fname, 'r')
  data = f.read()
  data = data.split('*\n')
  start_ts = float(data[0])
  log = data[1:]
  if fnames == []:
    flow_names = [d.split(": ")[1].split(", ")[0] for d in data[1].split("\n") if "flow_name" in d]
  else:
    flow_names = fnames
  print(flow_names)
  ret = []
  string1 = "assigned_bw"
  string2 = "lended_bw"
  for e in log:
    e = e.split("\n")
    ts = float(e[0].split(": ")[0]) - start_ts
    obs_bw = float(e[0].split(": ")[1])
    bws = [float(x.split(string1)[1].split(", ")[0].replace(":","").replace(" ","")) - float(x.split(string2)[1].split(", ")[0].replace(":","").replace(" ","")) for x in e[1:-1]]
    ret += [(ts, obs_bw, bws)]
  return ret, start_ts, flow_names

def parse_crab_log_get_max_bw(fname, td):
  f = open(fname, 'r')
  data = f.read()
  data = data.split('*\n')
  start_ts = float(data[0])
  log = data[1:]

  ret = []
  tss = []
  for e in log:
    e = e.split("\n")
    ts = float(e[0].split(": ")[0]) - start_ts
    obs_bw = float(e[0].split(": ")[1])
    # print(e[1:])
    bws = sum([float(x.split("actual_share")[1].split(", ")[0].replace(":","").replace(" ","")) for x in e[1:-1]])
    if ts > td:
      ret += [bws]
      tss += [ts - td]
  return ret, tss, start_ts


def parse_crab_log2(fname, string="actual_share", fnames=[]):
  f = open(fname, 'r')
  data = f.read()
  data = data.split('*\n')
  start_ts = float(data[0])
  log = data[1:]
  if fnames == []:
    flow_names = [d.split(": ")[1].split(", ")[0] for d in data[1].split("\n") if "flow_name" in d]
  else:
    flow_names = fnames
  print(flow_names)
  ret = []
  for e in log:
    e = e.split("\n")
    ts = float(e[0].split(": ")[0]) - start_ts
    obs_bw = float(e[0].split(": ")[1])
    # print(e[1:])
    # print([x for x in e[1].split("flow_id")])
    bws = [float(x.split(string)[1].split(", ")[0].replace(":","").replace(" ","")) for x in e[1].split("flow id")[1:]]
    ret += [(ts, obs_bw, bws)]
  print(data[1])
  print(ret[0])
  print(data[2])
  print(ret[1])
  return ret, start_ts, flow_names

def cdf(x):
  x, y = sorted(x), np.arange(len(x)) / len(x)
  return (x, y)

def plot_cdf(xs, names, xlabel, ylabel, title, save="", lines = []):
  cdfs = [cdf(x) for x in xs]
  plt.figure(figsize=(4,3))
  line = ['--', '-']
  colors = ["royalblue", "navy", "slategray", "black"]
  # colors = ["red", "green", "red", "rebeccapurple", "yellow", "blue", "black", "royalblue", "navy", "rebeccapurple", "gray", "red", "black", "royalblue", "navy", "rebeccapurple", "gray", "red"]
  for i in range(len(xs)):
    x,y = cdfs[i]
    print("median for", names[i], x[int(len(x)/2)])
    print("Mean for", names[i], np.nanmean(xs[i]))
    if lines != []:
      plt.plot(x,y,lines[i], label= names[i], color=colors[i])
    else:
      plt.plot(x,y, line[i%2],label= names[i], color=colors[int(i/2)])
  # font = {'family' : 'Helvetica',
  #   'weight' : 'normal',
  #   'size'   : 22}
  # plt.rc('font', **font)
  plt.xlabel(xlabel)
  # plt.xticks(np.arange(5, 60, 5.0))
  plt.ylabel(ylabel)
  x1,x2,y1,y2 = plt.axis()
  plt.axis([0,x2,0,y2])
  plt.grid(True)
  # plt.title(title)
  plt.legend()
  if save!="":
    plt.savefig(save, bbox_inches="tight")
  return cdfs

def plot_tail(tail, xs, names, xlabel, ylabel, title, save="", lines = []):
  cdfs = [cdf(x) for x in xs]
  plt.figure(figsize=(20,10))
  for i in range(len(xs)):
    x,y = cdfs[i]
    print("median for", names[i], x[int(len(x)/2)])
    print("Mean for", names[i], np.mean(xs[i]))
    olen = len(x)
    x = x[int(tail*len(x)):]
    y = y[int(tail*olen):]
    print(len(x), len(y))
    if lines != []:
      plt.plot(x,y,lines[i], label= names[i])
    else:
      plt.plot(x,y,label= names[i])
  font = {'family' : 'Helvetica',
    'weight' : 'normal',
    'size'   : 16}
  plt.rc('font', **font)
  plt.xlabel(xlabel)
  plt.xticks(np.arange(1, 80, 5.0))
  plt.ylabel(ylabel)
  # plt.title(title)
  plt.legend(loc="bottom right")
  if save!="":
    plt.savefig(save, bbox_inches="tight")

def str_to_list(fname, pg_sizes = False):
  f = open(fname+'plt.txt', 'r')
  x = f.read()
  f.close()
  x = x.split('])')
  # print(x)
  all_plts = []
  sizes = []
  for e in x:
    if e == "]":
      continue
    # print(e)
    try:
      plts = e.replace('[(','(').split(".com', [")[1].split(", ")
      plts = [int(p) for p in plts]
      all_plts += plts
    except:
      plts = e.replace('[(','(').split(".com', ")[1].split("), (")
      plts = [p.replace('(','').replace(')','').split(', ') for p in plts]
      plts_ = [int(p[0]) for p in plts]
      sizes.append([int(p[1]) for p in plts])
      all_plts += plts_
  # x = x.replace('[','').replace(']','')
  # x = x.split(', ')
  if pg_sizes:
    return sizes
  all_plts = [int(i)/1000 for i in all_plts]
  return all_plts

def plot_bws(flows, flow_names, trange=-1):
  bw_lists = [[] for i in range(len(flow_names))]
  time = []
  for t,_,b_list in flows:
    if (trange!=-1 and t>trange[0] and t<trange[1]) or trange==-1:
      time.append(t)
      for i in range(len(b_list)):
        bw_lists[i].append(b_list[i])
  plt.figure(figsize=(10,5))
  line = ['-', '-', '-', '-', '--', '--', '--']
  for i in list(range(len(bw_lists))):
    plt.plot(time,bw_lists[i],line[i],label= flow_names[i])
  font = {'family' : 'Helvetica',
    'weight' : 'normal',
    'size'   : 16}
  plt.rc('font', **font)
  plt.xlabel("Time(s)")
  plt.ylabel("Actual Share of Bandwidth")
  # plt.title(title)
  plt.legend()
  plt.grid()
  plt.savefig("bws.png", bbox_inches="tight")


def plot_video(xs, names, xlabel, ylabel, title):
  plt.figure(figsize=(10,5))
  for i in range(len(xs)):
    print(xs[i][0], xs[i][1])
    tmq = [t/(1000/td) for t in xs[i][1]]
    plt.plot(tmq, [q+1 for q in xs[i][0]], "*-", label=names[i])
  font = {'family' : 'Helvetica',
    'weight' : 'normal',
    'size'   : 16}
  plt.rc('font', **font)
  plt.ylabel("Quality")
  plt.yticks([2,3,4,5,6])
  # plt.set_yticklabels(['240p', '360p', '480p', '720p', '1080p'])
  plt.xlabel(xlabel)
  # plt.ylabel(ylabel)
  # plt.title(title)
  plt.legend(loc="bottom right")
  plt.savefig("vid.pdf", bbox_inches="tight")

def avg_video_quality(vq):
  quality, times = vq[0], vq[1]
  prev_time = 0
  vq_sum = 0
  for i in range(len(quality)):
    vq_sum += quality[i] * (times[i] - prev_time)
    prev_time = times[i]
  return vq_sum / times[-1]


def count_bytes(file, id, trange=-1):
  f = open(file, 'r')
  data = f.read()
  data = data.split('\n')
  data = [d for d in data if "> crab" in d or "> cs-ammart2" in d]
  data = [d for d in data if id in d]
  print(data[1000:1010])
  # data = [d for d in data if "> crab" in d]
  sent_data = [d for d in data if 'length' in d]
  abs_times = [convert_to_ms(e.split(' ')[0]) for e in sent_data]
  byte = [int(e.split('length ')[1].split(' ')[0].replace(':','')) for e in sent_data]
  time = [t-abs_times[0] for t in abs_times]
  num_bytes = 0
  if trange == -1:
    trange = (time[0], time[-1])
  else:
    trange = (trange[0]*1000, trange[1]*1000)
  for i in range(len(byte)):
    if time[i] >= trange[0] and time[i] <= trange[1]:
      num_bytes += byte[i]  
  return num_bytes

def packet_inter(file, id, trange=-1):
  f = open(file, 'r')
  data = f.read()
  data = data.split('\n')
  # data = [d for d in data if "> crab" in d or "> cs-ammart2" in d]
  data = [d for d in data if id in d]
  print(data[0], data[-1])
  # data = [d for d in data if "> crab" in d]
  # sent_data = [d for d in data if 'length' in d]
  sent_data = data
  for i in range(25):
    print(i, sent_data[i])
  abs_times = [convert_to_ms(e.split(' ')[0]) for e in sent_data]
  # byte = [int(e.split('length ')[1].split(' ')[0].replace(':','')) for e in sent_data]
  time = [t-abs_times[0] for t in abs_times]
  if trange == -1:
    trange = (time[0], time[-1])
  intertimes = []
  for i in range(len(time)):
    if time[i] >= trange[0]:
      for j in range(i, len(time)):
        if time[j] <= trange[1]:
          intertimes.append(time[j] - time[i])
      break
  plt.figure(figsize=(10,5))
  print(len(intertimes))
  # plt.figure(figsize=(10,7))
  font = {'family' : 'Helvetica',
        'weight' : 'normal',
        'size'   : 16}
  plt.rc('font', **font)
  plt.plot(intertimes, "*-")
  plt.xlabel("Packet #")
  plt.ylabel("Time (ms)")
  plt.title("Interarrival time from first packet")
  plt.show()