import configs
from traffic_generator import *

ports = []
congestions = []
schedule = []
rtts = []
i = 0
for f in configs.flows:
    port, start_time, cc, data, period, repeat, rtt = f[0], f[1], f[2], f[3], f[5], f[6], f[4]
    schedule.append((i, start_time, data, period, repeat))
    congestions.append(cc)
    ports.append(port)
    rtts.append(rtt)
    i+=1
cfg = Config(congestions, ports, schedule, rtts, bursts=[1,1])
tg = TrafficGenerator(cfg)
tg.run()
tg.close()
print("done")
