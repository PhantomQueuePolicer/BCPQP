import os
import configs

os.system("tc qdisc del dev eth1 root")
os.system("tc qdisc add dev eth1 root handle 1: prio priomap 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2")
os.system("tc qdisc add dev eth1 parent 1:1 handle 10: netem delay 5ms")
os.system("tc qdisc add dev eth1 parent 1:2 handle 40: netem delay 15ms")
os.system("tc qdisc add dev eth1 parent 1:3 handle 80: netem delay 25ms")

# RTTs

for f in configs.flows:
    port = f[0]
    rtt = f[4]
    os.system("tc filter add dev eth1 protocol ip parent 1:0 prio 1 u32 match ip sport %d 0xffff flowid 1:%d" %(port, rtt))
