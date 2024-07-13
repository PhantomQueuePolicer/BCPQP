This directory contains DPDK based implementation of BC-PQP and other baselines using Machnet. To run this code, you need F8sv2 Virtual Machines running Ubuntu 22.04 on Azure Public Cloud.

# Compile Machnet

Machnet is a kernel bypass networking framework for Azure Cloud VMs. You can find implementation of BC-PQP and other baselines in `machnet/src/apps/` To compile Machnet, follow the steps in `machnet` directory's README:


# Enable different congestion control protocols

Ensure that you have different congestion control protocols enabled, to check which protocols you have enabled, use:

`sysctl net.ipv4.tcp_available_congestion_control`

By default, only cubic is available, to enable additional protocols, use:

```
sudo sysctl -w net.ipv4.tcp_congestion_control=vegas
sudo sysctl -w net.ipv4.tcp_congestion_control=reno
sudo sysctl -w net.ipv4.tcp_congestion_control=bbr
```

# Run experiments

To run the experiments from the paper, you will need 3 F8sv2 Virtual Machines on Azure Public Cloud. A middlebox VM will run the rate enforcement mechanism (BC-PQP, shaper etc.) using machnet. Correct performance of different baseline is senstive to parameters like queue sizes etc., `scripts/parameters.py` can be used to generate commands to run BC-PQP and other baselines with correct parameters. Run `scripts/parameters.py` with information about rate (Kbps), maximum round trip time (max RTT in ms), number of queues per aggregate, sender IP, receiver IP as following:

```
python3 scripts/parameters.py --rate 1500 -maxrtt 100 --nq 64 --sender 192.168.0.1 --receiver 192.168.1.1
```

This generate commands with correct parameters for all baselines.

In addition to the middlebox VM, at least 2 more VMs are needed for sender and receiver. Sender, receiver and middlebox need to be configured such that sender's packets are routed through the middlebox to the receiver, to do this iptables can be used. Given sender, receiver and middlebox have IP addresses `ip1`, `ip2` and `ip3` respectively, at sender run:

```
iptables -t nat -A OUTPUT -p tcp -d ip2 -j DNAT --to-destination ip3
```

And at the receiver:

```
iptables -t nat -A OUTPUT -p tcp -d ip1 -j DNAT --to-destination ip3
```

Middlebox already has the implementation to do such address swapping.

Finally run the command for required baseline at the middlebox, the run following at the receiver:

```
python3 traffic_server.py
```

And run following at the sender:

```
python3 exp.py
```

This generates a file containing summary of per-flow throughput in `logs` directory at the receiver.