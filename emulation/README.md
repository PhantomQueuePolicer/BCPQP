This directory contains implementation of BC-PQP and other baselines in Mahimahi along with the scripts to run experiments with different kinds of flows.

# Compile Mahimahi

Mahimahi is an network link emulation tool, we implement BC-PQP, Policer and Shaper in Mahimahi's linkshell. To compile Mahimahi, please follow steps given below:

```
cd mahimahi
./autogen.sh
./configure
make
sudo make install
```

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

In `scripts/configs.py` file, configure experiment setup including

- The number, size and congestion control protocols of flows
- Rate enforcement mechanism (Shaper, BC-PQP etc.) and their configurations
- logs directory

To run the experiments and generate plots:

Run the server first:

```
sudo python3 traffic_server.py
```

Then run the following to create mahimahi shell and generate flows:

```
python3 run_all.py
```

To generate plots, run:
```
python3 plot.py
```