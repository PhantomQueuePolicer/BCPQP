This directory contains DPDK based implementation of BC-PQP and other baselines using Machnet. To run this code, you need F8sv2 Virtual Machines running Ubuntu 22.04 on Azure Public Cloud.

# Compile Machnet

Machnet is a kernel bypass networking framework for Azure Cloud VMs. You can find implementation of BC-PQP and other baselines in `machnet/src/apps/`. To compile Machnet, you can either follow the steps in `machnet` directory's README or follow the following instructions:

## Setup VMs

While we evaluated BC-PQP for our paper with F8sv2 VMs running Ubuntu 22.04, you can also use other cheaper and more readily available VMs such as D2sv3. You will need 3 VMs in the same virtual network: a sender, a receiver, and a shaping middlebox. The middlebox VM needs to support "Accelerated Networking" so that we can run DPDK on it. To do this, you'll need to create a new NIC and attach the middlebox VM to it so that you have two network interfaces: `eth0` and `eth1`. 

## Building Machnet

Do the following at the middlebox VMs (one with two interfaces):

Update and install dependencies:

```
sudo apt-get update && \
    apt-get install --no-install-recommends -y \
        git \
        build-essential cmake meson pkg-config libudev-dev \
        libnl-3-dev libnl-route-3-dev python3-dev \
        python3-docutils python3-pyelftools libnuma-dev \
        ca-certificates autoconf \
        libgflags-dev libgflags2.2 libhugetlbfs-dev pciutils libunwind-dev uuid-dev nlohmann-json3-dev

```

Remove conflicting packages and cleanup after install

```
sudo apt-get --purge -y remove rdma-core librdmacm1 ibverbs-providers libibverbs-dev libibverbs1
sudo rm -rf /var/lib/apt/lists/*
```
Get and build RDMA:

```
cd
export RDMA_CORE="/root/rdma-core"
git clone -b 'stable-v40' --single-branch --depth 1 https://github.com/linux-rdma/rdma-core.git ${RDMA_CORE}
cd ${RDMA_CORE}
mkdir build
cd build
cmake -GNinja -DNO_PYVERBS=1 -DNO_MAN_PAGES=1 ../
ninja install
```

Get and build DPDK:

```
cd
export RTE_SDK="/root/dpdk"
git clone --depth 1 --branch 'v21.11' https://github.com/DPDK/dpdk.git ${RTE_SDK}
cd ${RTE_SDK}
meson build -Dexamples='' -Dplatform=generic -Denable_kmods=false -Dtests=false -Ddisable_drivers='raw/*,crypto/*,baseband/*,dma/*'
cd build/ 
DESTDIR=${RTE_SDK}/build/install ninja install 
rm -rf ${RTE_SDK}/app ${RTE_SDK}/drivers ${RTE_SDK}/.git ${RTE_SDK}/build/app
```

Compile machnet:

```
cd machnet
git submodule update --init --recursive
sudo ldconfig
mkdir release_build
cd release_build
cmake -DCMAKE_BUILD_TYPE=Release -GNinja ../
ninja
```

Start machnet on the `eth1` interface:

```
MACHNET_IP_ADDR=`ifconfig eth1 | grep -w inet | tr -s " " | cut -d' ' -f 3`
MACHNET_MAC_ADDR=`ifconfig eth1 | grep -w ether | tr -s " " | cut -d' ' -f 3`

sudo modprobe uio_hv_generic
DEV_UUID=$(basename $(readlink /sys/class/net/eth1/device))
sudo driverctl -b vmbus set-override $DEV_UUID uio_hv_generic

echo "Machnet IP address: $MACHNET_IP_ADDR, MAC address: $MACHNET_MAC_ADDR"
./machnet.sh --mac $MACHNET_MAC_ADDR --ip $MACHNET_IP_ADDR
```

Doing so unbinds the NIC from the OS and thus this interface will not appear on `ifconfig` anymore. However, you can get information about the interface from the Azure metadata server:

```
curl -s -H Metadata:true --noproxy "*" "http://169.254.169.254/metadata/instance?api-version=2021-02-01" | jq '.network.interface[1]'
```

Add this information to the `src/apps/machnet/config.json` file.

Enable 2M hugepages:

```
sudo bash -c "echo 2048 > /sys/devices/system/node/node0/hugepages/hugepages-2048kB/nr_hugepages"
```

At this point, you're ready to run BC-PQP on the middlebox VM. Now, we need to configure the sender and receiver VMs.

# Configuring Sender and Receiver

## Enable different congestion control protocols

Ensure that you have different congestion control protocols enabled, to check which protocols you have enabled, use:

`sysctl net.ipv4.tcp_available_congestion_control`

By default, only cubic is available, to enable additional protocols, use:

```
sudo sysctl -w net.ipv4.tcp_congestion_control=vegas
sudo sysctl -w net.ipv4.tcp_congestion_control=reno
sudo sysctl -w net.ipv4.tcp_congestion_control=bbr
```

## Relay traffic through middlebox

Sender, receiver and middlebox need to be configured such that sender's packets are routed through the middlebox to the receiver, to do this iptables can be used. Given sender, receiver and middlebox have IP addresses `ip1`, `ip2` and `ip3` respectively, at sender run:

```
sudo iptables -t nat -F
sudo iptables -t nat -A INPUT -s ip2 -j SNAT --to-source ip3
sudo iptables -t nat -A OUTPUT -d ip2 -j DNAT --to-destination ip3
```

And at the receiver:

```
sudo iptables -t nat -F
sudo iptables -t nat -A INPUT -s ip1 -j SNAT --to-source ip3
sudo iptables -t nat -A OUTPUT -d ip1 -j DNAT --to-destination ip3
```

Middlebox already has the implementation to do such address swapping.


# Run experiments

Correct performance of different baseline is senstive to parameters like queue sizes etc., `scripts/parameters.py` can be used to generate commands to run BC-PQP and other baselines with correct parameters. Run `scripts/parameters.py` with information about rate (Kbps), maximum round trip time (max RTT in ms), number of queues per aggregate, number of shapers/aggregates, sender IP, receiver IP as following:

```
python3 scripts/parameters.py --rate 1500 -maxrtt 100 --nq 64 --ns 2050 --sender 192.168.0.1 --receiver 192.168.1.1
```

This generate commands with correct parameters for all baselines. You can now run the given commands from the middlebox VM from the `BCPQP/testbed/machnet/src/`.


At the sender, visit the `scripts/config.py` file to configure workload setup. We emulate different RTTs per-flow using netem, run the following to set the appropriate `tc` rules:

```
sudo python3 scripts/rtt-qdisc.py
```

Finally run the command for required baseline (generated above) at the middlebox, then run following at the receiver:

```
python3 traffic_server.py
```

And run following at the sender:

```
python3 run.py
```

This generates a file containing summary of per-flow throughput measured over 250 ms windows in `logs` directory at the receiver. 

Additionally, information about CPU cycles per packet is printed on middlebox VM's shell like this:

```
main.cc:352 [TX PPS: 1792.999784 (0.012731 Gbps), RX PPS: 2836.999659 (0.012731 Gbps), TX_DROP PPS: 1043.999874, Cycles(Enq: 90.513218, Deq:0.000000)]
```

# Generate Plots

To generate plots, you will need to label files in `logs` appropriately i.e. sort files into directories based on enforced rate [1.5, 7.5, 25, ...] with 1 file for each baseline. Then you can add file paths in the `scripts/plot_all.py` file and run:

```
python3 scripts/plot_all.py
```

This generates plots for average enforced rate, tail throughput and fairness index. 

Save the machnet logs and label them at the top of the file, and then run the following to get CPU cycles per packet and Drop rate:

```
python3 scripts/process_machnet_logs.py
```
