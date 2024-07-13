import socket
import threading
import time
import sys
import os
import datetime

counters = {}
throughputs = {}
statistics = {}

def report():
    global counters, throughputs
    previous_snapshot = counters.copy()
    prev_time = time.time()
    while True:
        counters_snapshot = counters.copy()
        current_time = time.time()
        time_dif = current_time - prev_time
        for adr in counters_snapshot.keys():
            if adr in previous_snapshot:
                throughputs[adr].append((current_time, ((counters_snapshot[adr] - previous_snapshot[adr]) / time_dif) * 8 / 10**6))
            else:
                throughputs[adr] = [(current_time, (counters_snapshot[adr] / time_dif) * 8 / 10**6)]
        prev_time = current_time
        previous_snapshot = counters_snapshot
        time.sleep(0.25)


def handle_connection(connection, address):
    global counters, statistics
    stats = "%s\n" %(str(address))

    while True:
        buf = connection.recv(100)
        # pcaps
        if len(buf) > 0 and "report" not in buf.decode('utf-8'):
            print(address, ": ", buf.decode('utf-8'))
            start, data_size, period, repeat, cc, rtt = [i if i.isalpha() else int(i) for i in buf.decode('utf-8').split(" ")]
            start_time = time.time()
            stats += "cc: %s,  rtt: %d\n" %(cc, rtt)
            stats += "start: %d,  data_size: %d, period: %d, repeat: %d\n" %(start, data_size, period, repeat)
            stats += "base timestamp: %d\n" %(start_time)
            # prev_end_time = 0
            if address not in counters:
                counters[address] = 0
            for i in range(repeat):
                recv_s = 0
                while recv_s < data_size:
                    buf = connection.recv(min(data_size-recv_s, 65536))
                    # print(len(buf.decode('utf-8')))
                    recv_s += len(buf)
                    counters[address] += len(buf)
                end_time = time.time() - start_time
                # execution_time = end_time - start_time
                stats += "Chunk %d downloaded: %f" %(i, end_time)
                stats += " Play Time: %f\n" %(start+i*period)
                statistics[address] = stats
                # prev_end_time = end_time
                print(address, cc, rtt, "-> No. %d chunk downloaded:" %(i), recv_s, end_time, "Deadline: ", start+i*period)
        else:
            connection.send(stats.encode("utf-8"))
            connection.close()
            break

def start_server():
    serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    serversocket.bind(('10.0.1.11', 8081))
    # serversocket.setsockopt(socket.IPPROTO_TCP, socket.TCP_CONGESTION, "abc".encode())
    serversocket.listen(5)
    stats_thread = threading.Thread(target=report, args=())
    stats_thread.start()
    while True:
        connection, address = serversocket.accept()
        print("New connection from:", address)
        connection_thread = threading.Thread(target=handle_connection, args=(connection, address))
        connection_thread.start()

try:
    start_server()
except KeyboardInterrupt:
    ts = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    file = open("logs/%s.log" %(ts), "w")
    for f in throughputs.keys():
        file.write(str(f))
        file.write("\n")
        for ts, tp in throughputs[f]:
            file.write("%f: %f\n" %(ts, tp))
        file.write(statistics[f])
        file.write("\n")
    file.close()
    try:
        sys.exit(130)
    except SystemExit:
        os._exit(130)