import socket
import threading
import time

class Config():
    def __init__(self, congestion_protocols, ports, schedule, bursts=[]):
        self.number_connections = len(ports)
        self.ports = ports
        self.congestion_protocols = congestion_protocols
        self.schedule = schedule
        self.bursts = bursts
        print(self.number_connections)
        print(self.ports)
        print(self.congestion_protocols)
        # print(self.schedule)

    def linearize_schedule(self, periodic_schedule, dt=4):
        l_schedule = []
        initial_burst, reg_burst = self.bursts
        for p in periodic_schedule:
            conn_id, start, data_to_send, period, repeat = p
            for i in range(repeat):
                if i < initial_burst:
                    l_schedule.append((conn_id, max(0, start - dt), data_to_send))
                else:
                    j = (i - initial_burst) % reg_burst
                    k = i - j
                    l_schedule.append((conn_id, max(0, start + k * period - dt), data_to_send))
        return l_schedule

class TrafficGenerator():
    def __init__(self, config, tcpdump_fname= "", tcpdump=False):
        self.config = config
        self.time_now = 0
        self.tcpdump = tcpdump
        if self.tcpdump:
            self.control_conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.control_conn.connect(('192.168.112.173', 8081))
            msg = "tcpdump:%s" %(tcpdump_fname)
            self.control_conn.send(msg.encode('utf-8'))
        #sort by times
        self.conns = [self.client_socket(config.ports[i], config.congestion_protocols[i], config.schedule[i]) for i in range(config.number_connections)]

        self.config.schedule = self.config.linearize_schedule(self.config.schedule)
        self.config.schedule = sorted(config.schedule, key=lambda x:x[1])
        print(self.config.schedule)
    
    def run(self):
        threads = []
        for c_id, next_time, data_size in self.config.schedule:
            if next_time-self.time_now > 0:
                time.sleep(next_time-self.time_now)
                self.time_now = next_time
            t = threading.Thread(target=self.traffic, args=(self.conns[c_id], data_size))
            t.start()
            threads.append(t)
        for t in threads:
            t.join()
    
    def client_socket(self, port, congestion_algo, schedule):
        clientsocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        clientsocket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        clientsocket.setsockopt(socket.IPPROTO_TCP, socket.TCP_CONGESTION, congestion_algo.encode())
        try:
            clientsocket.bind(("100.64.0.5", port))
        except:
            clientsocket.bind(("100.64.0.4", port))
        clientsocket.connect(('192.168.112.173', 8081))
        clientsocket.send("%d %d %d %d".encode('utf-8') %(schedule[1], schedule[2], schedule[3], schedule[4]))
        return clientsocket
    
    def traffic(self, clientsocket, size):
        # clientsocket.send(str(size).encode('utf-8'))
        msg = "".join(['h' for i in range(size)])
        clientsocket.send(msg.encode('utf-8'))

    def close(self):
        for clientsocket in self.conns:
            clientsocket.close()
    
    def report_metrics(self, fname):
        all_stats = []
        for clientsocket in self.conns:
            clientsocket.send("report".encode("utf-8"))
            stats = clientsocket.recv(65536)
            all_stats.append(stats.decode("utf-8"))
        file = open(fname, "w")
        for stat in all_stats:
            file.write(stat)
            print(stat)
        file.close()
        self.control_conn.send("close".encode("utf-8"))