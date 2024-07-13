import socket
import threading
import time
import subprocess
import os

tcpdump = None

def run_tcpdump(directory):
	process = subprocess.Popen(["tcpdump", "-i", "any", "-w", directory, "net", "100.64.0.0/24"], shell=False, stdout=subprocess.DEVNULL, preexec_fn=os.setsid)
	return process

def get_pid(name):
	try:
		output = subprocess.check_output(["pidof", name])
		return list(map(int, output.split()))
	except:
		return []

def kill_tcpdump():
	tcpdump_pids = get_pid("tcpdump")
	for pid in tcpdump_pids:
		os.system("sudo kill -9 %s" %(pid))

def handle_connection(connection, address):
    global tcpdump
    stats = "%s\n" %(str(address))

    while True:
        buf = connection.recv(100)
        # pcaps
        if len(buf) > 0 and "tcpdump" in buf.decode('utf-8'):
            dir_name = buf.decode('utf-8').split(":")[1]
            tcpdump = run_tcpdump(dir_name)
        elif len(buf) > 0 and "close" in buf.decode('utf-8'):
            kill_tcpdump()
            tcpdump = None
        elif len(buf) > 0 and "report" not in buf.decode('utf-8'):
            print(address, ": ", buf.decode('utf-8'))
            start, data_size, period, repeat = [int(i) for i in buf.decode('utf-8').split(" ")]
            start_time = time.time()
            stats += "start: %d,  data_size: %d, period: %d, repeat: %d\n" %(start, data_size, period, repeat)
            stats += "base timestamp: %d\n" %(start_time)
            for i in range(repeat):
                recv_s = 0
                while recv_s < data_size:
                    buf = connection.recv(min(data_size-recv_s, 65536))
                    recv_s += len(buf)
                end_time = time.time() - start_time
                stats += "Chunk %d downloaded: %f" %(i, end_time)
                stats += " Play Time: %f\n" %(start+i*period)
                print(address, "-> No. %d chunk downloaded:" %(i), recv_s, end_time, "Deadline: ", start+i*period)
        else:
            connection.send(stats.encode("utf-8"))
            connection.close()
            break

def start_server():
    serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    serversocket.bind(('192.168.112.173', 8081))
    serversocket.listen(5)

    while True:
        connection, address = serversocket.accept()
        print("New connection from:", address)
        connection_thread = threading.Thread(target=handle_connection, args=(connection, address))
        connection_thread.start()

os.system("sudo sysctl -w net.ipv4.tcp_ecn=1")
start_server()
