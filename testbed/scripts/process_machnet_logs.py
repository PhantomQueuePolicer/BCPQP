import numpy as np
import matplotlib.pyplot as plt

def extract_all_categories(file_path):
    with open(file_path, 'r') as file:
        data = file.read()
        data = data.split("\n\n")
        categories = [data[i] for i in range(len(data)) if i%2==0]
        data = [data[i] for i in range(len(data)) if i%2==1]
        data = [d for d in data if "Enq" in d or "Deq" in d]
        for i in range(len(categories)):
            data_i = [d for d in data[i].split("\n") if "Enq" in d or "Deq" in d]
            enq = [float(d.split("Enq: ")[1].split(",")[0]) for d in data_i if "nan" not in d and "inf" not in d]
            deq = [float(d.split("Deq:")[1].split(")")[0]) for d in data_i if "nan" not in d and "inf" not in d]
            pkt_deq = [float(d.split("TX PPS: ")[1].split(" ")[0]) for d in data_i if "nan" not in d and "inf" not in d]
            pkt_enq = [float(d.split("RX PPS: ")[1].split(" ")[0]) for d in data_i if "nan" not in d and "inf" not in d]
            print(categories[i], "CPU Cycles: ", np.mean(enq), " (Enq) " + np.mean(deq), " (Deq) ", "Drop%:", (sum(pkt_enq) - sum(pkt_deq)) / sum(pkt_enq))

def main():
    file_paths = ['machnet-logs/1_5mbps.log', 'machnet-logs/7_5mbps.log', 'machnet-logs/25mbps.log', 'machnet-logs/50mbps.log', 'machnet-logs/100mbps.log', 'machnet-logs/200mbps.log']
    
    for f in file_paths:
        print("\n", f)
        extract_all_categories(f)

if __name__ == "__main__":
    main()