
import os
import matplotlib.pyplot as plt
import numpy as np

def jains(flows):
    fairness = []
    for i in range(len(flows[list(flows.keys())[0]])):
        sum_ = 0
        sq_sum = 0
        N = 0
        for fl in flows.keys():
            sum_ += flows[fl][i]
            sq_sum += (flows[fl][i]**2)
            if (flows[fl][i] > 0):
                N+=1
        if sum_ != 0:
            fairness.append((sum_**2) / (N*sq_sum))
        else:
            fairness.append(0)
    return fairness

def aggregate(flows, dt=0.25):
    min_time = min([min(f['timestamps']) for f in flows])
    max_time = max([max(f['timestamps']) for f in flows])
    tpts_aggregate = [0.0 for t in range(1+int((max_time - min_time)/dt))]
    fls = {}
    for f in flows:
        fls[f['port']] = [0.0 for t in range(1+int((max_time - min_time)/dt))]
    for f in flows:
        for i in range(len(f['throughput'])):
            ts = int((f['timestamps'][i] - min_time) / dt)
            tpt = f['throughput'][i]
            tpts_aggregate[ts] += tpt
            fls[f['port']][ts] += tpt
    return fls, tpts_aggregate

def read_throughput_data(file_path):
    shaper_flows = {}
    flows = []
    current_flow = None

    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('('):
                # Start of a new flow
                if current_flow:
                    shaper_id = int(int(current_flow['port']) / 64)
                    if shaper_id not in shaper_flows:
                        shaper_flows[shaper_id] = []
                    shaper_flows[shaper_id].append(current_flow)
                    flows.append(current_flow)

                ip, port = eval(line.strip())
                current_flow = {'ip': ip, 'port': port, 'timestamps': [], 'throughput': []}
            elif ":" in line:
                try:
                    timestamp, throughput = map(float, line.strip().split(':'))
                    current_flow['timestamps'].append(timestamp)
                    current_flow['throughput'].append(throughput)
                except: 
                    current_flow = None

    if current_flow:
        flows.append(current_flow)
    # shaper_flows = [aggregate(f) for f in shaper_flows]
    print(file_path)
    frns = []
    tpt = []
    tpt_p99 = []
    max_tpt = []
    agg_tpts = []
    fairnesses = []
    for s in shaper_flows.keys():
        fls, agg_tpt = aggregate(shaper_flows[s])
        fairness = jains(fls)
        frns.append(np.mean([t for t in fairness if t!=0 and t!=1]))
        tpt.append(np.mean([t for t in agg_tpt if t!=0]))
        tpt_p99.append(np.percentile([t for t in agg_tpt if t!=0], 99))
        max_tpt.append(max(agg_tpt))
        agg_tpts += [t for t in agg_tpt if t!=0]
        fairnesses += [t for t in fairness if t!=0]
        # print(s, ":", np.mean([t for t in agg_tpt if t!=0]), np.percentile([t for t in agg_tpt if t!=0], 95), np.percentile([t for t in agg_tpt if t!=0], 99), np.mean([t for t in fairness if t!=0]))
    print("Fairness:", np.mean(frns))
    print("Avg Throughput/shaper:", min(tpt))
    print("Avg p99 Throughput/shaper:", np.mean(tpt_p99))
    print("Max rate:", max(max_tpt), "\n")
    return flows, agg_tpts, fairnesses

def plot_bar(names, ys, ylabel, save_path=None):
    fig, ax = plt.subplots(figsize=(2,3))
    p = ax.bar(names, ys, color="gray")
    ax.bar_label(p, labels=names, rotation=90, label_type='center')
    plt.xticks([])
    plt.ylabel(ylabel)
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Plot saved to: {save_path}")

def plot_throughput(flows, save_path=None):
    labels = ["Reno", "Cubic", "BBR"]
    # labels = ["RTT = 5 ms", "RTT = 25 ms", "RTT = 50 ms"]
    plt.figure(figsize=(5,4))
    i = 0
    min_t = min([min(f['timestamps']) for f in flows])
    for flow in flows:
        ts = [t-min_t for t in flow["timestamps"]]
        plt.plot(ts, flow['throughput'], label=labels[i])
        i+=1

    plt.xlabel('Timestamp')
    plt.ylabel('Throughput')
    plt.legend()
    plt.xlim(0, 23)
    plt.ylim(0, 30)

    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()


def compute_cdf(numbers):
    sorted_numbers = np.sort(numbers)
    n = len(sorted_numbers)
    cdf_values = np.arange(1, n + 1) / n
    return sorted_numbers, cdf_values

def plot_cdfs(names, number_dicts, labels, xlabel, filename, xrange=-1, yrange=-1):
    plt.figure(figsize=(4, 3))
    colors = ["navy", "royalblue", "black", "purple", "skyblue"]
    lines = ["--", "-", ":", "-.", "--"]
    i=0
    for n in names:
        numbers = number_dicts[n]
        sorted_numbers, cdf_values = compute_cdf(numbers)
        plt.plot(sorted_numbers, cdf_values, label=labels[i], color=colors[i], linestyle=lines[i])
        i+=1
    plt.xlabel(xlabel)
    plt.ylabel("CDF")
    if xrange!=-1:
        plt.xlim(xrange[0], xrange[1])
    
    if yrange!=-1:
        plt.ylim(yrange[0], yrange[1])
    plt.legend()
    plt.grid(True)
    plt.savefig(filename, bbox_inches="tight")
    plt.close()


def process_directory(input_dirs, output_dir, rates):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    names = ["bcpqp", "fairpolicer", "shaper", "policer", "policer-bdp^2"]
    labels = ["BC-PQP", "FP", "Shaper", "Policer", "Policer+"]
    throughputs = {}
    fairness = {}
    for n in names:
        throughputs[n] = []
        fairness[n] = []
    i = 0
    for input_dir in input_dirs:
        for file_name in os.listdir(input_dir):
            if file_name.endswith('.log'):
                input_file_path = os.path.join(input_dir, file_name)
                output_file_path = os.path.join(output_dir, f'{os.path.splitext(file_name)[0]}_plot.png')

                throughput_data, tpts, fairnesses = read_throughput_data(input_file_path)
                name = file_name.split(".")[0]
                tpts_norm = [t/rates[i] for t in tpts]
                if name in names:
                    throughputs[name] += tpts_norm
                    fairness[name] += fairnesses
        i+=1
    
    plot_cdfs(names, throughputs, labels, "Normalized Throughput", filename="%s/avgthroughput.pdf" %(output_dir), xrange = (0.5, 1.5))
    plot_cdfs(names, throughputs, labels, "Normalized Throughput", filename="%s/p99throughput.pdf" %(output_dir), yrange = (0.95, 1.01))
    plot_cdfs(names, fairness, labels, "Fairness Index", filename="%s/fairness.png" %(output_dir))
    avg_tpt = [np.mean(throughputs[n]) / np.mean(throughputs["shaper"]) for n in names]
    plot_bar(labels, avg_tpt, "Normalized Throughput", save_path="%s/avg-tpt.pdf" %(output_dir))
            # plot_throughput(throughput_data, save_path=output_file_path)

rates = [1.5]
# Replace 'log' with the actual directory containing your log files
input_directories = ['logs-1_5mbps']
# Replace 'plots' with the desired directory to save the plots
output_directory = 'plots/all'

process_directory(input_directories, output_directory, rates)
