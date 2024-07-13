
'''
Configure the list of flows, where each flow is a tuple as follows:

flow = (port, 
    start time, 
    congestion control protocol, 
    bytes to send, 
    rtt, 
    time period between flow repeats, 
    number of times flow should be repeated
)
'''


cc = ["reno", "cubic", "bbr", "vegas"]
rtt = [1,2,3]
port_start = 2049
multipler = 100
m2 = 1
num_shapers = 30

flows = []

# mix CCs mix RTTs
# only backlogged flows
for s in range(int(num_shapers/6)):
    flows.append((rtt[s % 3]+port_start+64*s, 0, cc[s%len(cc)], multipler*(3*10**6), rtt[s % 3], 5, 1))
    flows.append((rtt[(s+1) % 3]+port_start+64*s, 0, cc[(s+1)%len(cc)], multipler*(3*10**6), rtt[(s+1) % 3], 5, 1))
    flows.append((rtt[(s+2) % 3]+port_start+64*s, 0, cc[(s+2)%len(cc)], multipler*(3*10**6), rtt[(s+2) % 3], 5, 1))

# only short bursty flows
for s in range(int(num_shapers/6), int(2*num_shapers/6)):
    flows.append((rtt[s % 3]+port_start+64*s, 60, cc[(s+1)%len(cc)], multipler*m2*(1*10**5), rtt[(s+1) % 3], 6, 5))
    flows.append((rtt[(s+1) % 3]+port_start+64*s, 60, cc[(s+2)%len(cc)], multipler*m2*(1*10**5), rtt[(s+2) % 3], 6, 5))
    flows.append((rtt[(s+2) % 3]+port_start+64*s, 60, cc[(s+3)%len(cc)], multipler*m2*(1*10**5), rtt[(s+3) % 3], 6, 5))

# mix flows
for s in range(int(2*num_shapers/6), int(3*num_shapers/6)):
    flows.append((rtt[s % 3]+port_start+64*s, 120, cc[(s+1)%len(cc)], multipler*(3*10**6), rtt[(s+1) % 3], 5, 1))
    flows.append((rtt[(s+1) % 3]+port_start+64*s, 120, cc[(s+2)%len(cc)], multipler*(3*10**6), rtt[(s+2) % 3], 5, 1))
    flows.append((rtt[(s+2) % 3]+port_start+64*s, 120, cc[(s+3)%len(cc)], multipler*m2*(1*10**5), rtt[(s+3) % 3], 6, 5))


# same CCs same RTTs
# only backlogged flows
for s in range(int(3*num_shapers/6), int(4*num_shapers/6)):
    flows.append((rtt[s % 3]+port_start+64*s, 180, cc[s%len(cc)], multipler*(3*10**6), rtt[s % 3], 5, 1))
    flows.append((1+rtt[s % 3]+port_start+64*s, 180, cc[s%len(cc)], multipler*(3*10**6), rtt[s % 3], 5, 1))
    flows.append((2+rtt[s % 3]+port_start+64*s, 180, cc[s%len(cc)], multipler*(3*10**6), rtt[s % 3], 5, 1))

# only short bursty flows
for s in range(int(4*num_shapers/6), int(5*num_shapers/6)):
    flows.append((rtt[s % 3]+port_start+64*s, 240, cc[s%len(cc)], multipler*m2*(1*10**5), rtt[s % 3], 6, 5))
    flows.append((1+rtt[s % 3]+port_start+64*s, 240, cc[s%len(cc)], multipler*m2*(1*10**5), rtt[s % 3], 6, 5))
    flows.append((2+rtt[s % 3]+port_start+64*s, 240, cc[s%len(cc)], multipler*m2*(1*10**5), rtt[s % 3], 6, 5))

# mix flows
for s in range(int(5*num_shapers/6), int(6*num_shapers/6)):
    flows.append((rtt[s % 3]+port_start+64*s, 300, cc[s%len(cc)], multipler*m2*(3*10**6), rtt[s % 3], 5, 1))
    flows.append((1+rtt[s % 3]+port_start+64*s, 300, cc[s%len(cc)], multipler*m2*(3*10**6), rtt[s % 3], 5, 1))
    flows.append((2+rtt[s % 3]+port_start+64*s, 300, cc[s%len(cc)], multipler*m2*(1*10**5), rtt[s % 3], 6, 5))
