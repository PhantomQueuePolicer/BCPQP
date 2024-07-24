# Burst Controlled Phantom Queue Policer (BC-PQP)

BC-PQP is an efficient and policy rich rate enforcement algorithm which has system efficiency comparable to a traffic policer and rate and policy enforcement guarantees comparable to traffic shaper. It allows networks operators, ISPs and cellular service providers to rate limit traffic aggregates while also implementing policies like per-flow fairness and prioritization within each traffic aggregate. Please read our [SIGCOMM'24 paper](https://ammart2.web.illinois.edu/files/phantom.pdf) for more details.

In this repository, we present two implementations of BC-PQP: one using DPDK (`testbed`) and another using Mahimahi (`emulation`). Please see README's in respective directories on details on how to use this codebase.

Please feel free to reach out at ammart2@illinois.edu if you have any questions.
