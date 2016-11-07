#!/bin/bash

grep eth0: measurements/start_net_vm1 > measurements/eth0.vm1.txt
grep eth0: measurements/stop_net_vm1 >> measurements/eth0.vm1.txt

grep lo: measurements/start_net_vm1 > measurements/lo.vm1.txt
grep lo: measurements/stop_net_vm1 >> measurements/lo.vm1.txt

grep eth0: measurements/start_net_vm2 > measurements/eth0.vm2.txt
grep eth0: measurements/stop_net_vm2 >> measurements/eth0.vm2.txt

grep lo: measurements/start_net_vm2 > measurements/lo.vm2.txt
grep lo: measurements/stop_net_vm2 >> measurements/lo.vm2.txt

grep eth0: measurements/start_net_vm3 > measurements/eth0.vm3.txt
grep eth0: measurements/stop_net_vm3 >> measurements/eth0.vm3.txt

grep lo: measurements/start_net_vm3 > measurements/eth0.vm3.txt
grep lo: measurements/stop_net_vm3 >> measurements/eth0.vm3.txt

grep eth0: measurements/start_net_vm4 > measurements/eth0.vm4.txt
grep eth0: measurements/stop_net_vm4 >> measurements/eth0.vm4.txt

grep lo: measurements/start_net_vm4 > measurements/lo.vm4.txt
grep lo: measurements/stop_net_vm4 >> measurements/lo.vm4.txt

grep eth0: measurements/start_net_vm5 > measurements/eth0.vm5.txt
grep eth0: measurements/stop_net_vm5 >> measurements/eth0.vm5.txt

grep lo: measurements/start_net_vm5 > measurements/lo.vm5.txt
grep lo: measurements/stop_net_vm5 >> measurements/lo.vm5.txt
