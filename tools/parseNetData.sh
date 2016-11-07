#!/bin/bash

if [[ -z "$1" ]] ; then
    echo 'enter directory'
    exit -1
fi

grep eth0: "$1"/start_net_vm1 > "$1"/eth0.vm1.txt
grep eth0: "$1"/stop_net_vm1 >> "$1"/eth0.vm1.txt

grep lo: "$1"/start_net_vm1 > "$1"/lo.vm1.txt
grep lo: "$1"/stop_net_vm1 >> "$1"/lo.vm1.txt

grep eth0: "$1"/start_net_vm2 > "$1"/eth0.vm2.txt
grep eth0: "$1"/stop_net_vm2 >> "$1"/eth0.vm2.txt

grep lo: "$1"/start_net_vm2 > "$1"/lo.vm2.txt
grep lo: "$1"/stop_net_vm2 >> "$1"/lo.vm2.txt

grep eth0: "$1"/start_net_vm3 > "$1"/eth0.vm3.txt
grep eth0: "$1"/stop_net_vm3 >> "$1"/eth0.vm3.txt

grep lo: "$1"/start_net_vm3 > "$1"/eth0.vm3.txt
grep lo: "$1"/stop_net_vm3 >> "$1"/eth0.vm3.txt

grep eth0: "$1"/start_net_vm4 > "$1"/eth0.vm4.txt
grep eth0: "$1"/stop_net_vm4 >> "$1"/eth0.vm4.txt

grep lo: "$1"/start_net_vm4 > "$1"/lo.vm4.txt
grep lo: "$1"/stop_net_vm4 >> "$1"/lo.vm4.txt

grep eth0: "$1"/start_net_vm5 > "$1"/eth0.vm5.txt
grep eth0: "$1"/stop_net_vm5 >> "$1"/eth0.vm5.txt

grep lo: "$1"/start_net_vm5 > "$1"/lo.vm5.txt
grep lo: "$1"/stop_net_vm5 >> "$1"/lo.vm5.txt
