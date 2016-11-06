#!/bin/bash

mkdir measurements

for i in {1..5}; do
    ssh vm-28-"$i" cat /proc/net/dev > measurements/stop_net_vm$i;
    ssh vm-28-"$i" cat /proc/diskstats > measurements/stop_disk_vm$i;
done
