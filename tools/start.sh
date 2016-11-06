#!/bin/bash

mkdir measurements

if [ "$?" -ne "0" ]; then
    exit -1;
fi

for i in {1..5}; do
    ssh vm-28-"$i" cat /proc/net/dev > measurements/start_net_vm$i;
    ssh vm-28-"$i" cat /proc/diskstats > measurements/start_disk_vm$i;
done
