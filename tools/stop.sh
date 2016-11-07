#!/bin/bash

if [[ -z "$1" ]] ; then
    echo 'enter directory'
    exit -1
fi

for i in {1..5}; do
    ssh vm-28-"$i" cat /proc/net/dev > "$1"/stop_net_vm$i;
    ssh vm-28-"$i" cat /proc/diskstats > "$1"/stop_disk_vm$i;
done
