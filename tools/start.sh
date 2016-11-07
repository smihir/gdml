#!/bin/bash

if [[ -z "$1" ]] ; then
    echo 'enter directory'
    exit -1
fi

mkdir "$1"

if [ "$?" -ne "0" ]; then
    exit -1;
fi

for i in {1..5}; do
    ssh vm-28-"$i" cat /proc/net/dev > "$1"/start_net_vm$i;
    ssh vm-28-"$i" cat /proc/diskstats > "$1"/start_disk_vm$i;
done
