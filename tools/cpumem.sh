#!/bin/bash

while true; do
    echo "Start Epoch";
    ps -C "python" -L -o pid=,tid=,%mem=,vsz=,%cpu=,psr=,command=;
    sleep 1;
    echo "End Epoch";
done
