#!/bin/bash

if [[ -z "$1" ]] ; then
    DISK="sdb1"
    echo 'using default disk:' $DISK
fi

if [[ -z "$2" ]] ; then
    IF="eth0"
    echo 'using default interface:' $IF
fi

LOGSDIR="sslogs"

mkdir -p $LOGSDIR
rm -rf $LOGSDIR/*

TICKS=0
while true; do
    cat /proc/net/dev | egrep $IF >> $LOGSDIR/$IF.$HOSTNAME.txt;
    cat /proc/diskstats | egrep "$DISK " >> $LOGSDIR/$DISK.$HOSTNAME.txt;
    cat /proc/stat | egrep 'cpu ' >> $LOGSDIR/cpu.$HOSTNAME.txt;
    printf "\rTime $TICKS s";
    sleep 1;
    TICKS=$((TICKS + 1))
done
