from __future__ import division
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

if len(sys.argv) == 1:
    print("enter directory")
    sys.exit(1)

vms = ["node1", "node2", "node3", "node4", "node5"]
cpus = ["cpu"]
base_path = sys.argv[1]
for vm in vms:
    Cpu = []

    luser = 0
    lsystem = 0
    lidle = 0
    lirq = 0
    lsoftirq = 0
    cuser = 0
    csystem = 0
    cidle = 0
    cirq = 0
    csoftirq = 0
    totaltime = 0

    inituser = 0
    initsystem = 0
    initidle = 0
    initirq = 0
    initsoftirq = 0
    for cpu in cpus:
        filename = cpu+"."+vm+".txt"
        filepath = base_path + "/" + filename
        with open(filepath) as f:
            csvreader = csv.reader(f, delimiter=' ')
            for stat in csvreader:
                cuser = int(stat[2])
                csystem = int(stat[4])
                cidle = int(stat[5])
                cirq = int(stat[7])
                csoftirq = int(stat[8])
                luser = cuser if luser == 0 else luser
                lsystem = csystem if lsystem == 0 else lsystem
                lidle = cidle if lidle == 0 else lidle
                lirq = cirq if lirq == 0 else lirq
                lsoftirq = csoftirq if lsoftirq == 0 else lsoftirq

                inituser = cuser if inituser == 0 else inituser
                initsystem = csystem if initsystem == 0 else initsystem
                initidle = cidle if initidle == 0 else initidle
                initirq = cirq if initirq == 0 else initirq
                initsoftirq = csoftirq if initsoftirq == 0 else initsoftirq

                userps = cuser - luser
                systemps = csystem - lsystem
                idleps = cidle - lidle
                irqps = cirq - lirq
                softirqps = csoftirq - lsoftirq

                totalps = userps + systemps + idleps + irqps + softirqps
                totalps = 1 if totalps == 0 else totalps
                percentact = ((userps + systemps + irqps + softirqps) / totalps) * 100
                Cpu.append(percentact)
                totaltime += 1


        clist = Cpu

        x1 = np.arange(len(clist))
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(1, 1, 1)
        ax1.set_title("%s - CPU Usage vs Time" % vm)
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("CPU used (%)")

        rects1 = ax1.plot(x1,clist)
        plt.ylim([0, 100])
        plt.yticks(np.arange(0, 100, 10))
plt.show()
