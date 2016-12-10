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
disks = ["sdb1"]
base_path = sys.argv[1]

for vm in vms:
    ReadDiskActivity = list()
    WriteDiskActivity = list()
    lr = 0
    lw = 0
    cr = 0
    cw = 0
    initwrites = 0
    initreads = 0
    totaltime = 0
    for disk in disks:
        filename = disk+"."+vm+".txt"
        filepath = base_path + "/" + filename
        with open(filepath) as f:
            csvreader = csv.reader(f, delimiter=' ')
            for stat in csvreader:
                cr = int(stat[13])
                cw = int(stat[17])
                lr = cr if lr == 0 else lr
                lw = cw if lw == 0 else lw
                initreads = cr if initreads == 0 else initreads
                initwrites = cw if initwrites == 0 else initwrites
                ReadDiskActivity.append(cr - lr)
                WriteDiskActivity.append(cw - lw)
                lr = cr
                lw = cw
                totaltime += 1


    print("{}: total reads: {}, total writes {} (MB), time {}".format(vm,
        ((lr - initreads) * 512) / (1024 * 1024),
        ((lw - initwrites) * 512) / (1024 * 1024)), totaltime)

    rlist = ReadDiskActivity
    wlist = WriteDiskActivity

    rlist = [(x * 512) / (1024 * 1024) for x in rlist]
    wlist = [(x * 512) / (1024 * 1024) for x in wlist]

    x1 = np.arange(len(wlist))
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.set_title("%s - Disk I/O vs Time" % vm)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Disk I/O (MB/s)")

    rects1 = ax1.plot(x1,rlist)
    rects2 = ax1.plot(x1,wlist)
    ax1.legend((rects1[0], rects2[0]), ('Reads', 'Writes'), shadow=True, loc='upper right')
plt.show()
