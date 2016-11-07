from __future__ import division
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

if len(sys.argv) == 1:
    print("enter directory")
    sys.exit(1)

vms = ["vm1", "vm2", "vm3", "vm4", "vm5"]
disks = ["vda1"]
base_path = sys.argv[1]
ReadDiskActivity = []
WriteDiskActivity = []

for vm in vms:
    TotalReadDiskActivity = 0
    TotalWriteDiskActivity = 0
    for disk in disks:
        filename = disk+"."+vm+".txt"
        filepath = base_path + "/" + filename
        #print filepath
        diskStats = pd.read_csv(filepath, header=None, delim_whitespace=True)
        #print diskStats
        numReadsCompleted = diskStats[6][len(diskStats)-1] - diskStats[6][0]
        #print "numReadsCompleted = " + str(numReadsCompleted)
        numWritesCompleted = diskStats[10][len(diskStats)-1] - diskStats[10][0]
        #print "numWritesCompleted = " + str(numWritesCompleted)
        TotalReadDiskActivity = TotalReadDiskActivity + numReadsCompleted
        TotalWriteDiskActivity = TotalWriteDiskActivity + numWritesCompleted

        ReadDiskActivity.append(TotalReadDiskActivity)
        WriteDiskActivity.append(TotalWriteDiskActivity)


DiskActivity = [vms, ReadDiskActivity, WriteDiskActivity]
print DiskActivity

wlist1 = ReadDiskActivity
wlist2 = WriteDiskActivity
#wlist2 = [112648,48256,19039312,9504183,22872505]
wlist1 = [(x * 512) / (1024 * 1024) for x in wlist1]
wlist2 = [(x * 512) / (1024 * 1024) for x in wlist2]
print wlist1
print wlist2
#style.use('ggplot')
x1 = np.arange(len(wlist1))
fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1.set_title("Parameter Server- Per Node Disk I/O")
ax1.set_xlabel("VMs")
ax1.set_ylabel("Disk I/O (MB)")
rects1 = ax1.bar(x1-0.1, wlist1, width=0.2, color='r', align='center')
rects2 = ax1.bar(x1+0.1, wlist2, width=0.2, color='b', align='center')
#xlabels = ['', '12', '', '21', '', '50', '', '71', '', '85']
xlabels = ['', '1', '2', '3', '4', '5']
#ax1.set_xticklabels(xlabels, rotation='vertical')
ax1.set_xticklabels(xlabels)
#ax1.legend((rects1[0], rects2[0]), ('Unoptimized Binary', 'Optimized Binary'), bbox_to_anchor=(1.05, 1), loc='upper right', shadow=True)
ax1.legend((rects1[0], rects2[0]), ('Reads', 'Writes'), shadow=True, loc='upper right')
plt.show()
