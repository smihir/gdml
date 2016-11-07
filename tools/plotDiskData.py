import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

vms = ["vm1", "vm2", "vm3", "vm4", "vm5"]
disks = ["vda1"]
base_path = "measurements"
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
