from __future__ import division
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

if len(sys.argv) == 1:
    print("enter directory")
    sys.exit(1)

vms = ["vm1","vm2","vm3","vm4", "vm5"]
base_path = sys.argv[1]
BytesReceived = []
BytesTransmitted = []

for vm in vms:
    TotalBytesReceived = 0
    TotalBytesTransmitted = 0
    filename = "eth0."+vm+".txt"
    filepath = base_path + "/" + filename
    #print filepath
    eth0Stats = pd.read_csv(filepath, header=None, delim_whitespace=True)
    # eth0Stats[2] = BytesReceived, eth0Stats[10] = Bytes Transmitted
    #print len(eth0Stats)
    numBytesReceived = eth0Stats[2][len(eth0Stats)-1] - eth0Stats[2][0]
    #print "numBytesReceived = " + str(numBytesReceived)
    numBytesTransmitted = eth0Stats[10][len(eth0Stats)-1] - eth0Stats[10][0]
    #print "numBytesTransmitted = " + str(numBytesTransmitted)
    TotalBytesReceived = TotalBytesReceived + numBytesReceived
    TotalBytesTransmitted = TotalBytesTransmitted + numBytesTransmitted
    BytesReceived.append(TotalBytesReceived)
    BytesTransmitted.append(TotalBytesTransmitted)

NetActivity = [vms, BytesReceived, BytesTransmitted]
print NetActivity

wlist1 = BytesReceived
wlist2 = BytesTransmitted
#wlist2 = [112648,48256,19039312,9504183,22872505]
wlist1 = [x / (1024 * 1024) for x in wlist1]
wlist2 = [x / (1024 * 1024) for x in wlist2]
print wlist1
print wlist2
#style.use('ggplot')
x1 = np.arange(len(wlist1))
fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1.set_title("Parameter Server - Network I/O")
ax1.set_xlabel("Nodes")
ax1.set_ylabel("Network I/O (MB)")
rects1 = ax1.bar(x1-0.1, wlist1, width=0.2, color='r', align='center')
rects2 = ax1.bar(x1+0.1, wlist2, width=0.2, color='b', align='center')
#xlabels = ['', '12', '', '21', '', '50', '', '71', '', '85']
xlabels = ['', '1', '2', '3', '4', '5']
#ax1.set_xticklabels(xlabels, rotation='vertical')
ax1.set_xticklabels(xlabels)
#ax1.legend((rects1[0], rects2[0]), ('Unoptimized Binary', 'Optimized Binary'), bbox_to_anchor=(1.05, 1), loc='upper right', shadow=True)
ax1.legend((rects1[0], rects2[0]), ('Received', 'Transmitted'), shadow=True, loc='upper right')
plt.show()
