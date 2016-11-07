import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

vms = ["vm1","vm2","vm3","vm4"]
base_path = "measurements"
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
