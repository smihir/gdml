#!/bin/bash

grep vda1 measurements/start_disk_vm1 > measurements/vda1.vm1.txt
grep vda1 measurements/stop_disk_vm1 >> measurements/vda1.vm1.txt

grep vda1 measurements/start_disk_vm2 > measurements/vda1.vm2.txt
grep vda1 measurements/stop_disk_vm2 >> measurements/vda1.vm2.txt

grep vda1 measurements/start_disk_vm3 > measurements/vda1.vm3.txt
grep vda1 measurements/stop_disk_vm3 >> measurements/vda1.vm3.txt

grep vda1 measurements/start_disk_vm4 > measurements/vda1.vm4.txt
grep vda1 measurements/stop_disk_vm4 >> measurements/vda1.vm4.txt

grep vda1 measurements/start_disk_vm5 > measurements/vda1.vm5.txt
grep vda1 measurements/stop_disk_vm5 >> measurements/vda1.vm5.txt
