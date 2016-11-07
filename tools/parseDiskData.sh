#!/bin/bash

if [[ -z "$1" ]] ; then
    echo 'enter directory'
    exit -1
fi

grep vda1 "$1"/start_disk_vm1 > "$1"/vda1.vm1.txt
grep vda1 "$1"/stop_disk_vm1 >> "$1"/vda1.vm1.txt

grep vda1 "$1"/start_disk_vm2 > "$1"/vda1.vm2.txt
grep vda1 "$1"/stop_disk_vm2 >> "$1"/vda1.vm2.txt

grep vda1 "$1"/start_disk_vm3 > "$1"/vda1.vm3.txt
grep vda1 "$1"/stop_disk_vm3 >> "$1"/vda1.vm3.txt

grep vda1 "$1"/start_disk_vm4 > "$1"/vda1.vm4.txt
grep vda1 "$1"/stop_disk_vm4 >> "$1"/vda1.vm4.txt

grep vda1 "$1"/start_disk_vm5 > "$1"/vda1.vm5.txt
grep vda1 "$1"/stop_disk_vm5 >> "$1"/vda1.vm5.txt
