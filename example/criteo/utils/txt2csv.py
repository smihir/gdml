#!/usr/bin/env python

import os, sys

usage_string = 'usage: txt2csv.py input output'

if len(sys.argv) != 3:
    print(usage_string)
    exit(1)

src_path, dst_path = sys.argv[1:]

with open(dst_path, 'w') as f:
    for line in open(src_path, 'r'):
        line = line.replace('\t', ',')
        f.write(line.replace('\n', '\r\n'))
