#!/bin/bash

filepath=$1
labelspath=$2
outfile=$3

echo "Using first col of "${1}" to get labels"

cat ${filepath} | cut -d, -f1 > ${labelspath}
cut -d, -f1 --complement ${filepath} > ${outfile}