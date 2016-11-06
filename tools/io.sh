#!/bin/bash
#sudo iotop -batqqqk -p $(pidof python | tr ' ' ',')
sudo iotop -batqqqk | grep python
