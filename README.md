# Geo Distributed Machine Learning using Parameter Servers
For motivation behind this project please see the [wiki](https://github.com/smihir/gdml/wiki).

We are using the [MXNet](https://github.com/dmlc/mxnet) framework for evaluating the support for GDML added in [ps-lite](https://github.com/dmlc/ps-lite) parameter server. Architecutre docs and evaluation of our work will be added in the wiki as we make progress.

## Compiling
MXNet support Linux, OSX and Windows. But, the top-level makefile for GDML is written to work only on Ubuntu.
On a Ubuntu VM, Machine GDML can be downloaded and installed by following the instructions below:
```
sudo apt-get install git
make
make install
```

All the required dependencies will be installed by the makefile.

## Testing the installation
The installation can be tested by running the example present in the mxnet directory
```
cd example
python train_mnist.py
```

## Distributed Learning
```
cd example
echo <NIC ip> > hosts # just for testing add NICs ip
../mxnet/tools/launch.py -n 1 --launcher ssh -H hosts python train_mnist.py --kv-store dist_sync
```
If running on 2 or more nodes add IP for each node in the **hosts** file. Make sure that you can enable
passwordless ssh to each host. Each host should have MXNet installed and the dataset should be in
a shared location with same relative path on all the nodes. We are using NFS to share the dataset,
NFS installation instructions on Ubuntu are [here](https://help.ubuntu.com/community/SettingUpNFSHowTo).
