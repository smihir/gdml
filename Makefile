CPUS := $(shell getconf _NPROCESSORS_ONLN)
ROOT := $(shell pwd)
OS := $(shell lsb_release -si)
ARCH := $(shell uname -m | sed 's/x86_//;s/i[3-6]86/32/')
VER := $(shell lsb_release -sr)

.bootstrap: .check
	@echo "Installing build-essential, libatlas-base-dev, libopencv-dev..."
	@sudo apt-get update
	@sudo apt-get install -y build-essential libatlas-base-dev libopencv-dev \
	    python-numpy python-setuptools unzip
	@touch .bootstrap

.PHONY: mxnet all
all: mxnet

mxnet: .check .bootstrap
	cd mxnet; make -j $(CPUS)

clean:
	cd mxnet; make clean

install: .bootstrap
	cd mxnet/python; sudo python setup.py install

.check:
ifneq ($(OS), Ubuntu)
	$(error Only Ubuntu supported, detected OS is $(OS))
endif
ifeq ($(ARCH),)
	$(error Only x86 supported, detected arch is $(shell uname -m))
endif
	touch .check

.DEFAULT_GOAL:=all
