.PHONY: all preprocess run clean postprocess

SHELL := /bin/bash

THREADS ?= $(shell nproc)

CXX = g++
CXXFLAGS = -O3 -DNDEBUG -std=c++14 -mfma -m64 -DEIGEN_USE_THREADS -DTHREADS=$(THREADS)
INCLUDES = -I ../Eigen -I ../benchmark/include
LDFLAGS = -L./target/release -L/usr/local/lib -Wl,-rpath=../gemm_microbench/target/release
LDLIBS = -lgemm_microbench -lbenchmark -lpthread -lrt

all: clean preprocess run postprocess

preprocess:
	sudo cpupower frequency-set -g performance
	RUSTFLAGS="-C target-cpu=native" cargo build --release
	$(CXX) $(CXXFLAGS) main.cpp $(INCLUDES) $(LDFLAGS) $(LDLIBS) -o gemm_microbench

run:
	setarch "$(shell uname -m)" -R ./gemm_microbench

postprocess:
	sudo cpupower frequency-set -g powersave
	sudo sysctl -w kernel.randomize_va_space=2

clean:
	cargo clean
	rm -rf gemm_microbench