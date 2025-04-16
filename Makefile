.PHONY: full preprocess run clean

full: clean preprocess run

preprocess:
	RUSTFLAGS="-C target-cpu=native" cargo build --release
	g++ -O3 -DNDEBUG -std=c++11 -mfma -m64 main.cpp -L./target/release -lgemm_microbench -Wl,-rpath=../gemm_microbench/target/release -o gemm_microbench -lrt -I ../Eigen

run:
	./gemm_microbench

clean:
	cargo clean
	rm -rf gemm_microbench