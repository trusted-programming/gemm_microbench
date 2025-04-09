.PHONY: full preprocess run clean

full: clean preprocess run

preprocess:
	cargo build --release
	g++ main.cpp -L./target/release -lmatmul_microbench -Wl,-rpath=../matmul_microbench/target/release -o matmul_bench -ldl

run:
	./matmul_bench

clean:
	cargo clean
	rm -rf matmul_bench