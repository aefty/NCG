all: ncg test

ncg: src/ncg.cu
	nvcc -g -O3 -std=c++11 -arch=sm_30 $^ -o bin/ncg

test: src/test.cu
	nvcc -g -O3 -std=c++11 -arch=sm_30 $^ -o bin/test

clean:
	rm -rf bin/*.o bin/ncg bin/test