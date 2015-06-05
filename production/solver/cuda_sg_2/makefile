all: ncg test tt

ncg: src/ncg.cu
	nvcc -g -O3 -std=c++11 -arch=sm_30 $^ -o bin/ncg

test: src/test.cu
	nvcc -g -O3 -std=c++11 -arch=sm_30 $^ -o bin/test

tt: src/tt.cpp
	g++ -g -std=c++11 -O3 $^ -o bin/tt

clean:
	rm -rf bin/*.o bin/ncg bin/test bin/tt