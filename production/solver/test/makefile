all: t0 t1 t2

t1: async.cu
	nvcc -g -O3 -std=c++11 -arch=sm_30 $^ -o t1

t1: async2.cu
	nvcc -g -O3 -std=c++11 -arch=sm_30 $^ -o t1

clean:
	rm -rf *.o t1 t2