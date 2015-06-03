all: ncg test

ncg: src/ncg.cpp
	g++ -g  -std=c++11 -O3 $^ -o bin/ncg

test: src/test.cpp
	g++ -g  -std=c++11 -O3 $^ -o bin/test

clean:
	rm -rf bin/*.o bin/ncg bin/test

