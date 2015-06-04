
#include <iomanip>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include <vector>
#include <algorithm>

static inline uint32 FloatFlip(uint32 f) {
	uint32 mask = -int32(f >> 31) | 0x80000000;
	return f ^ mask;
}

static inline uint32 IFloatFlip(uint32 f) {
	uint32 mask = ((f >> 31) - 1) | 0x80000000;
	return f ^ mask;
}


int main(int argc, char const* argv[]) {
	float t = 1.23;
	int k = FloatFlip(t);

	cout << k;

	return 0;
}