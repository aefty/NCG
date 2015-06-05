#include <iomanip>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>

#include "config.cpp"
#include "lib/cpu_sbrutns.cpp"
#include "lib/json.cpp"

using namespace std;


int main(int argc, char* argv[]) {

	if (argc > 1) { _GLB_N_ = _GLB_N_ * atoi(argv[1]); }

	if (argc > 2) { _GLB_ITR_ = _GLB_ITR_ * atoi(argv[2]); }

	if (argc > 3) { _GLB_ITR_LINE_ = _GLB_ITR_LINE_ * atoi(argv[3]); }

	if (argc > 4) { _GLB_EPS_ = _GLB_EPS_ * atoi(argv[4]); }

	//gpu::deviceSpecs();

	JSON json;

	vector<double> x0;
	GUESS(_GLB_N_, x0);

	vector<long int> C;
	C.reserve((_GLB_N_ * _GLB_N_ - _GLB_N_) / 2);

	cpu::funGraph(_GLB_N_, _GLB_EPS_, x0, C);



	int s = C.size();
	json.append("C", C);
	json.append("size", s);



	cout << "\n\n" << json.dump() << "\n\n";
	return 0;
}
